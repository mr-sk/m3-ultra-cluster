#!/bin/bash
# Cluster health watch, run every 2 minutes on the client Mac by a LaunchAgent (install_health.sh).
#
# Checks: the server answers /health; the proxy on this Mac listens (restarted via its
# LaunchAgent if installed); both nodes are reachable; on each node the GPU wired limit,
# the Thunderbolt IP, both RDMA links and free memory; new tracebacks in the server log.
# Alerts: a macOS notification when the state changes (problem found, problem changed, healthy
# again), and a line in ~/dsv4-logs/health.log.
# Recovery: only a dead server is restarted, with $EVERYDAY_FLAGS, and only after 3 failed checks
# in a row, when the wired limit, Thunderbolt IPs and RDMA links are fine (otherwise a reboot
# needs the README's "Before any cluster run" steps, which take sudo: it says so instead), at
# most once per 30 minutes, and never while a benchmark (~/mtp-bench) is running.
# `health.sh pause` / `health.sh resume` turn alerts and restarts off and on (e.g. while you stop
# the server on purpose).
# Settings: see cluster_env.sh. Optional: WIRED_WANT (default 86016), MIN_FREE_PCT (default 5),
# DRY_RUN=1 (log the restart it would do, don't do it), STATE_DIR (default ~/.cache/m3-cluster).
. "$(dirname "$0")/cluster_env.sh"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
STATE_DIR=${STATE_DIR:-$HOME/.cache/m3-cluster}; mkdir -p "$STATE_DIR" "$HOME/dsv4-logs"
LOG=$HOME/dsv4-logs/health.log
WIRED_WANT=${WIRED_WANT:-86016}; MIN_FREE_PCT=${MIN_FREE_PCT:-5}
now=$(date +%s); stamp=$(date "+%Y-%m-%d %H:%M:%S")
case "$1" in  # health.sh pause | resume: stop/start alerts and restarts (e.g. while you use the memory)
  pause) touch "$STATE_DIR/paused"; echo "$stamp paused by hand" >> "$LOG"; echo paused; exit 0;;
  resume) rm -f "$STATE_DIR/paused"; echo "$stamp resumed" >> "$LOG"; echo resumed; exit 0;;
esac
[ -f "$STATE_DIR/paused" ] && exit 0
get() { cat "$STATE_DIR/$1" 2>/dev/null || echo "$2"; }
put() { echo "$2" > "$STATE_DIR/$1"; }
notify() {  # macOS notification; DRY_RUN only logs it
  if [ -n "$DRY_RUN" ]; then echo "$stamp (dry run) would notify: $1: $2" >> "$LOG"; return; fi
  osascript -e "display notification \"$2\" with title \"M3 cluster: $1\"" >/dev/null 2>&1; }

http=$(curl -s -m 10 -o /dev/null -w "%{http_code}" "$SERVER_URL/health")
SSH="ssh -o ConnectTimeout=8 -o BatchMode=yes"
nodes=$($SSH "$HEAD" "~/m3_node_status.sh; $SSH $PEER '~/m3_node_status.sh' || echo 'host=peer unreachable=1'; echo tracebacks=\$(grep -c Traceback ~/dsv4-server.log 2>/dev/null)" 2>/dev/null) \
  || nodes="host=head unreachable=1"

problems=(); prereq_ok=1; server_down=0; bench=0
while read -r line; do
  [ -z "$line" ] && continue
  case $line in tracebacks=*) tb=${line#tracebacks=}; continue;; esac
  unset host wired rdma rdma_want tbip server free unreachable
  eval "$line"  # trusted: our own node_status.sh output
  if [ -n "$unreachable" ]; then problems+=("$host not reachable over ssh"); prereq_ok=0; continue; fi
  [ "$wired" != "$WIRED_WANT" ] && { problems+=("$host: GPU wired limit is $wired, want $WIRED_WANT (after a reboot: README 'Before any cluster run')"); prereq_ok=0; }
  [ "$tbip" = none ] && { problems+=("$host: no Thunderbolt IP on en4 (after a reboot: README 'Before any cluster run')"); prereq_ok=0; }
  [ "$rdma" -lt "$rdma_want" ] 2>/dev/null && { problems+=("$host: $rdma of $rdma_want RDMA links active"); prereq_ok=0; }
  [ "$server" = 0 ] && server_down=1
  [ "$bench" -gt 0 ] 2>/dev/null && bench=1
  [ "$free" != "?" ] && [ "$free" -lt "$MIN_FREE_PCT" ] 2>/dev/null && problems+=("$host: only $free% memory free")
done <<< "$nodes"

# the recording proxy OpenCode talks to (this Mac)
if ! nc -z -G 3 127.0.0.1 "$PROXY_PORT" 2>/dev/null; then
  if [ -f "$HOME/Library/LaunchAgents/com.m3cluster.proxy.plist" ] && [ -z "$DRY_RUN" ]; then
    launchctl kickstart -k "gui/$(id -u)/com.m3cluster.proxy" 2>/dev/null; sleep 3
  fi
  nc -z -G 3 127.0.0.1 "$PROXY_PORT" 2>/dev/null \
    || problems+=("proxy not listening on 127.0.0.1:$PROXY_PORT (OpenCode can't reach the cluster; runners/install_proxy.sh)")
fi

prev_tb=$(get tracebacks "${tb:-0}")
if [ -n "$tb" ] && [ "$tb" -gt "$prev_tb" ] 2>/dev/null; then problems+=("new traceback in the server log on $HEAD"); fi
[ -n "$tb" ] && put tracebacks "$tb"

if [ "$bench" = 1 ]; then  # a benchmark has the cluster: don't alert or restart
  [ "$(get mode ok)" != bench ] && echo "$stamp benchmark running: watch paused" >> "$LOG"
  put mode bench; put fails 0; exit 0
fi
put mode watch

down=0
if [ "$http" != 200 ] || [ "$server_down" = 1 ]; then down=1; problems+=("server not answering (HTTP $http)"); fi
fails=$(get fails 0)
if [ "$down" = 1 ]; then fails=$((fails + 1)); else fails=0; fi
put fails "$fails"

if [ "$down" = 1 ] && [ "$fails" -ge 3 ] && [ "$prereq_ok" = 1 ] && [ $((now - $(get last_restart 0))) -ge 1800 ]; then
  put last_restart "$now"
  echo "$stamp restarting the server ($EVERYDAY_FLAGS) after $fails failed checks" >> "$LOG"
  notify "restarting" "Server down for $fails checks; restarting it with $EVERYDAY_FLAGS"
  if [ -n "$DRY_RUN" ]; then out="(dry run: not restarted)"; else out=$("$RUNNERS/restart.sh" $EVERYDAY_FLAGS 2>&1 | tail -1); fi
  echo "$stamp restart: $out" >> "$LOG"
  put fails 0
  exit 0
fi

status=$( [ ${#problems[@]} -eq 0 ] && echo OK || printf '%s; ' "${problems[@]}")
prev=$(get status OK)
if [ "$status" != "$prev" ]; then
  echo "$stamp $status" >> "$LOG"
  if [ "$status" = OK ]; then notify "healthy again" "All checks pass"; else notify "problem" "${status:0:200}"; fi
  put status "$status"
elif [ $((now - $(get last_beat 0))) -ge 3600 ]; then
  echo "$stamp still $( [ "$status" = OK ] && echo OK || echo "failing: $status")" >> "$LOG"; put last_beat "$now"
fi
