# Shared settings and helpers for the runners in this directory. Source it; don't run it.
#
# Machine-specific values live outside the repo, in ~/.config/m3-cluster.env (or the
# file named by $CLUSTER_ENV), for example:
#   HEAD=studio-a                 # ssh host of rank 0 (it runs the HTTP server)
#   PEER=10.0.0.2                 # rank 1, as reached from HEAD by ssh
#   SERVER_URL=http://10.0.0.1:8090
#   MODEL_DIR='$HOME/models/DeepSeek-V4-Flash-0731-4bit'   # expanded on HEAD
# Optional:
#   EVERYDAY_FLAGS="--concurrent 2 --copy-spec"   # server flags restored after a test
#   EVERYDAY_RULES=~/dsv4-proxy/working_rules.txt  # proxy rules restored after a test
#   PROXY_LOG=~/dsv4-proxy/proxy.log  PROXY_PORT=8091
#   PROXY_SCRIPT=~/dsv4-proxy/record_proxy.py      # default: the repo's copy

RUNNERS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$RUNNERS")
CONF=${CLUSTER_ENV:-$HOME/.config/m3-cluster.env}
[ -f "$CONF" ] && . "$CONF"
: "${HEAD:?set HEAD in $CONF}" "${PEER:?set PEER in $CONF}" "${SERVER_URL:?set SERVER_URL in $CONF}"
MODEL_DIR=${MODEL_DIR:-'$HOME/models/DeepSeek-V4-Flash-0731-4bit'}
EVERYDAY_FLAGS=${EVERYDAY_FLAGS:---concurrent 2 --copy-spec}
PROXY_PORT=${PROXY_PORT:-8091}
PROXY_LOG=${PROXY_LOG:-$HOME/dsv4-proxy/proxy.log}
PROXY_SCRIPT=${PROXY_SCRIPT:-$REPO_ROOT/record_proxy.py}

# restart_server "FLAGS": stop the server on both ranks, keep the old log, start it with
# FLAGS, and wait until it serves. The 15 s pause avoids JACCL's "queue pair RTR errno 14"
# when the ranks start again too soon after stopping.
restart_server() {
  ssh "$HEAD" "pkill -f vlm_cluster_server.py; for i in \$(seq 1 30); do pgrep -f vlm_cluster_server.py >/dev/null || break; sleep 1; done; ssh $PEER 'for i in \$(seq 1 30); do pgrep -f vlm_cluster_server.py >/dev/null || break; sleep 1; done'; mkdir -p ~/dsv4-logs/server; cp ~/dsv4-server.log ~/dsv4-logs/server/dsv4-server.\$(date +%Y%m%d-%H%M%S).log 2>/dev/null; sleep 15; nohup ~/cluster-env/bin/python ~/cluster_launch.py --backend jaccl --hostfile ~/m3-ultra-jaccl.json -- \$HOME/vlm-env/bin/python \$HOME/vlm_cluster_server.py --model $MODEL_DIR --port 8090 $1 > ~/dsv4-server.log 2>&1 < /dev/null &"
  for i in $(seq 1 90); do ssh "$HEAD" 'grep -qE "serving on|Traceback" ~/dsv4-server.log' && break; sleep 10; done
  ssh "$HEAD" 'grep -E "serving on|Traceback" ~/dsv4-server.log | tail -1'
}

# restart_proxy RULES_FILE: (re)start record_proxy.py on this machine with those rules.
restart_proxy() {
  local pid; pid=$(lsof -nP -t -iTCP:"$PROXY_PORT" -sTCP:LISTEN)
  [ -n "$pid" ] && kill $pid && sleep 1
  nohup python3 "$PROXY_SCRIPT" --port "$PROXY_PORT" --upstream "$SERVER_URL" --rules "$1" --fill-edit-path \
    > "$PROXY_LOG" 2>&1 < /dev/null &
  disown 2>/dev/null  # it outlives this script; and no "Terminated" notice when it is replaced
  sleep 2; tail -1 "$PROXY_LOG"
}
