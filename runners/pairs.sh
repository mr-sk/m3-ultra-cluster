#!/bin/bash
# Agent runs on the cluster with OpenCode (opencode_run.sh), two at a time on
# --concurrent 2 (default) or one at a time (SIDES=a), each run starting from a clean repo.
#
# Required:
#   TASK=task.txt       the prompt given to every run
#   REPO_A=path         git repo the "a" runs work in (reset before each run)
#   REPO_B=path         a second copy of it for the "b" runs (two at a time only)
#   RULES=rules.txt     proxy rules during the runs
#   OUT=dir             where opencode_run.sh writes each run's files
# Optional:
#   PAIRS=3  PREFIX=c  SIDES="a b"  LIMIT=2400 (seconds per pair, then the runs are killed)
#   TEST_CMD="swift test"   passed to opencode_run.sh
#   TEST_SUMMARY=regex      the test-output line to print per run (default: XCTest's summary)
# Ends with the everyday server ($EVERYDAY_FLAGS), the everyday proxy ($EVERYDAY_RULES)
# and both repos reset. Settings: see cluster_env.sh.
. "$(dirname "$0")/cluster_env.sh"
: "${TASK:?}" "${REPO_A:?}" "${RULES:?}" "${OUT:?}"
PAIRS=${PAIRS:-3}; PREFIX=${PREFIX:-c}; LIMIT=${LIMIT:-2400}; SIDES=${SIDES:-"a b"}
case " $SIDES " in *" b "*) : "${REPO_B:?REPO_B is needed for two at a time}";; esac
mkdir -p "$OUT"

cleanup() {
  echo "--- restoring the everyday server ($EVERYDAY_FLAGS) and proxy"
  restart_server "$EVERYDAY_FLAGS"
  [ -n "$EVERYDAY_RULES" ] && restart_proxy "$EVERYDAY_RULES"
  git -C "$REPO_A" checkout -q -- .
  [ -n "$REPO_B" ] && git -C "$REPO_B" checkout -q -- .
}
trap cleanup EXIT
restart_proxy "$RULES"
restart_server "--concurrent 2"

for k in $(seq 1 "$PAIRS"); do
  echo "=== pair $k: $(date +%H:%M:%S)"
  t0=$(date +%s); pids=""
  for side in $SIDES; do
    repo=$([ "$side" = a ] && echo "$REPO_A" || echo "$REPO_B")
    ( ALLOW_PARALLEL=1 RESET=1 TEST_CMD="${TEST_CMD:-swift test}" "$REPO_ROOT/opencode_run.sh" "$repo" "$TASK" "$OUT" "$PREFIX$side$k" \
        > "$OUT/$PREFIX$side$k.out" 2>&1 ) &
    pids="$pids $!"
    sleep 3
  done
  # Wait for the agents only (a bare `wait` would also wait for the proxy, forever), with a hard limit.
  for i in $(seq 1 "$LIMIT"); do
    alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break; sleep 1
  done
  if [ $alive = 1 ]; then echo "TIMEOUT: pair $k still running after ${LIMIT}s, killing"; pkill -f "opencode run --standalone"; sleep 2; fi
  echo "pair $k wall: $(( ($(date +%s) - t0) / 60 )) min $(( ($(date +%s) - t0) % 60 )) s"
  for side in $SIDES; do
    n=$PREFIX$side$k
    echo "-- $n: $(grep -E '^(start|exit|steps)' "$OUT/$n.out" | tr '\n' ' ')"
    grep -E "${TEST_SUMMARY:-Executed [0-9]+ tests, with|tests? passed|tests? failed}" "$OUT/$n.test.txt" 2>/dev/null | tail -1
  done
done
ssh "$HEAD" 'grep -E "Traceback" ~/dsv4-server.log | head -5; echo "requests: $(grep -c chatcmpl ~/dsv4-server.log)"'
