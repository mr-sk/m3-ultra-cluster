#!/bin/bash
# Live check of --concurrent 2 against normal mode, with two recorded agent requests
# (record_proxy.py's *.request.json), greedy, 96 tokens each:
#   1. normal mode, one at a time (the reference)
#   2. --concurrent 2, one at a time
#   3. --concurrent 2, both at once
# and reports whether 2 and 3 produced exactly the reference output.
# usage: runners/live_test.sh REQUEST1.json REQUEST2.json
# Ends with the server restarted with $EVERYDAY_FLAGS. Settings: see cluster_env.sh.
. "$(dirname "$0")/cluster_env.sh"
R1=${1:?usage: $0 REQUEST1.json REQUEST2.json}; R2=${2:?usage: $0 REQUEST1.json REQUEST2.json}
OUT=$(mktemp -d)
trap 'echo "--- restoring the everyday server ($EVERYDAY_FLAGS)"; restart_server "$EVERYDAY_FLAGS"' EXIT

echo "### normal mode, reference"; restart_server ""
timeout 1200 python3 "$RUNNERS/conc_client.py" "$SERVER_URL" "$OUT/normal" seq "$R1" "$R2"
echo "### concurrent mode, one at a time"; restart_server "--concurrent 2"
timeout 1200 python3 "$RUNNERS/conc_client.py" "$SERVER_URL" "$OUT/conc-seq" seq "$R1" "$R2"
echo "### concurrent mode, both at once"
timeout 1200 python3 "$RUNNERS/conc_client.py" "$SERVER_URL" "$OUT/conc-par" par "$R1" "$R2"
for i in 0 1; do
  cmp -s "$OUT/normal-$i.txt" "$OUT/conc-seq-$i.txt" && echo "request $i: concurrent-mode alone == normal" || echo "request $i: concurrent-mode alone DIFFERS from normal"
  cmp -s "$OUT/normal-$i.txt" "$OUT/conc-par-$i.txt" && echo "request $i: both-at-once == normal" || echo "request $i: both-at-once DIFFERS from normal"
done
echo "(outputs in $OUT)"
ssh "$HEAD" 'grep -E "Traceback| gen tokens" ~/dsv4-server.log | cut -c1-170 | tail -4'
