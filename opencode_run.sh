#!/bin/bash
# One unattended OpenCode run of a task prompt against a git repo, for
# repeatable agent tests (Test B and its variants were run this way).
#
#   opencode_run.sh REPO PROMPT_FILE OUT_DIR NAME
#
# Writes OUT_DIR/NAME.jsonl (OpenCode's event stream), NAME.err, NAME.diff,
# and prints the diff stat, the test result and how the session ended.
#
# Env:
#   RESET=1    discard uncommitted changes to tracked files first
#              (git checkout -- .); otherwise the repo must already be clean
#   TEST_CMD   run after the session, e.g. "swift test" (default: none)
#
# OpenCode must not be running: quit the TUI too, since it restarts the
# background service within seconds, and a second session editing the same
# repo spoils the run. The run uses a private server (--standalone) and the
# default model from your OpenCode config. No --auto.
set -u

if [ $# -ne 4 ]; then
    sed -n '5p' "$0" | sed 's/^# *//'
    exit 64
fi
REPO=$1; PROMPT=$2; OUT=$3; NAME=$4

[ -f "$PROMPT" ] || { echo "ABORT: no prompt file $PROMPT"; exit 66; }
mkdir -p "$OUT"
PROMPT=$(cd "$(dirname "$PROMPT")" && pwd)/$(basename "$PROMPT")
OUT=$(cd "$OUT" && pwd)
cd "$REPO" || exit 66

# Match OpenCode executables only, not command lines that mention the word.
if ps -axo comm | grep -qi opencode; then
    echo "ABORT: OpenCode is already running:"
    ps -axo pid,comm | grep -i opencode
    exit 2
fi

[ "${RESET:-0}" = 1 ] && git checkout -- .
if [ -n "$(git status --short --untracked-files=no)" ]; then
    echo "ABORT: $REPO has uncommitted changes (set RESET=1 to discard them)"
    exit 3
fi

echo "start $(date +%H:%M:%S)"
opencode run --standalone --title "$NAME" --format json "$(cat "$PROMPT")" \
    > "$OUT/$NAME.jsonl" 2> "$OUT/$NAME.err"
echo "exit $?, end $(date +%H:%M:%S)"

git diff > "$OUT/$NAME.diff"
git diff --stat
if [ -n "${TEST_CMD:-}" ]; then
    sh -c "$TEST_CMD" > "$OUT/$NAME.test.txt" 2>&1
    echo "tests: exit $? ($OUT/$NAME.test.txt)"
fi

# "stop" means the model finished; anything else means the session ended early.
python3 - "$OUT/$NAME.jsonl" <<'EOF'
import json, sys
steps = []
for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except ValueError:
        continue
    if e.get("type") == "step_finish":
        steps.append(e["part"].get("reason"))
print(f"steps {len(steps)}, last {steps[-1] if steps else None}")
EOF
