"""Check infer_edit_path against real rejected edits in proxy recordings.

usage: python3 path_fill_eval.py DIR...   (folders of *.request.json, kept local)

For each assistant edit call without `path` that OpenCode rejected, the truth is the
path of the model's next edit call with the same oldString (its retry). The
inference sees only the history before the rejected call.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_proxy import infer_edit_path  # noqa: E402

cases = {}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.request.json"))):
        try:
            msgs = json.load(open(f))["messages"]
        except (ValueError, KeyError):
            continue
        for i, m in enumerate(msgs):
            for c in m.get("tool_calls") or []:
                fn = c.get("function") or {}
                if fn.get("name") != "edit":
                    continue
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    continue
                if "path" in args or not args.get("oldString"):
                    continue
                truth = None
                for later in msgs[i + 1:]:
                    for c2 in later.get("tool_calls") or []:
                        f2 = c2.get("function") or {}
                        try:
                            a2 = json.loads(f2.get("arguments") or "{}")
                        except ValueError:
                            continue
                        if f2.get("name") == "edit" and a2.get("path") and a2.get("oldString") == args["oldString"]:
                            truth = a2["path"]
                            break
                    if truth:
                        break
                key = (args["oldString"], msgs[0].get("content", "")[:80] if isinstance(msgs[0].get("content"), str) else "")
                if key in cases and cases[key][1] is None and truth:
                    cases[key] = (msgs[:i], truth, args["oldString"])
                elif key not in cases:
                    cases[key] = (msgs[:i], truth, args["oldString"])

right = wrong = abstain = no_truth = 0
for history, truth, old in cases.values():
    guess = infer_edit_path(history, old)
    if truth is None:
        no_truth += 1
        print(f"  no retry with same oldString; inference {'abstains' if guess is None else os.path.basename(guess)}")
        continue
    if guess is None:
        abstain += 1
    elif guess == truth:
        right += 1
    else:
        wrong += 1
        print(f"  WRONG: inferred {guess} but the retry used {truth}")
print(f"{len(cases)} rejected no-path edits: right {right}, wrong {wrong}, abstained {abstain}, no ground truth {no_truth}")
