"""Needle-in-a-haystack correctness test for Qwen3-Coder-Next, 1-node or pipelined 2-node.

Run alone for 1 node, or under `mlx.launch --backend jaccl` for 2 nodes (pipeline).
Prints one JSON line per (length, position) from rank 0.
"""
import json
import random
import sys

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.utils import sharded_load

MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
LENGTHS = [4096, 8192, 16384]
POSITIONS = [0.10, 0.50, 0.90]
NEEDLE = "Note for auditors: Section 37 has internal code Q-4471."
QUESTION = "What is the internal code for Section 37? Answer with the code only."
EXPECTED = "Q-4471"

# Plain filler sentences: varied, realistic, and containing no codes or numbers
# that could be confused with the needle.
SUBJECTS = ["The warehouse team", "The night shift", "The regional office", "The audit committee",
            "The maintenance crew", "The logistics group", "The finance department", "The site manager"]
VERBS = ["reviewed", "updated", "discussed", "reorganised", "documented", "inspected", "postponed", "approved"]
OBJECTS = ["the loading schedule", "the safety checklist", "the supplier contracts", "the inventory records",
           "the training plan", "the shipping routes", "the quarterly budget", "the equipment logs"]
TAILS = ["after a long meeting.", "before the end of the week.", "with no major concerns.",
         "following feedback from staff.", "as part of routine operations.", "ahead of the busy season.",
         "without any changes to policy.", "in line with earlier guidance."]


def filler_sentences(seed):
    rng = random.Random(seed)
    while True:
        yield f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(TAILS)}"


def build_prompt(tokenizer, target_tokens, position, seed):
    # Grow filler until the user message reaches the target length in tokens.
    gen = filler_sentences(seed)
    sentences = []
    while len(tokenizer.encode(" ".join(sentences))) < target_tokens:
        sentences.extend(next(gen) for _ in range(50))
    insert_at = int(len(sentences) * position)
    sentences.insert(insert_at, NEEDLE)
    body = " ".join(sentences)
    messages = [{"role": "user", "content": f"{body}\n\n{QUESTION}"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt, len(tokenizer.encode(prompt))


def main():
    world = mx.distributed.init()
    rank = world.rank()
    if world.size() > 1:
        model, tokenizer = sharded_load(MODEL, pipeline_group=world, tensor_group=None)
        mode = f"{world.size()}-node"
    else:
        model, tokenizer = load(MODEL)
        mode = "1-node"
    run_label = sys.argv[1] if len(sys.argv) > 1 else ""

    for length in LENGTHS:
        for pos in POSITIONS:
            prompt, n_tokens = build_prompt(tokenizer, length, pos, seed=length)
            # Default sampler in mlx_lm.generate is greedy (temperature 0).
            answer = generate(model, tokenizer, prompt, max_tokens=24, verbose=False).strip()
            if rank == 0:
                print(json.dumps({
                    "mode": mode, "run": run_label, "target": length, "tokens": n_tokens,
                    "position": pos, "answer": answer, "correct": EXPECTED in answer,
                }), flush=True)


if __name__ == "__main__":
    main()
