"""DeepSeek-V4-Flash-4bit across two nodes (pipeline) via mlx-vlm: speed sweep + needle test.

Run under:
  mlx.launch --backend jaccl --hostfile ~/m3-ultra-jaccl.json -- ~/vlm-env/bin/python ~/dsv4_run.py
Rank 0 prints one JSON line per measurement.

mlx-vlm 0.7.12 has sharded_load() but no CLI that uses it, hence this script.
"""
import json
import os
import random
import sys
import time

import mlx.core as mx
from mlx_vlm import generate
import dsv4_encoding  # DeepSeek's reference encoder (the mlx-community template garbles prompts)
from mlx_vlm.utils import sharded_load

# Same needle and filler as needle_test.py (copied: that file imports mlx_lm,
# which is not installed in the mlx-vlm venv).
NEEDLE = "Note for auditors: Section 37 has internal code Q-4471."
QUESTION = "What is the internal code for Section 37? Answer with the code only."
EXPECTED = "Q-4471"
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

MODEL = os.environ.get("DSV4_MODEL", "mlx-community/DeepSeek-V4-Flash-4bit")
LENGTHS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["2048", "4096", "8192", "16384"])]
POSITIONS = [0.10, 0.50, 0.90]


def build_body(tokenizer, target_tokens, position, seed):
    gen = filler_sentences(seed)
    sentences = []
    while len(tokenizer.encode(" ".join(sentences))) < target_tokens:
        sentences.extend(next(gen) for _ in range(50))
    if position is not None:
        sentences.insert(int(len(sentences) * position), NEEDLE)
    return " ".join(sentences)


def main():
    world = mx.distributed.init()
    rank = world.rank()

    def out(**kw):
        if rank == 0:
            print(json.dumps(kw), flush=True)

    t0 = time.perf_counter()
    model, processor = sharded_load(MODEL, pipeline_group=world)
    config = model.config.to_dict() if hasattr(model.config, "to_dict") else model.config
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    out(event="loaded", seconds=round(time.perf_counter() - t0, 1),
        active_gib=round(mx.get_active_memory() / 2**30, 1), nodes=world.size())

    def run(body, question, max_tokens):
        prompt = dsv4_encoding.encode_messages(
            [{"role": "user", "content": f"{body}\n\n{question}"}], thinking_mode="chat")
        return generate(model, processor, prompt, max_tokens=max_tokens, temperature=0.0,
                        prefill_step_size=int(os.environ.get("PREFILL_STEP", "2048")))

    for length in LENGTHS:
        # Speed: summarise a filler prompt, 128 output tokens.
        body = build_body(tokenizer, length, None, seed=length)
        r = run(body, "Summarise the text above in detail.", 128)
        out(test="speed", target=length, prompt_tokens=r.prompt_tokens,
            prompt_tps=round(r.prompt_tps, 1), generation_tps=round(r.generation_tps, 1),
            gen_tokens=r.generation_tokens, peak_gib=round(r.peak_memory, 1))

        # Correctness: needle at three positions.
        for pos in POSITIONS:
            body = build_body(tokenizer, length, pos, seed=length)
            r = run(body, QUESTION, 24)
            answer = r.text.strip()
            out(test="needle", target=length, prompt_tokens=r.prompt_tokens, position=pos,
                answer=answer, correct=EXPECTED in answer)


if __name__ == "__main__":
    main()
