# Cluster scripts

> Scripts for running DeepSeek-V4-Flash split across two Mac Studios (M3 Ultra,
> 96 GB each) over Thunderbolt 5 RDMA with MLX JACCL. Deploy by copying the
> scripts to `~/` on every node. Node 1 is "purple", node 2 "orange".

All commands run **on purple**, the first node in `~/m3-ultra-jaccl.json`.
`mlx.launch` starts the same script on both machines.

## TL;DR

- **What:** DeepSeek-V4-Flash-0731 (MLX 4-bit, 144 GB, converted with
  `dsv4_0731_convert.py`) split across two M3 Ultra Mac Studios (96 GB each),
  half the layers on each, over Thunderbolt 5 with MLX's JACCL backend. Served as
  an OpenAI-compatible API on purple, port 8090.
- **Start** (on purple, after the reboot checks below):
  ```bash
  nohup ~/cluster-env/bin/python ~/cluster_launch.py --backend jaccl --hostfile ~/m3-ultra-jaccl.json -- \
    $HOME/vlm-env/bin/python $HOME/vlm_cluster_server.py \
    --model $HOME/models/DeepSeek-V4-Flash-0731-4bit --port 8090 --concurrent 2 > ~/dsv4-server.log 2>&1 < /dev/null &
  ```
  Stop: `pkill -f vlm_cluster_server.py` (orange's half exits with it).
  `--concurrent 2` lets two agents share the cluster (below); leave it out for the
  original one-request-at-a-time path.
- **Speed:** ~29 tok/s decoding, ~700 tok/s reading new prompt tokens; a prompt
  that repeats an earlier prefix is read from a cache.
- **Coding agent:** OpenCode through `record_proxy.py --rules working_rules.txt
  --fill-edit-path` on the client Mac, which adds the working rules to every agent
  turn and fills an edit call's missing `path` when the history makes it
  unambiguous. Test B (a Swift change with a data migration and tests) runs
  unattended in ~7 min: 4 runs in 29.3 min, all finishing with tests passing.
- **DeepSeek tool-call fixes in the server** (`repeat_test.py`): identical or
  conflicting repeated DSML parameters, calls cut off by the loop guard, calls left
  open when the next one starts.
- **All 43 layers run** (`split_test.py`): mlx-vlm 0.7.12's pipeline split drops a
  layer when the layer count doesn't divide evenly; until 2026-11-19 layer 21 ran
  nowhere. The server now installs mlx-lm's split (perplexity 6.20 to 6.02).
- **Two agents at once:** `--concurrent 2` interleaves two requests across the
  pipeline (our everyday setting; off unless passed): 6 agent runs in 26.4 min instead of ~41.5, a lone
  request decodes at about normal speed (28.6-28.9 tok/s).
- **Each reply is cached for the next turn** (with `--concurrent 2`): while the agent
  runs its tools, the server reads its own reply into the prompt cache, so a typical
  turn reads ~39 new tokens (0.23 s) instead of ~550 (1.4 s); 71 of 75 turns hit.
- **Exact sparse-attention selection in decode** below 48k tokens of context, instead
  of mlx-vlm's approximate HISA: same speed, better predictions on text that repeats
  earlier context (0.54 vs 0.68 nats/token).
- **Tried and not used:** speculative decoding with the model's own DSpark drafter
  (correct, but verifying 5 tokens costs ~5x one, so slower); tensor parallelism,
  both nodes on every layer (18.7 tok/s against 29.3: each all_sum after GPU work
  costs ~170-210 us); thinking mode (no gain on Test B).

## How this works, and what we learned

This section explains the setup from the ground up: what each piece does, why it is
there, and what we measured along the way. The per-script sections further down are
the reference for flags and details.

### 1. The model, and why it needs two machines

[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) is a
[mixture-of-experts](https://en.wikipedia.org/wiki/Mixture_of_experts) (MoE) language
model: each layer holds many small "expert" networks, and a router picks a few of them
for every token. Only those experts run, so a token costs far less compute than the
model's size suggests, but all experts still have to be in memory, because the next
token may pick different ones.

The official checkpoint stores most expert weights in 4-bit floating point and the rest
of the large matrices in 8-bit floating point. `dsv4_0731_convert.py` repacks those into
MLX's own [microscaling](https://arxiv.org/abs/2310.10537) formats (MXFP4 and MXFP8,
lossless) and [quantizes](https://en.wikipedia.org/wiki/Quantization_(signal_processing))
the remaining full-precision layers to 4 bits. The result is 144 GB. An M3 Ultra Mac
Studio has 96 GB of [unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html),
shared by the CPU and GPU, so one machine can't hold it; two can, at about 71 GiB each.
The GPU's share is capped by `iogpu.wired_limit_mb` (84 GiB here), which is why that
setting is in the reboot checklist.

Everything runs on [MLX](https://github.com/ml-explore/mlx), Apple's array framework,
with the model code from [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). MLX is
[lazy](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html):
operations build a graph that runs only when a result is needed. That detail matters
twice below.

### 2. Splitting the model: pipeline parallelism over Thunderbolt

The two Studios are joined by two [Thunderbolt 5](https://en.wikipedia.org/wiki/Thunderbolt_(interface))
cables used for [RDMA](https://en.wikipedia.org/wiki/Remote_direct_memory_access), which
lets one machine write into the other's memory without going through the network stack.
MLX talks to it through its JACCL [distributed backend](https://ml-explore.github.io/mlx/build/html/usage/distributed.html).

The model is split by **[pipeline parallelism](https://arxiv.org/abs/1811.06965)**: one
Studio holds the first half of the 43 layers, the other the second half. For each token,
orange runs its layers and hands the result to purple, which runs the rest and shares
the output with every rank. Only a few small tensors cross the link per token. The cost is
that each machine waits while the other works.

The split itself had a bug, found late. mlx-vlm's `PipelineMixin.pipeline` (0.7.12, and
`main` at the time of writing) computes each rank's first layer from that rank's own
layer count, so when the layers don't divide evenly the ranges don't meet: with 43 layers
on 2 ranks, rank 0 ran layers 22-42 and rank 1 ran 0-20, and **layer 21 was never loaded or
run**, with no error. The model still wrote fluent, mostly correct code, so nothing looked
wrong. Measured as the [perplexity](https://huggingface.co/docs/transformers/perplexity) of
a fixed 3,071-token text, it was 6.20 with the layer missing and 6.02 with it back (tensor
parallelism, which runs every layer, gave 6.04). The server now replaces the split with
mlx-lm's, which assigns every layer to exactly one rank (`fixed_pipeline`, tested by
`split_test.py`); rank 0 runs layers 21-42 and rank 1 runs 0-20.

`vlm_cluster_server.py` turns this into an OpenAI-style chat API. Purple (rank 0) serves
HTTP, and every request is broadcast to both ranks so they run the same generation in
lockstep. Two fixes keep idle machines idle:

- **Waiting without spinning.** Waiting for the next job inside a JACCL collective polls
  in a loop and keeps a CPU core at 100%. Workers now block on a plain TCP "wake socket"
  instead, and join the collective only when a job is coming.
- **`cluster_launch.py`.** Stock `mlx.launch` (0.32.4) never sleeps in its I/O loop and
  burned ~1.7 cores for the whole life of the job. This is a copy of it with that loop fixed.

The other server-side problem was the GPU stalling on memory that macOS had quietly
evicted. If a single GPU command takes too long, macOS aborts it with a "GPU timeout",
and both ranks crash. The fixes: keep the weights resident (MLX's residency refresh, with
the wired limit held raised); touch every weight, one tensor per GPU command, at startup
and before any job that follows more than 60 s of idle; and run a tiny warm-up generation
before serving.

### 3. Prefill, decode, and why decoding is slow

Generating a reply has two phases:

- **Prefill** reads the prompt. All its tokens go through the model together, so each
  weight read from memory is used for hundreds of tokens. Here that runs at roughly
  600-700 tokens/s, in chunks of 512 (larger chunks made the GPU time out).
- **Decode** produces the reply one token at a time. Every token needs all the active
  weights read from memory again, for just that one token, so speed is set by
  [memory bandwidth](https://en.wikipedia.org/wiki/Memory_bandwidth), not by compute.
  Here that's about 29 tokens/s.

Coding agents send the whole conversation again on every turn, so the server keeps a
[KV cache](https://huggingface.co/docs/transformers/main/en/kv_cache) of earlier prompts
and only prefills what is new. In practice almost all of an agent run's time is decode:
in four measured coding-task runs, 83% of the wall time was the model writing tokens,
14% reading prompts, and about 3% the agent's own tools and tests. So a run takes
roughly *tokens generated ÷ 29 tok/s*, and there are only two ways to make it faster:
write fewer tokens, or write them faster. Sections 5 and 6 are about each.

### 4. Making tool calls reliable

The client is [OpenCode](https://opencode.ai), a terminal coding agent. It gives the
model [tools](https://platform.openai.com/docs/guides/function-calling) (read a file, edit
a file, run a shell command), streams replies over
[server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events),
runs whatever tool calls come back, and returns the results to the model.

DeepSeek-V4 writes tool calls in its own markup, DSML. The server builds the prompt with
DeepSeek's reference encoder (`dsv4_encoding.py`) and parses the calls back into the
OpenAI format. One property drives most of the work here: **if a tool call fails to
parse, the client receives it as plain text and ends the session.** Every malformed
call the model writes is therefore fatal unless the server can do something with it.
Each fix below came from a real run that died that way, and each one keeps the rule of
never inventing a value. A call is either repaired structurally or passed on incomplete,
so the client rejects it and the model retries:

- the same parameter sent twice with the same value: duplicates dropped;
- a required parameter sent twice with different values: left out (no guessing which);
- a parameter the tool doesn't take, sent with conflicting values: left out;
- a call left open when the next one starts: closed after its finished parameters;
- a call cut off by the loop guard: kept with its finished parameters.

The **loop guard** stops generation when the last 200 characters already appeared 4
times in the last 6,000, which catches a model repeating itself. We started with
[greedy decoding](https://huggingface.co/blog/how-to-generate) (always the most likely
token), which can repeat a failed edit forever; replays showed that
[sampling](https://huggingface.co/blog/how-to-generate) at temperature 1.0, DeepSeek's
own recommendation, breaks such streaks, so agent runs now sample.

### 5. Writing fewer tokens: testing the agent

To change anything with confidence we needed a repeatable test. **Test B** asks the
agent for a real feature in a small macOS menu-bar app (Swift): store notes by desktop ID
instead of desktop number, migrate existing notes, add a helper, and write tests, with
five numbered steps and several "and also" clauses. **Test C** is a Python change to a
text-chunking module with 18 clauses, scored by a hidden grader the agent never sees.
`opencode_run.sh` runs a test unattended (and refuses to start if another OpenCode
session could interfere), and each change was judged on batches of four runs. That is a
small sample: run times for the same task vary by 2x, so only large differences count.

What helped, in order:

1. **A self-check rule.** The model kept missing half of a two-part requirement. A rule
   telling it to *quote every clause of the task, word for word, next to the line of code
   that does it* took that requirement from 3 of 7 runs to 22 of 23. A weaker "check every
   clause" left it paraphrasing away the very clause it had missed.
2. **Rules added by the proxy.** OpenCode 2.2.6 ignores `AGENTS.md`, so the rules had to
   be pasted into every task. `record_proxy.py --rules` now adds them to every agent turn.
3. **An edit rule, and filling a missing edit path.** A quarter of all generated tokens
   went into edit calls OpenCode rejected, most for a missing file `path`, and each
   rejection makes the model rewrite the whole edit. A rule ("every edit call gives path
   first, then oldString, then newString") helped, but how often the model forgets the
   path varies a lot from run to run. With `--fill-edit-path` the proxy adds the path when
   exactly one file the session has seen contains the edit's old text. This is the one
   place a value is filled in, so it was checked first against recordings (51 of 53 cases
   right, none wrong), and none of the 33 fills checked in later runs went to the wrong file.
4. **"Keep text before a tool call under 50 words."** When stuck, the model wrote long
   analyses before cheap calls; in one batch, half of all tokens were in turns that only
   read a file or ran a command.

| Test B, 4 runs | Total time | Tokens written |
|---|---|---|
| Self-check rule, added by the proxy (1, 2) | 47.2 min | 69k |
| + edit rule and path filling (3) | 35.4 min | 51k |
| + 50-word rule (4) | **29.3 min** | **42k** |

What didn't help: thinking mode (no gain), a rule to test every limit at its boundary
(no gain, and two runs got worse), and a stronger 50-word rule that added "when stuck,
read the code or run the tests instead": faster still, but the model then deleted
existing tests to make failures go away, in every run. Rules make the model recheck what
it already sees; they don't make it notice a meaning it missed. For that, the task text
has to spell it out.

### 6. Writing tokens faster: two experiments that didn't pay off

**Speculative decoding** ([Leviathan et al.](https://arxiv.org/abs/2211.17192)) uses a
small, fast "drafter" to guess several next tokens, then checks all the guesses with one
pass of the big model, keeping the ones it agrees with. It helps when checking k tokens
costs about the same as producing one. DeepSeek-V4-Flash-0731 ships its own drafter, a
"DSpark" head that guesses 5 tokens at a time from the model's internal states, related
to [multi-token prediction](https://arxiv.org/abs/2404.19737) as used in
[DeepSeek-V3](https://arxiv.org/abs/2412.19437). Getting it to run across two machines
took two fixes (`bench_mtp.py`), and it was correct and accurate: up to 4.6 tokens
accepted per round. But it was slower, 24.8 against 28.8 tok/s, because checking 5
tokens took 155 ms against 29 ms for one. In an MoE model each token goes to its own
experts, so checking 5 tokens reads close to 5 times the expert weights, and decode is
limited by reading weights: there is nothing to win.

**Tensor parallelism** ([Megatron-LM](https://arxiv.org/abs/1909.08053)) splits every
layer across both machines instead of splitting the layer stack. Both machines then work
on every token and each reads only half of each layer, so in principle it is up to twice
as fast. The price is that the halves have to be combined inside every layer with an
[all-reduce](https://en.wikipedia.org/wiki/Collective_operation) (`all_sum`): 86 per token
for this model. mlx-vlm supports it, and it ran correctly at 73.2 GiB per machine, but
decoded at 18.7 tok/s against 29.3. A bare `all_sum` over the link takes about 20 µs;
one that follows GPU work takes 170-210 µs, most likely because the GPU has to finish
and hand over to the CPU first. 86 of those per token outweigh the half of the weight
reading saved. If MLX could make that handoff close to the bare time, tensor parallelism
would by our estimate reach about 50 tok/s; that is a question for MLX itself.

### 7. Where that leaves things

Decoding at ~29 tok/s is close to what this hardware and software can do; the speed-ups
that worked came from the agent writing fewer, better-aimed tokens. The model still
misses an occasional requirement (about 1 run in 4 in recent Test B batches), so its work needs
reviewing, but a typical Test B run now finishes unattended in about 7 minutes with all
tests passing.

## Before any cluster run

These reset on every reboot. Check them first.

```bash
# Thunderbolt IPs (JACCL needs IPv4 on the RDMA ports)
# purple:
sudo ifconfig bridge0 down
sudo ifconfig en4 inet 192.168.0.1 netmask 255.255.255.252
sudo route change 192.168.0.2 -interface en4
sudo ifconfig en5 inet 192.168.0.5 netmask 255.255.255.252
sudo route change 192.168.0.6 -interface en5
# orange: same, with .2 on en4 and .6 on en5, routes to .1 and .5

# GPU memory limit, both machines (only needed for DeepSeek-V4-Flash)
sudo sysctl iogpu.wired_limit_mb=86016
```

lil-sis (`com.local.mlx-server`, `:8080`) must be stopped for DeepSeek, which
needs ~71 GiB per node:

```bash
launchctl bootout gui/$(id -u)/com.local.mlx-server          # stop
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.local.mlx-server.plist   # start
```

## Scripts

| Script | Venv | What it does |
|---|---|---|
| `vlm_cluster_server.py` | `~/vlm-env` | OpenAI-compatible chat server for an mlx-vlm model split across both nodes |
| `split_test.py` | `~/vlm-env` | Offline check that the server's pipeline split runs every layer on exactly one rank |
| `proto_interleave.py` | `~/vlm-env` | Prototype: two requests decoded at once, interleaved across the pipeline (1.87x combined throughput) |
| `interleave_test.py` | `~/vlm-env` | Offline check of the server's interleaved scheduling (`--concurrent 2`) with two fake ranks |
| `profile_half.py`, `active_bytes.py` | `~/vlm-env` / any | Where each rank's half of a decoded token spends its time, against the bandwidth floor |
| `cluster_launch.py` | `~/cluster-env` | `mlx.launch` with its IO loop fixed so it doesn't spin ~1.7 cores while the job runs |
| `needle_test.py` | `~/cluster-env` | Needle-in-a-haystack correctness test, Qwen3-Coder-Next via mlx-lm |
| `dsv4_run.py` | `~/vlm-env` | Speed sweep + needle test for DeepSeek-V4-Flash via mlx-vlm |
| `opencode_run.sh` | (any) | One unattended OpenCode run of a task prompt against a git repo, for repeatable agent tests |
| `record_proxy.py` | (any, stdlib) | Recording proxy between a client and the server; can add working rules to every agent turn |
| `bench_mtp.py` | `~/vlm-env` | Speculative decoding with DeepSeek-V4-Flash-0731's DSpark drafter on the pipelined cluster (experiment: no speedup) |
| `allsum_repro.py` | `~/vlm-env` | Times JACCL all_sum alone vs after GPU work (why tensor parallelism loses) |

The two venvs are separate on purpose. `~/cluster-env` has mlx-lm 0.32.2;
`~/vlm-env` has mlx-vlm 0.7.12. Both pin mlx 0.32.4, which must match on both
machines.

### vlm_cluster_server.py

Why it exists: mlx-vlm can split a model across machines (`sharded_load`) but
its own server cannot. Rank 0 serves HTTP and hands each request to the other
node over `mx.distributed`; both generate in lockstep.

```bash
nohup ~/cluster-env/bin/python ~/cluster_launch.py --backend jaccl --hostfile ~/m3-ultra-jaccl.json -- \
  $HOME/vlm-env/bin/python $HOME/vlm_cluster_server.py \
  --model $HOME/models/DeepSeek-V4-Flash-0731-4bit --port 8090 --concurrent 2 > ~/dsv4-server.log 2>&1 < /dev/null &

curl -s localhost:8090/health
```

Client settings (Cline "OpenAI Compatible", Continue, curl):

```
base URL   http://<first-node>:8090/v1
model      mlx-community/DeepSeek-V4-Flash-4bit
API key    anything
```

`cluster_launch.py` takes the same arguments as `mlx.launch` (`launch_test.py`,
offline). Stock `mlx.launch` (mlx 0.32.4) never sleeps in its IO loop: each
rank's thread selects on the child's stdin as writable even with nothing to
write, and the main loop selects on its own stdin, which under `nohup` is at EOF
and always readable. Measured ~1.7 cores on purple for the server's whole life.
`cluster_launch.py` runs `mlx.launch` with a copy of that loop with both waits
fixed (test: 0.006 s of CPU over 2 s idle, against 3.6 s for stock). It only
patches mlx 0.32.4; on another version it warns and runs the stock loop.

Stop it: `pkill -f vlm_cluster_server.py` on purple (orange's half exits with
it, through the wake socket below; if not, run the same on orange).

Limits:

- **One request at a time.** Others wait.
- **Native DeepSeek tool calling** (2026-11-08). For DeepSeek-V4 the prompt is
  built by DeepSeek's own reference encoder, vendored as `Scripts/dsv4_encoding.py`
  (MIT, preview revision a7aaed80, found via exo). It renders tools, tool calls
  and `<tool_result>` turns in the DSML format the model was trained on, and its
  `parse_tool_calls` turns `<｜DSML｜tool_calls>` output back into OpenAI
  `tool_calls`, several per reply. Generation stops on `</｜DSML｜tool_calls` on
  both nodes. Tests: `Scripts/dsml_test.py`.
  - **This also fixed a prompt bug.** Through mlx-vlm's `apply_chat_template`,
    the mlx-community Jinja template received every message as a list of parts
    and rendered it as Python repr: `"hello"` reached the model as
    `[{'type': 'text', 'text': 'hello', 'content': 'hello'}]`, with a doubled
    `</think>` on assistant turns. Every request before 11:00 on 2026-11-08 had
    this.
  - Other models fall back to an emulated `<tool_call>{json}</tool_call>` format.
- **Thinking mode** (DeepSeek-V4). Off by default. Turn it on per request with
  `reasoning_effort` (`low`/`medium`/`high`/`max`), `reasoning: {"effort": ...}`
  or `chat_template_kwargs: {"thinking": true}`, or for every request with
  `--default-reasoning-effort high`. The prompt is encoded in DeepSeek's
  `thinking` mode; the reasoning comes back as `reasoning_content` (streamed as
  its own deltas) and the answer as `content`. Tool calls are only parsed after
  `</think>`. Thinking defaults: temperature 1.0, top_p 0.95 with tools (DeepSeek's
  recommendation), max_tokens 32768. `max`/`xhigh` add the preview encoder's
  effort prompt. Tests: `think_test.py` (offline), `think_live.py`,
  `think_compare.py`.
- **Status lines in the reasoning stream** (streaming requests). The server adds
  `⟦cluster⟧` lines to `reasoning_content`, which Cline shows in its Thinking
  section: one before the prompt is read (new vs cached tokens, estimated read
  time from a running average) and one at the end (tokens, tok/s, read time,
  total time, cache state, peak GPU memory). Lines carrying the marker are
  stripped from any history a client sends back, so the model never sees them.
  `--no-stats-in-reasoning` turns them off.
- **Recovery when a tool keeps failing** (`repeat_test.py`, offline). If the same
  tool fails with the same error twice in a row and the client set no temperature,
  that one turn is sampled at temperature 1.0 / top_p 0.95. Replays of a stuck edit
  showed that at temperature 0 the model resends the failed call verbatim even when
  told what was wrong. `--no-resample-on-repeat` turns this off. Errors are
  recognised in OpenCode's JSON tool-result format (`{"error": ...}`).
- **Conflicting DSML parameters.** A required parameter sent twice with different
  values is left out of the call, so the client rejects it and the model retries;
  no value is guessed. So is one the tool's schema doesn't list at all (B7: a
  complete `write` carried two different `oldString`s, an `edit` parameter, and
  went to the client as text, ending the session); the tool doesn't take it, so
  nothing is lost. A conflict on an optional parameter the tool does take still
  fails parsing.
- **Loop guard** (`repeat_test.py`, offline). Generation stops when the last 200
  characters already appeared 4 times in the last 6,000. A turn stopped in plain
  text ends with finish `length`. A turn stopped inside a tool call keeps the
  parameters that finished, goes through the duplicate and conflict handling
  above, and ends with `tool_calls`: a complete call runs, an incomplete one
  fails the client's validation and the model retries. Before this, the cut-off
  call went to the client as text with `length`; in W5 (2026-11-14) the model
  was repeating `oldString` inside an edit, OpenCode answered `length` with its
  "smaller pieces" message, the model looped again, and after three such turns
  the headless session gave up with 4 tests failing. Replayed from that state:
  the guard fired twice in 8 resends and both turns came back as edits missing
  `path`; after OpenCode's rejection, 8 of 8 next turns were proper tool calls
  with no loop.
- **A call left open when the next one starts** (`repeat_test.py`). In X3
  (2026-11-14) the model stopped partway into an edit's `newString` and wrote
  the same edit again as a new call; the parser read both as one call with two
  `oldString`s, the reply reached OpenCode as text and the session ended after
  1.7 minutes. The open call now keeps only its finished parameters and is
  closed, so the client rejects it and runs the next one.
- **Warm-up before serving.** All ranks run a tiny generation after loading (about
  7 s; a normal tiny request takes 0.3 s). Without it, a request arriving the
  moment the server came up crashed both ranks with a Metal GPU timeout.
- **Keeping the weights wired** (`pin_test.py`, offline). macOS drops GPU
  residency after ~2 s idle (a macOS bug, ml-explore/mlx#4633). The server sets
  `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=1000` before importing MLX (set it
  yourself to override) and holds the wired limit raised for its whole life,
  because mlx-vlm's `stream_generate` sets it back to 0 after every call, and at
  0 the refresh has nothing to keep. Measured: ~75 GiB per node stayed wired
  through 150 s idle, and the first request after idle took 0.49 s (5.3 s
  before). `--no-pin-weights` turns the held limit off.
- **Paging the weights back in after idle** (`prefault_test.py`, offline). macOS
  unwires idle GPU memory within seconds, even with `mx.set_wired_limit` raised
  (60 GiB measured unwired after ~15 s idle), and may then compress or swap it.
  The first request after a long idle (12 h, then ~2 days) paged the weights back
  in inside GPU work while the other rank's GPU waited, and the ~5 s GPU watchdog
  killed rank 1. Now, after `--prefault-idle` seconds without a job (default 60;
  0 = every job, negative = never), each rank touches its weights one tensor per
  command buffer and the ranks meet at a CPU barrier before any cross-rank GPU
  work. The warm-up does the same. Log: `paged in N weight tensors in Xs, then
  waited Ys for the other ranks`.
- **Waiting for jobs without spinning** (`wake_test.py`, offline). The other
  ranks receive each job through a CPU `all_sum`, and JACCL waits for one by
  polling the Thunderbolt RDMA completion queue (`tbt_poll_cq`), so rank 1 held
  one core at 100% for as long as the server was idle. Rank 0 now sends one byte
  over TCP (`--wake-port`, default 8095, on rank 0's host from
  `MLX_JACCL_COORDINATOR`) before each broadcast; the other ranks block on it and
  only then enter the `all_sum`. If rank 0 dies, they see the socket close and
  shut down. `--no-wake` turns it off. Log: `wake socket up on port 8095`.
- Default `max_tokens` is 8192. Cline sends none, and the old 2048 default cut
  an answer off at exactly 2048 tokens.
- Server log: `~/dsv4-server.log` on purple. (`~/vlm-server.log` there is a stale
  symlink into an old scratchpad; don't write to it.)
- **Prompt cache, one slot** (added 2026-11-08). DeepSeek-V4's cache (sliding
  window + compressed pooling layers) can't be trimmed, so mlx-vlm's built-in
  turn reuse never fires. The server deep-copies the cache at the end of each
  prompt and reuses it when the next request's tokens start with exactly that
  prompt. Measured: 10.8k-token follow-up in 0.5 s cached vs 15.1 s cold,
  identical output (`Scripts/cache_test.py`). A request with a different prefix,
  or `{"cache": false}`, reads from scratch. `--no-prompt-cache` turns it off.
  The log shows `N prompt (M cached)` per request.
- `stop` strings trim what is sent; generation still runs to EOS or
  `max_tokens` on both nodes so they stay in step.
- Only models whose pipeline support sits where mlx-vlm's loader looks
  (`language_model.model.pipeline`) can be split. DeepSeek-V4 does. Qwen3 and
  Ministral 3 do not.

Tested 2026-11-07: every HTTP path on one node (Ministral 3B: health, models,
non-stream, stream, multipart content, tools rejection, stop strings, client
disconnect), and the cross-node job broadcast on both nodes (0 B to 400 KB,
byte-identical, 0.3 to 8 ms). **Not yet tested with a model split across both
nodes.** DeepSeek-V4-Flash is the first.

### needle_test.py

```bash
~/cluster-env/bin/python ~/needle_test.py single          # one node
~/cluster-env/bin/mlx.launch --backend jaccl --hostfile ~/m3-ultra-jaccl.json -- \
  $HOME/cluster-env/bin/python $HOME/needle_test.py run1
```

Result 2026-11-07: 36/36 correct, identical across 1 and 2 nodes. See
Cluster Setup - What Worked.

### dsv4_run.py

```bash
~/cluster-env/bin/mlx.launch --backend jaccl --env PREFILL_STEP=512 --hostfile ~/m3-ultra-jaccl.json -- \
  $HOME/vlm-env/bin/python $HOME/dsv4_run.py 4096
```

**Use `PREFILL_STEP=512`.** Measured 2026-11-08: with mlx-vlm's default 2048-token
prefill chunks, a ~4.5k-token prompt crashed with `kIOGPUCommandBufferCallbackErrorTimeout`
on purple. With 512 it passed, and prefill rose from 243 to 643 tok/s. The server
now defaults to 512 (`--prefill-step`).

| Run | Prompt | Prefill | Decode | Peak (purple) | Needle |
|---|---|---|---|---|---|
| 2k, chunk 2048 | 2,587 | 243 tok/s | 35.9 tok/s | 76.8 GiB | 3/3 |
| 4k, chunk 2048 | ~4,500 | GPU timeout | – | – | – |
| 4k, chunk 512 | 4,526 | 643 tok/s | 33.0 tok/s | 75.6 GiB | 3/3 |
| 8k, chunk 512 | 8,396 | 619 tok/s | 33.6 tok/s | 75.6 GiB | 3/3 |
| 16k, chunk 512 | 16,786 | 596 tok/s | 33.1 tok/s | 75.7 GiB | 3/3 |

Orange never dropped below 21% free memory; swap did not grow on either node.

The argument is a comma-separated list of prompt lengths (default
`2048,4096,8192,16384`). Start with `2048` alone: at ~71 GiB of weights per
node, DeepSeek sits right at the memory cliff.

### bench_mtp.py: speculative decoding (tried, not worth it here)

DeepSeek-V4-Flash-0731 ships a DSpark block drafter (`dspark_block_size = 5`, reading
layers 40-42), which our 4-bit conversion dropped. Its tensors are in the last 3 of
the official checkpoint's 48 files (~10.5 GB); mlx-vlm 0.7.12 builds the drafter when
`load_drafter` is given that folder (config, index and those 3 files). Running it on
the pipelined model needed two fixes, both in this script:

- Layer captures are recorded only on the rank that owns layers 40-42 (the final
  hidden state is all_gathered; captures are not). Each capture is all_summed, the
  owner contributing it and the other rank zeros.
- MLX runs collectives in dependency order. The zeros depend on nothing, so the
  non-owner's all_sum went out before its pipeline send while the owner waited in
  recv: deadlock. Each capture now depends on the final output (`mx.depends`).

Memory: the drafter is ~10 GiB (already FP8/FP4), so each node runs at 80.6 GiB
active, 82.6-83.7 GiB peak, against the 84 GiB wired limit; the warm-up needs the
server's wired-limit setup or it hits a GPU timeout.

Results (greedy, a real 15k-token prompt, 2026-11-15):

| Run | tok/s | Tokens per round | Drafts accepted | Output = plain greedy |
|---|---|---|---|---|
| No drafter | 28.8 | 1 | - | - |
| Block 3 | 23.7 | 2.91 | 95.5% | yes |
| Block 5 | 24.8 | 4.57 | 90.1% | yes |

The drafter is good and cheap (16 ms per round), but verifying 5 tokens costs 155 ms
against 29 ms for one: each token routes to its own experts, so a 5-token pass reads
~5x the expert weights, and decoding is bound by memory reads. Speculative decoding
only pays when checking k tokens costs much less than k steps; here it can't beat
plain decoding, so the server doesn't use it.

The same script also tries tensor parallelism (`--tensor --prefault`): every layer
split across both nodes instead of half the layers on each. mlx-vlm 0.7.12 supports
it for DeepSeek-V4 (`sharded_load(..., tensor_group=...)`), but defines `shard()` on
the inner LanguageModel, so the script adds a delegating `shard()` to the outer
Model. Each node then holds 73.2 GiB (2.8 GiB is copied to both). Like the server,
it needs every weight touched once before the warm-up, or the first pass hits a GPU
timeout.

| Split | tok/s (greedy, real 15k-token prompt) |
|---|---|
| Pipeline (what the server uses) | 29.3 |
| Tensor parallel | 18.7 |

Each token needs 86 all_sums in a row (two per layer), each after GPU work.
`allsum_repro.py` (standalone, launch it on both nodes) separates the costs, per
all_sum of 8 / 32 KiB bfloat16:

| Variant | 8 KiB | 32 KiB |
|---|---|---|
| all_sum of an already evaluated array | 18 us | 25 us |
| GPU multiply, then all_sum, eval each | 212 us | 213 us |
| the same, 86 chained in one graph (as in a forward pass) | 166 us | 182 us |
| as above with the all_sum on `stream=mx.cpu` | 171 us | 180 us |
| a GPU multiply alone, for comparison | 9 us | 9 us |

So the cost is the handoff between GPU work and the collective, not the link or the
chaining. At ~20 us each, tensor parallelism would by our estimate reach about 50
tok/s. (`bench_mtp.py --allsum-bench` gives rougher numbers for the same thing.)
Launch it with the server stopped and give the RDMA link ~15 s after stopping: an
immediate start failed with `[jaccl] Changing queue pair to RTR failed with errno 14`.

### proto_interleave.py: two requests at once (prototype)

With one request at a time, each node idles half of every token: rank 1 runs layers
0-20 while rank 0 waits, then rank 0 runs 21-42 while rank 1 waits. Batching two
requests into one forward pass doesn't help this MoE model (each token reads its own
experts, so 2 tokens cost ~2x). Interleaving does: each step, rank 1 runs its layers
for one request while rank 0 runs its layers for the other, then a single
`all_gather` swaps rank 1's hidden state and rank 0's sampled token (packed in
float32), so neither side waits on a one-way send. Each request advances every two
steps; both nodes are busy every step.

On two real 15k-token prompts, 64 greedy tokens each (`--no-state-eval`):

| | Combined tok/s | Per request |
|---|---|---|
| One at a time (this script's simple loop) | 26.6 | 26.6 |
| Interleaved | 49.7 | 24.9 |

Tokens were identical to one-at-a-time decoding for both requests. A step takes
17.8 ms: the slower node's half (16.5 ms) plus ~1.5 ms. Evaluating every cache's
`.state` each step (the script's default) made steps 44 ms; mlx-vlm's own generate
doesn't do it.

**In the server: `--concurrent 2`** (off by default). Rank 0's HTTP handlers run in
their own threads and queue jobs for an engine on the main thread, the only thread
that uses MLX; rank 1 follows. Both ranks run the same protocol: a "change"
broadcast (finished requests, jobs to admit), prefill and first token for each new
job (with the usual prompt-cache snapshots, done by the engine itself because
mlx-vlm's `generate_step` runs one token ahead), a "go" broadcast, then interleaved
decoding until rank 0 calls for an event (a request finished, or one is waiting).
Rank 0 alone samples after the first token, applies the stop checks and the loop
guard, and streams. The scheduling is tested offline by `interleave_test.py` (two
fake ranks as threads; 40 random runs of request lengths and event timings).

With only one request active, the engine decodes in lockstep like normal mode,
building one token ahead: both ranks sample the same token (a per-token seed from the
job's seed) and make the same stop decisions (`advance()` is deterministic, and rank 1
keeps the text too); every 8 tokens a CPU all_sum tells rank 1 whether rank 0 wants
to stop for its own reasons (a request waiting, or the client gone).

Live, two recorded agent requests (17-20k-token prompts, greedy, 96 tokens): output
identical to normal mode, alone and with both at once. Decode: normal mode 29.1-29.4
tok/s; one request in concurrent mode 28.6-28.9 (24.4 before the lockstep path); two
at once 24.5-24.6 each, about 49 combined.

With real agents (OpenCode, Test B, two at a time on two copies of the repo, 3
pairs): 6 runs in 26.4 min (pairs of 8:30, 8:44, 9:12) against about 41.5 min one
after another in normal mode, 1.57x the throughput. All 6 finished with every test
passing and met the requirement the tests watch most closely; each agent took
7.9-9.2 min, against about 6.9 alone.

### Caching each reply for the next turn (`--concurrent 2`)

Snapshots of the prompt cache are taken during prefill, before any generated token goes
in, so each turn used to re-read the previous reply (~550 tokens, ~1.4 s). Splicing the
generated tokens back in is fragile (the client sends the reply back re-encoded), so
instead: after a reply, rank 0's handler encodes the conversation with the reply as the
client will send it back (`queue_precache`; the content as streamed, the tool calls as
sent), once followed by a tool result and once by a user turn, and the engine prefills
their common tokens on top of the prompt's snapshot, on both ranks, at its next change
point (usually while the agent runs its tools). A reply the client changed simply
misses, and the lookup falls back to the prompt's snapshot. `precache_check.py` checks
the encoding offline against recorded sessions (19 of 19 turns would hit). Live, 3 Test
B runs: 71 of 75 requests reused a precached reply; median new tokens per request 39
(was 551), mean prefill 1.64 s per request (2.20 s before); runs 6:47, 5:34, 6:43, all
passing.

### profile_half.py, active_bytes.py: where a decoded token's time goes

`active_bytes.py` (headers only) puts the weights read per token at 4.84 GB on rank 1
(layers 0-20) and 5.67 GB on rank 0 (21-42 and the output layer): a floor of 6-9 ms
per half at 600-800 GB/s. `profile_half.py` times each rank's half alone at an 8k-token
context:

| | Rank 1 | Rank 0 |
|---|---|---|
| Synced / CPU building ahead | 17.2 / 15.8 ms | 19.0 / 17.2 ms |
| Attention | 11.9 ms | 13.0 ms |
| Experts | 7.6 ms | 7.5 ms |
| Sparse compressed attention (1:4, indexer), per layer | 0.72 ms (10 layers) | 0.76 ms (11) |
| Compressed attention (1:128), per layer | 0.43 ms (9) | 0.42 ms (11) |
| Sliding-window attention, per layer | 0.42 ms (2) | - |

So each half runs at about half its bandwidth floor, and attention is ~60% of it; even
the simplest attention layer takes 0.42 ms against ~0.19 ms of weight reads. mlx-vlm's
DeepSeek-V4 code already uses custom Metal kernels and `mx.compile`, and single-token
decode takes the fused paths. If attention ran at memory speed, decode would be ~50
tok/s instead of ~29; that would take faster attention kernels, mainly for the
sparse-indexed layers.

`--ablate-only` replaces one part with a free stand-in and times the half pipelined
(15k-token context; ms per half, rank 1 / rank 0):

| | Rank 1 | Rank 0 |
|---|---|---|
| As is | 16.6 | 18.1 |
| Exact indexer selection instead of HISA | 15.9 | 17.5 |
| Indexer free (its compressor and top-512 selection) | 14.2 | 15.5 |
| Indexer and every compressor free | 13.7 | 14.8 |

The indexer costs ~2.5 ms per half, compressors ~0.5-1 ms more; the remaining ~14 ms is
attention and the experts. mlx-vlm decodes with HISA (coarse block filter, then top-k
among the kept blocks), an approximation meant for long contexts. The exact selection
was faster here by 0.6-0.7 ms per half at 15k and 32k tokens and equal at 60k, but
in real decoding the difference is within run-to-run noise (28.4-29.4 tok/s both ways,
3 runs each). It predicts better, though: scoring the last 600 of 30k tokens one at a
time (`bench_mtp.py --decode-nll 600`, a tiled text, so the tail repeats earlier
context), 0.540 nats/token and 94.3% top-1 against HISA's 0.675 and 92.5%. The server
uses the exact selection in decode below 12,000 compressed positions (48k tokens) and
HISA beyond (`--hisa-min-pooled`, 0 = HISA always); prefill is unchanged.

### record_proxy.py

Sits between a client (OpenCode) and the server on 127.0.0.1 and saves each request
and the streamed reply as it arrives, so a turn that loops or gets killed still
leaves its text. Recordings contain your prompts and code: keep them local.

```
python3 record_proxy.py --upstream http://SERVER:8090 [--rules working_rules.txt]
```

With `--rules`, the file's text goes at the top of the first user message of every
request that offers tools, unless the message already starts with it. That is where
the Test B runs pasted the working rules (OpenCode 2.2.6 ignores `AGENTS.md`), so
every session gets them without pasting. Checked on Test B: the injected first
message matched the hand-pasted prompt byte for byte, and 3 of 4 runs met the
requirement the rule targets (22 of 23 across all runs with the rule, 3 of 7
without). `working_rules.txt` is a general version. Tests: `proxy_test.py`. Its edit line ("path first, then oldString, then newString") halved rejected
edit turns on Test B (30 to 16 over 4 runs; missing `path` 20 to 7), though the share of
tokens in rejected edits stayed ~24%: the rest are large edits missing oldString or newString.

With `--fill-edit-path` the proxy also fills a missing `path` in an `edit` call when
the session history makes it unambiguous: exactly one file the session has read,
edited or written contains the call's `oldString` (20+ characters). Each fill is
logged; anything else passes unchanged and the client rejects it as before. OpenCode
still checks that `oldString` is in that file before editing. Checked against all our
recordings (`path_fill_eval.py`): of the rejected no-path edits whose retry shows the
intended file, 51 inferred right, 0 wrong, 2 left alone.

Its last rule, "Keep text before a tool call under 50 words", targets long write-ups
before cheap calls (in one Test C batch, 49% of generated tokens were in turns that
only called `read` or `shell`). With it, Test B took 29.3 min over 4 runs against 35.4
(tokens -17%), with the same quality; on Test C quality held and runs were faster
than without it. A version that added "When stuck, don't reason it out in text: read
the code or run the tests" was faster still but deleted existing tests in all 4
Test C runs, so it was dropped.

### opencode_run.sh

Runs one OpenCode session unattended (`opencode run --standalone`, no
`--auto`) from a prompt file and keeps what it did:

```
RESET=1 TEST_CMD="swift test" ./opencode_run.sh ~/path/to/repo prompt.txt runs b10
```

It writes `runs/b10.jsonl` (OpenCode's event stream), `.err`, `.diff` and,
with `TEST_CMD`, `.test.txt`, then prints the diff stat and how the session
ended (`last stop` means the model finished). It refuses to start if any
OpenCode process is running: a TUI left open restarts OpenCode's background
service within seconds, and in one run a second session edited the same repo
at the same time. Without `RESET=1` the repo must already be clean.

Used for the Test B repeats. Adding this general line to the task's working
rules took the one requirement the model kept missing from 3 of 7 runs to
4 of 4:

> Before finishing: re-read the task. For each numbered step, quote every
> clause in the task's exact words (do not paraphrase), and next to each, the
> line of your code that does it. If a clause has no matching line, fix it,
> then run `swift test` again.

With "check every clause against your diff" instead, it was 3 of 4: the model
paraphrased the clause it had missed and then reported it as done.

## Copying models between the nodes

huggingface_hub 2.x stores weights in a shared `~/.cache/huggingface/hub/blobs/`
and leaves only symlinks in each model folder. Plain `rsync -a` copies the links
and not the weights. Use `-L`, and go over Thunderbolt:

```bash
rsync -aL ~/.cache/huggingface/hub/models--ORG--NAME 192.168.0.2:.cache/huggingface/hub/
```
