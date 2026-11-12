# Cluster scripts

> Scripts for running DeepSeek-V4-Flash split across two Mac Studios (M3 Ultra,
> 96 GB each) over Thunderbolt 5 RDMA with MLX JACCL. Deploy by copying the
> scripts to `~/` on every node. Node 1 is "purple", node 2 "orange".

All commands run **on purple**, the first node in `~/m3-ultra-jaccl.json`.
`mlx.launch` starts the same script on both machines.

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
| `needle_test.py` | `~/cluster-env` | Needle-in-a-haystack correctness test, Qwen3-Coder-Next via mlx-lm |
| `dsv4_run.py` | `~/vlm-env` | Speed sweep + needle test for DeepSeek-V4-Flash via mlx-vlm |

The two venvs are separate on purpose. `~/cluster-env` has mlx-lm 0.32.2;
`~/vlm-env` has mlx-vlm 0.7.12. Both pin mlx 0.32.4, which must match on both
machines.

### vlm_cluster_server.py

Why it exists: mlx-vlm can split a model across machines (`sharded_load`) but
its own server cannot. Rank 0 serves HTTP and hands each request to the other
node over `mx.distributed`; both generate in lockstep.

```bash
nohup ~/cluster-env/bin/mlx.launch --backend jaccl --hostfile ~/m3-ultra-jaccl.json -- \
  $HOME/vlm-env/bin/python $HOME/vlm_cluster_server.py \
  --model mlx-community/DeepSeek-V4-Flash-4bit --port 8090 > ~/vlm-server.log 2>&1 &

curl -s localhost:8090/health
```

Client settings (Cline "OpenAI Compatible", Continue, curl):

```
base URL   http://<first-node>:8090/v1
model      mlx-community/DeepSeek-V4-Flash-4bit
API key    anything
```

Stop it: `pkill -f vlm_cluster_server.py` on purple (orange's half exits with
it; if not, run the same on orange).

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
  no value is guessed. A conflict on an optional parameter still fails parsing.
- **Warm-up before serving.** All ranks run a tiny generation after loading (about
  7 s; a normal tiny request takes 0.3 s). Without it, a request arriving the
  moment the server came up crashed both ranks with a Metal GPU timeout.
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

## Copying models between the nodes

huggingface_hub 2.x stores weights in a shared `~/.cache/huggingface/hub/blobs/`
and leaves only symlinks in each model folder. Plain `rsync -a` copies the links
and not the weights. Use `-L`, and go over Thunderbolt:

```bash
rsync -aL ~/.cache/huggingface/hub/models--ORG--NAME 192.168.0.2:.cache/huggingface/hub/
```
