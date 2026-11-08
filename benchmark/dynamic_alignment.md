# Dynamic valid-token alignment benchmark

`dynamic_alignment.py` measures the public CCE forward/backward API while the
number of valid labels changes. It uses synthetic BF16 inputs and reports the
first encounter of each count separately from warmed repetitions. It does not
measure model training TPS or establish gradient equivalence.

Use the same CUDA-compatible PyTorch/Triton environment and GPU for both runs.
The alignment exclusion option was tested with Triton 3.8; compatibility with
older Triton releases is not established by this example. Use an existing GPU
environment that already contains compatible PyTorch, Triton and NumPy. The
commands below select that interpreter without resolving this checkout as a uv
project or installing GPU dependencies.

From this checkout, create a parent worktree and run the **same script** twice:

```bash
git worktree add --detach ../cce-parent 87a86aba72cfd2f0d8abecaf81c13c4528ea07d8
CUDA_VISIBLE_DEVICES=0 uv run --no-project --python /path/to/gpu-env/bin/python \
  benchmark/dynamic_alignment.py \
  --source ../cce-parent --output parent.json
CUDA_VISIBLE_DEVICES=0 uv run --no-project --python /path/to/gpu-env/bin/python \
  benchmark/dynamic_alignment.py \
  --source . --output patched.json
```

Each invocation creates an empty temporary Triton cache **before importing
PyTorch, Triton or CCE**, overriding `TRITON_CACHE_DIR` and removing the temporary
cache at exit. A custom `TRITON_CACHE_MANAGER` is rejected, as is
`CCE_AUTOTUNE` other than 0. The JSON prints both the valid-count and lock-count
(`ceil(B/128)`) divisibility classes; the defaults exercise all four pairs. The imported CCE path,
changed-file hashes, package/runtime versions, device, runner hash and input
hashes are included in the JSON. The script asserts it imported the requested
checkout rather than an installed copy. An output file must not already exist.

The small default is 2048 tokens, hidden 512, vocabulary 8192 with valid counts
1925, 1805, 1808, 1936. For the original geometry and representative count sequence,
add `--original-sized` to **both** invocations: 16384 tokens, hidden 2880,
vocabulary 201088 and counts 6308, 6093, 6080, 6320. This requires substantially more
GPU memory. `--rows`, `--hidden`, `--vocab` and `--valid-counts` allow other shapes;
use identical arguments and seed for both sources. Labels are randomized but
repeatable, not retained training examples. Count order matters for compilation.

Check paired identity and print the two timing ratios:

```python
import json

parent = json.load(open("parent.json"))
patched = json.load(open("patched.json"))
for key in ("geometry", "valid_counts", "seed", "labels_sha256", "hidden_sha256",
            "classifier_sha256", "runner_sha256", "math", "torch", "triton", "device"):
    assert parent[key] == patched[key], key
for key in ("first_pass_total_ms", "warm_pass_median_ms"):
    print(key, "parent:", parent[key], "patched:", patched[key],
          "parent/patched:", parent[key] / patched[key])
```

`first_pass` contains the chronological initial forward/backward calls and
includes CCE JIT costs (autotuning is disabled); import, input generation and validation are
outside the timing. `warm_passes` contains five subsequent passes by default
(`--warm-repeats`). Both use CUDA synchronization around wall timing, include
backward, and clear gradients before each call. Every call checks finite loss
and gradients **after** timing. No optimizer changes the inputs. The options
use sum reduction, disabled gradient filtering and FP32 hidden-gradient
accumulation consistently across sources.

A prior original-size GB200 investigation observed about 29% lower cold elapsed
time and essentially unchanged warmed latency. That used a longer real-label
chronology; it is historical context, not a promised result of this synthetic
four-count program.

The intended improvement is fewer cold specializations for changing alignment;
warmed kernel execution may be unchanged. First-pass totals also include shared
initial compilation and are hardware/runtime dependent. Inspect individual
counts as well as totals, and repeat independent process pairs to gauge noise.
Do not report a guaranteed speedup from one ratio or treat finite gradients as
an equivalence test.

CPU-only label/argument checks, without a CUDA workload:

```bash
uv run --no-project --python /path/to/gpu-env/bin/python \
  benchmark/dynamic_alignment.py --check-inputs --output inputs.json
```
