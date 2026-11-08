"""Compare cold/warm CCE alignment behavior in separate source-checkout invocations."""

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--original-sized", action="store_true", help="16384x2880, vocab201088")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--vocab", type=int)
    parser.add_argument(
        "--valid-counts", help="Comma-separated positive counts, in encounter order"
    )
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-inputs", action="store_true", help="CPU labels/provenance only")
    args = parser.parse_args()
    defaults = (16384, 2880, 201088) if args.original_sized else (2048, 512, 8192)
    rows, hidden, vocab = (
        v if v is not None else d for v, d in zip((args.rows, args.hidden, args.vocab), defaults)
    )
    raw_counts = args.valid_counts or (
        "6308,6093,6080,6320" if args.original_sized else "1925,1805,1808,1936"
    )
    try:
        counts = [int(x) for x in raw_counts.split(",")]
    except ValueError:
        parser.error("valid counts must be comma-separated integers")
    if min(rows, hidden, vocab, args.warm_repeats) <= 0 or not counts:
        parser.error("geometry, repeats and valid counts must be positive")
    if any(n <= 0 or n > rows for n in counts):
        parser.error("each valid count must be in [1, rows]")
    if args.output and args.output.exists():
        parser.error("output already exists; use a new filename")
    source = args.source.resolve()
    files = [
        source / "cut_cross_entropy" / name for name in ("cce_lse_forward.py", "cce_backward.py")
    ]
    if not all(p.is_file() for p in files):
        parser.error("source must be a CCE checkout")
    if os.environ.get("TRITON_CACHE_MANAGER"):
        parser.error("unset TRITON_CACHE_MANAGER to use an isolated standard cache")
    if os.environ.get("CCE_AUTOTUNE", "0") != "0":
        parser.error("set CCE_AUTOTUNE=0 to isolate alignment specialization")
    # Create the cache BEFORE torch, Triton or CCE imports, with a new process per source.
    cache = tempfile.TemporaryDirectory(prefix="cce-alignment-")
    os.environ["TRITON_CACHE_DIR"] = cache.name
    assert not any(Path(cache.name).iterdir())
    import torch

    torch.manual_seed(args.seed)
    labels_cpu = torch.randint(vocab, (rows,), dtype=torch.long)
    order = torch.randperm(rows)
    targets = []
    for n in counts:
        labels = torch.full((rows,), -100, dtype=torch.long)
        labels[order[:n]] = labels_cpu[order[:n]]
        assert int((labels != -100).sum()) == n
        assert torch.all((labels == -100) | ((labels >= 0) & (labels < vocab)))
        targets.append(labels)

    def digest(tensor):
        data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    result = dict(
        geometry=dict(rows=rows, hidden=hidden, vocab=vocab),
        valid_counts=counts,
        alignment_classes=[
            dict(
                B=n,
                locks=(n + 127) // 128,
                B_divisible_by_16=n % 16 == 0,
                locks_divisible_by_16=((n + 127) // 128) % 16 == 0,
            )
            for n in counts
        ],
        CCE_AUTOTUNE="0",
        seed=args.seed,
        source=str(source),
        torch=torch.__version__,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        labels_sha256=[digest(t) for t in targets],
        cache_directory=cache.name,
        cache_initially_empty=not any(Path(cache.name).iterdir()),
        cache_retained=False,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        check_inputs_only=args.check_inputs,
    )
    if not args.check_inputs:
        if not torch.cuda.is_available():
            parser.error("CUDA is required for timing; --check-inputs performs CPU checks only")
        torch.cuda.set_device(0)
        torch.set_float32_matmul_precision("highest")
        sys.path.insert(0, str(source))
        import triton

        import cut_cross_entropy
        from cut_cross_entropy import linear_cross_entropy

        assert Path(cut_cross_entropy.__file__).resolve().parent == files[0].parent
        e = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        c = torch.randn((vocab, hidden), device="cuda", dtype=torch.bfloat16) * 0.02
        c.requires_grad_()
        targets = [t.cuda() for t in targets]
        options = dict(
            ignore_index=-100,
            reduction="sum",
            shift=False,
            impl="cce",
            filter_eps=None,
            accum_e_fp32=True,
        )
        result.update(
            cce_path=cut_cross_entropy.__file__,
            cce_version=cut_cross_entropy.__version__,
            triton=triton.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            hidden_sha256=digest(e),
            classifier_sha256=digest(c),
            math=dict(dtype="bfloat16", matmul_precision="highest", **options),
        )

        def evaluate(index):
            e.grad = c.grad = None
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss = linear_cross_entropy(e, c, targets[index], **options)
            loss.backward()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Validation stays OUTSIDE the timed forward/backward interval.
            assert (
                torch.isfinite(loss)
                and torch.isfinite(e.grad).all()
                and torch.isfinite(c.grad).all()
            )
            return dict(valid_count=counts[index], wall_ms=elapsed_ms, loss=loss.item())

        result["first_pass"] = [evaluate(i) for i in range(len(counts))]
        result["warm_passes"] = [
            [evaluate(i) for i in range(len(counts))] for _ in range(args.warm_repeats)
        ]
        result["first_pass_total_ms"] = sum(r["wall_ms"] for r in result["first_pass"])
        result["warm_pass_median_ms"] = statistics.median(
            sum(r["wall_ms"] for r in iteration) for iteration in result["warm_passes"]
        )
        result["finite_checks_passed"] = True
        result["scope"] = (
            "Synthetic CCE forward+backward only; not full-model TPS or gradient parity"
        )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(payload)
    else:
        print(payload, end="")
    cache.cleanup()


if __name__ == "__main__":
    main()
