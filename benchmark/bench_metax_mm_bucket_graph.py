"""Frozen-shape GPU control: distinguish short host-launch noise from regressions."""

import argparse
import importlib
import json
import statistics
from pathlib import Path

import torch

import flag_gems


def main(output, small_prefill=False):
    module = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
    report = {
        "bucket_enabled": module._BUCKET_MM_NT_M,
        "diagnostic_only": True,
        "cases": [],
        "small_prefill_probe": small_prefill,
        "small_bucket_enabled": importlib.import_module(
            "flag_gems.runtime.backend._metax.ops._mm_tuning_policy"
        )._SMALL_PREFILL_BUCKETS,
        "complete": False,
    }
    for index, (n, k, m) in enumerate(
        (n, k, m)
        for n, k in [(2560, 2048), (2048, 2048), (12288, 2048), (2048, 6144)]
        for m in (
            (523, 524, 541, 542, 543, 544, 545, 546)
            if small_prefill
            else (2007, 2008, 2009)
        )
    ):
        torch.manual_seed(20261102 + index)
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.02
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).T * 0.02
        reference = torch.mm(a, b)
        with flag_gems.use_gems(include=["mm"]):
            for _ in range(2):
                torch.mm(a, b)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):
                    result = torch.mm(a, b)
            samples = []
            for _ in range(5):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(16):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 128)
        assert result.shape == (m, n)
        torch.testing.assert_close(result, reference, atol=3e-4, rtol=0.02)
        report["cases"].append(
            dict(
                shape=[m, n, k],
                samples_ms=samples,
                median_ms=statistics.median(samples),
                correctness="passed",
            )
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--small-prefill", action="store_true")
    args = parser.parse_args()
    main(args.output, args.small_prefill)
