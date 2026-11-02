"""Frozen-shape GPU control: distinguish short host-launch noise from regressions."""

import argparse
import importlib
import json
import statistics
from pathlib import Path

import flag_gems
import torch


def main(output):
    module = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
    report = {
        "bucket_enabled": module._BUCKET_MM_NT_M,
        "diagnostic_only": True,
        "cases": [],
    }
    for index, (n, k, m) in enumerate(
        (n, k, m)
        for n, k in [(2560, 2048), (2048, 2048), (12288, 2048), (2048, 6144)]
        for m in (2007, 2008, 2009)
    ):
        torch.manual_seed(20261102 + index)
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.02
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).T * 0.02
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
        report["cases"].append(
            dict(
                shape=[m, n, k],
                samples_ms=samples,
                median_ms=statistics.median(samples),
            )
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
