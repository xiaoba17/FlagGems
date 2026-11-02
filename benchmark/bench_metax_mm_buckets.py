"""Bounded cold/steady matrix-multiply comparison, not an end-to-end score."""

import argparse
import importlib
import json
import os
import statistics
import time
from pathlib import Path

import flag_gems
import torch
from flag_gems.utils.libentry import LibTuner


def main(output, start_index=0):
    module = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
    report = {
        "bucket_enabled": module._BUCKET_MM_NT_M,
        "cache_dir": os.environ.get("FLAGGEMS_CACHE_DIR"),
        "module": module.__file__,
        "cases": [],
        "tuner_events": [],
    }
    if start_index:
        report = json.loads(output.read_text())
        assert len(report["cases"]) == start_index
        assert report["bucket_enabled"] == module._BUCKET_MM_NT_M
    original = LibTuner.run

    def observed(self, *args, **kwargs):
        call_args = {**dict(zip(self.arg_names, args)), **kwargs}
        started = time.perf_counter()
        result = original(self, *args, **kwargs)
        report["tuner_events"].append(
            {
                "name": self.__name__,
                "ms": (time.perf_counter() - started) * 1000,
                "key": {k: call_args.get(k) for k in ("M", "N", "K")},
                "benchmark_success_count": self.benchmark_success_count,
                "benchmark_cache_hit_count": self.benchmark_cache_hit_count,
            }
        )
        return result

    LibTuner.run = observed
    shapes = [
        (m, n, k, torch.bfloat16, True)
        for n, k in [(2560, 2048), (2048, 2048), (12288, 2048), (2048, 6144)]
        for m in (2007, 2008, 2009)
    ]
    shapes += [
        (m, 96, 65, dtype, False)
        for dtype in (torch.bfloat16, torch.float16, torch.float32)
        for m in (1, 31, 1023, 1024, 1025, 2016, 2017, 2047, 2048, 2049)
    ]
    shapes += [
        (m, 2560, 2048, torch.bfloat16, False) for m in (2016, 2017, 2047, 2048, 2049)
    ]
    try:
        for index, (m, n, k, dtype, measured) in enumerate(shapes):
            if index < start_index:
                continue
            torch.manual_seed(20261102 + index)
            a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.02
            b = torch.randn((n, k), device="cuda", dtype=dtype).T * 0.02
            # Native reference is computed outside the FlagGems dispatch scope.
            if dtype == torch.float32:
                # GPU FP32 mm may use reduced-precision hardware math. These
                # small boundary cases use an independent CPU FP64 reference.
                reference = torch.mm(a.cpu().double(), b.cpu().double()).to(a)
            else:
                reference = torch.mm(a, b)
            torch.cuda.synchronize()
            event_begin = len(report["tuner_events"])
            started = time.perf_counter()
            with flag_gems.use_gems(include=["mm"]):
                result = torch.mm(a, b)
                torch.cuda.synchronize()
                cold_ms = (time.perf_counter() - started) * 1000
                samples = []
                if measured:
                    for _ in range(5):
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        start.record()
                        for _ in range(8):
                            torch.mm(a, b)
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) / 8)
            atol, rtol = {
                torch.bfloat16: (3e-4, 0.02),
                torch.float16: (1e-5, 0.002),
                torch.float32: (1e-6, 1e-4),
            }[dtype]
            torch.testing.assert_close(result, reference, atol=atol, rtol=rtol)
            row = dict(
                shape=[m, n, k],
                dtype=str(dtype),
                measured=measured,
                first_call_ms=cold_ms,
                steady_samples_ms=samples,
                steady_median_ms=statistics.median(samples) if samples else None,
                max_abs_error=(result.float() - reference.float()).abs().max().item(),
                tuner_events=report["tuner_events"][event_begin:],
                correctness="passed",
            )
            report["cases"].append(row)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    finally:
        LibTuner.run = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    main(args.output, args.start_index)
