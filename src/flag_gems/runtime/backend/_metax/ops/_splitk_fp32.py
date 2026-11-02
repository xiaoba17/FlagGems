"""Promote cross-partition accumulation, preserving BF16 inputs and outputs."""

import torch


def promote_splitk_accumulation(original):
    def splitk_mm_fp32(a, b, c, m, n, k):
        if c.dtype != torch.bfloat16:
            return original(a, b, c, m, n, k)
        # The original kernel already computes each partition in FP32. Keep
        # the inter-partition atomic accumulation FP32 as well, then round once.
        accumulator = torch.zeros_like(c, dtype=torch.float32)
        original(a, b, accumulator, m, n, k)
        c.copy_(accumulator)
        return c

    return splitk_mm_fp32
