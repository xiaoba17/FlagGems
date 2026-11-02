"""Cross-partition BF16 rounding regression and dtype/out contracts."""

import importlib
import os
import sys

import pytest
import torch

import flag_gems


@pytest.mark.skipif(
    flag_gems.vendor_name != "metax", reason="MetaX accumulation policy"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_splitk_promotion_contract(dtype, strided):
    module = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops._splitk_fp32"
    )
    seen = []

    def original(a, b, c, m, n, k):
        seen.append((a, b, c.dtype))
        c.fill_(0.125)
        return c

    a, b = torch.empty((2, 4), dtype=dtype), torch.empty((4, 3), dtype=dtype)
    out = torch.empty((2, 6 if strided else 3), dtype=dtype)
    if strided:
        out = out[:, ::2]
    result = module.promote_splitk_accumulation(original)(a, b, out, 2, 3, 4)
    assert result is out and result.dtype == dtype
    assert seen[0][0] is a and seen[0][1] is b
    assert seen[0][2] == (torch.float32 if dtype == torch.bfloat16 else dtype)
    assert torch.equal(result, torch.full_like(out, 0.125))


@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX Split-K regression")
@pytest.mark.parametrize("m", [1, 8, 16, 32])
@pytest.mark.parametrize("seed", [20261121, 20261122, 20261123])
def test_splitk_promoted_against_fp64(m, seed):
    if os.environ.get("FLAGGEMS_METAX_SPLITK_FP32_ACCUM") != "1":
        pytest.skip("Run with the opt-in enabled in a fresh process")
    module = sys.modules[flag_gems.mm.__module__]
    assert module.splitk_mm.__name__ == "splitk_mm_fp32"
    torch.manual_seed(seed)
    a = torch.randn((m, 6144), device=flag_gems.device, dtype=torch.bfloat16) * 0.02
    b = (
        torch.randn((2048, 6144), device=flag_gems.device, dtype=torch.bfloat16).T
        * 0.02
    )
    expected = torch.mm(a.cpu().double(), b.cpu().double())
    with flag_gems.use_gems(include=["mm"]):
        result = torch.mm(a, b)
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result.cpu().double(), expected, atol=3e-4, rtol=0.02)


@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX serving out/graph")
@pytest.mark.parametrize("graph", [False, True])
def test_splitk_promoted_out_and_graph(graph):
    if os.environ.get("FLAGGEMS_METAX_SPLITK_FP32_ACCUM") != "1":
        pytest.skip("Run with the opt-in enabled in a fresh process")
    torch.manual_seed(20261121)
    a = torch.randn((16, 6144), device=flag_gems.device, dtype=torch.bfloat16) * 0.02
    b = (
        torch.randn((2048, 6144), device=flag_gems.device, dtype=torch.bfloat16).T
        * 0.02
    )
    out = torch.empty((16, 4096), device=flag_gems.device, dtype=torch.bfloat16)[:, ::2]
    expected = torch.mm(a.cpu().double(), b.cpu().double())
    with flag_gems.use_gems(include=["mm_out"]):
        result = torch.mm(a, b, out=out)
        torch.cuda.synchronize()
        if graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                result = torch.mm(a, b, out=out)
            capture.replay()
            torch.cuda.synchronize()
    assert result is out and not result.is_contiguous()
    torch.testing.assert_close(result.cpu().double(), expected, atol=3e-4, rtol=0.02)
