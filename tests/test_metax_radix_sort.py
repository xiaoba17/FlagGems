"""MetaX explicit tiled sort: stable indices must match existing FlagGems."""

import pytest
import torch

import flag_gems


@pytest.mark.sort
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX explicit kernel")
@pytest.mark.parametrize("batch", [1, 8, 64])
@pytest.mark.parametrize(
    "distribution", ["normal", "bf16", "ties", "constant", "masked"]
)
def test_metax_radix_sort_stable_indices(batch, distribution):
    x = torch.randn((batch, 130560), device=flag_gems.device)
    if distribution == "bf16":
        x = x.bfloat16().float()
    elif distribution == "ties":
        x = x.round()
    elif distribution == "constant":
        x.zero_()
    elif distribution == "masked":
        x[:, ::3] = -float("inf")
    original = x.clone()
    # Match serving: MetaX native torch lacks UInt32 copy used by FG sort.
    with flag_gems.use_gems():
        expected, expected_ids = flag_gems.sort(x, dim=-1)
        actual, actual_ids = flag_gems.radix_sort_rows(x)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_ids, expected_ids)
    assert torch.equal(x, original)


@pytest.mark.sort
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX explicit kernel")
@pytest.mark.parametrize("block", [256, 512, 1024])
def test_metax_radix_sort_blocks(block):
    x = torch.randn((3, 1024), device=flag_gems.device).round()
    with flag_gems.use_gems():
        expected, expected_ids = flag_gems.sort(x, dim=-1)
        actual, actual_ids = flag_gems.radix_sort_rows(x, block=block)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_ids, expected_ids)
