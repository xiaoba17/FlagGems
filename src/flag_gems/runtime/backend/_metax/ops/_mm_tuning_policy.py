"""Selected-config keys only; never change actual shapes or benchmark keys."""

import os

_SMALL_PREFILL_BUCKETS = os.environ.get("FLAGGEMS_METAX_MM_SMALL_BUCKET_M", "0") == "1"
_SMALL_NAMESPACE = 1 << 32


def mm_nt_m_tuning_key(m):
    if _SMALL_PREFILL_BUCKETS and 64 < m < 1024:
        # In particular, 993..1023 must not share -1024 with large M=1024:
        # the kernel's config-pruning policy changes at that boundary.
        return -(_SMALL_NAMESPACE + ((m + 31) // 32) * 32)
    # Preserve the existing policy and its cached large-M configurations.
    return m if m < 1024 else -((m + 31) // 32) * 32
