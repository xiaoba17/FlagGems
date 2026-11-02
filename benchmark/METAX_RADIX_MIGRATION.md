# MetaX stable radix migration

This feature only moves the previously tested tiled FP32 row sorter from
vllm-plugin-FL commit `9658d99eba5aa0f152c4270c856fb3ecaade4d07` into the MetaX
FlagGems backend. All seven sort/kernel function ASTs are unchanged.

The backend exports `flag_gems.radix_sort_rows`. This is an explicit API, not
a replacement of the generic `torch.sort` or FlagGems sort registrations.
The companion plugin branch `feat/metax-sampler-flaggems` retains the opt-in
sampling gate, unchanged softmax/cumsum/mask/scatter, and original fallback.
It falls back if this API is unavailable in the installed FlagGems version.

This branch is independent from `feat/metax-mm-tuning-buckets`. Moving the
sorter does not fix GEMM cold-shape tuning, model prefill or request queueing.
No end-to-end throughput or TTFT gain is claimed from migration itself.

Validation requirements: stable values and indices, unchanged logits/tokens/RNG,
fallback coverage, and migration-only sampling microbenchmark. Formal 4K/16K
and Math-500 must later run against fixed companion commits without a profiler.
