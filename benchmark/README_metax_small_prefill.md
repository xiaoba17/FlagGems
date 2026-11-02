# MetaX small-Prefill tuning buckets (opt-in)

This extends the existing `FLAGGEMS_METAX_MM_BUCKET_M=1` policy. Enable
`FLAGGEMS_METAX_MM_SMALL_BUCKET_M=1` before importing FlagGems to also share
selected configurations for `64 < M < 1024` in groups of at most 32 rows.
Both switches default off. With the new switch off, existing keys are unchanged.

M<=64 keeps exact keys, and M>=1024 keeps its existing large-bucket keys.
The small buckets have a separate negative-key namespace: M=1023 cannot collide
with M=1024, where configuration pruning changes. N/K, dtype keys, actual shapes,
launch masks, and raw benchmark keys are unchanged. This is not request padding
or a new matmul kernel. Other workloads with Decode batches above 64 can also
use the policy; the name does not imply automatic identification of Prefill.

Keep the existing Triton functions and decorator source positions unchanged.
MetaX Triton includes source locations in cache identities; the strategy helper
is imported after the existing kernel definitions for that reason.

## Development validation, MetaX C500

The original 4K test encountered exact M=541 and 523 after large buckets had
been prepared. This feature addresses that remaining tuning mechanism, not the
sorting kernel or sampling parameters.

Two processes used identical source and independent cache copies of the same
snapshot, toggling only the small-bucket switch. They ran 58 GPU boundary tests,
94 probe cases (BF16/FP16/FP32 and masking/pruning boundaries), and 32 frozen-shape
graph controls each. All numerical checks passed. The 32 performance shapes use
M=523/524/541/542/543/544/545/546 and the four real model N/K combinations.

| Diagnostic | Exact small-M keys | Small buckets |
|---|---:|---:|
| New tuning benchmarks, 32 real shapes | 224 | 56 |
| Median first call, 24 subsequent same-bucket shapes | 6566.15 ms | 2.23 ms |

The first shape in each bucket still requires tuning. All 32 first-call samples
are retained; the 24-case statistic explicitly excludes the two bucket leaders
per N/K. Graph-control median-time changes across all shapes range from -0.25%
to +0.21%, with five raw samples retained per shape. This supports a cold-tuning
fix, not a claim of faster steady-state GEMM kernels or zero overhead.

Relevant commands (run both switch values in fresh processes with independently
copied caches, not sequentially against one shared mutable cache):

```bash
python tests/test_metax_mm_tuning_policy_cpu.py -v
python -m pytest tests/test_mm.py -k metax_mm_nt_bucket --ref cpu -q
python benchmark/bench_metax_mm_buckets.py probe.json --small-prefill
python benchmark/bench_metax_mm_bucket_graph.py graph.json --small-prefill
```

The validation above used an explicitly recorded development snapshot. Fixed
commit combination tests, official four-round throughput/TTFT (skip first, mean
last three), and all 105 Math-500 Level-3 questions remain required before merge.
Do not reuse old accuracy results or select favorable rounds. Tianshu is untested.
