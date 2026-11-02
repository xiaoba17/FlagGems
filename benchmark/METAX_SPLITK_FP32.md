# MetaX BF16 Split-K accumulation safety

Parent: competition commit `a7620cc191a0b42e040194622c5758b22a7a25dc`.
Opt-in at import: `FLAGGEMS_METAX_SPLITK_FP32_ACCUM=1`. Default is off.

The original Split-K kernel computes each partition in FP32 but atomically adds
it to BF16 output, rounding between partitions. A stopped cache-prewarm case
at `(M,N,K)=(16,2048,6144)` exposed errors against an independent CPU FP64
oracle. The original output had 2–9 strict mismatches over two ten-repeat probes;
native GPU mm had none under the same `atol=3e-4, rtol=0.02` test.

This feature uses a fresh FP32 cross-partition accumulator and casts to the
original BF16 output once. Input and output dtypes, output aliasing and strides
are preserved. Non-BF16 output retains the original path. FP32 atomics are still
order-dependent; the implementation does not promise bitwise determinism.

The opt-in hook is appended after existing definitions. All pre-existing
function ASTs and their source positions remain unchanged: MetaX Triton embeds
source positions into cache keys. This avoids unnecessary invalidation of
unmodified kernels. FP32-output variants still need their own legitimate tuning.

Development validation: 18 dtype/out and multi-shape/multi-seed FP64 tests,
plus 2 noncontiguous mm_out/CUDA Graph tests passed. The original failing input
passed the same strict test in ten promoted-accumulation repeats. RMS error
was about 5.21e-5, compared with about 1.27e-4 for the original kernel.

There is a measured tradeoff, not a throughput claim: one short-loop probe was
0.12336 ms original versus 0.14106 ms promoted. The correction remains separate
from MM tuning buckets and stable-sort migration. Wider serving performance,
model precision and fixed-commit official 4K/16K/TTFT acceptance are pending.
