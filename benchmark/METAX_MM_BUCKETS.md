# MetaX mixed-prefill tuning buckets

Parent: competition FlagGems commit `a7620cc191a0b42e040194622c5758b22a7a25dc`.
Opt-in at process startup: `FLAGGEMS_METAX_MM_BUCKET_M=1`. Default is off.

The NT matrix-multiply configuration cache shares nearby M>=1024 shapes in
groups of at most 32 rows. Small M, N and K retain exact keys. Negative encoded
M keys separate the new policy from pre-existing exact-shape configurations.
The actual kernel arguments, heuristics, masks, dtype and mathematical
computation are not changed. Raw per-configuration benchmark keys remain exact.

This reduces repeated online tuning; it does not remove the first tuning of a
new bucket, all possible JIT compilation, or model prefill computation. Selecting
one configuration for a bucket still requires steady-state regression checks.
Do not switch the flag in a process which already cached kernel dispatches.

Development validation on C500: baseline and candidate each passed 47 cases,
including real BF16 model shapes, FP16/FP32 and M/K/N boundary masks. The first
12 measured shapes have 60 versus 20 actual tuning benchmarks. Subsequent
same-bucket shapes reduce first-call stalls from roughly 5 seconds to 2–6 ms.
First-call measurements and all steady samples are retained, including slower
ones. A 12-shape CUDA Graph control checks GPU kernel timing separately from
short host launch measurements; this is not an official throughput score.

The small FP32 cases use an independent CPU FP64 reference: native GPU FP32
matmul showed reduced-precision reference error. No assertion was loosened.

End-to-end mixed-serving regression, fixed-commit official 4K/16K/TTFT and
Math-500 acceptance are pending. This feature is unmerged and independent from
the stable-sort migration. Complete local raw evidence is indexed under
`profiling/results/metax-optimization-20261102/README.md` in the workspace.
