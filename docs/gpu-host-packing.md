# GPU host packing changes

This document describes the worker-private packing optimization ported to
LibRI on 2026-11-06, based on standalone commit
`df004e5787c3354b391aff83278ea61db2b58af1`. The original performance experiment
used LibRPA commit `4fd3b1568857e28ee8726dd37d439a1335f75b75` and its bundled LibRI.
The production timings below describe that experiment, not a new benchmark
of the standalone repository or a pure GPU-kernel measurement.

## What changed in the code

| File under `include/RI/ri/` | Change and purpose |
| --- | --- |
| `LRI-cal_loop3_GPU.hpp` | Remove 40 packing-loop OpenMP critical directives. Keep the blocks, contributions, shapes, symmetry filters and GEMM calls. |
| `gpu/GPU_Data_Pack.h` | Add `Thread_Batch`, `Pair_Packs` and `HostMemoryResource`: collect metadata per worker, then resolve unique pair offsets once. |
| `gpu/GPU_Data_Input.h` | Record input tensor references while traversing; merge duplicate pairs, then copy each unique tensor into one contiguous host buffer during `upload()`. |
| `gpu/GPU_Data_Mul.h` | Collect first-stage pairs privately, merge their offsets, and use the merged map for second-stage lookup. Keep both host pointer tables alive. |
| `gpu/GPU_Data_Output.h` | Collect output pairs privately, merge once, and retain the host pointer table while its upload is queued. |
| `gpu/GPU_Data_Tmp.h` | Store one element count per contribution instead of a full `Pack`. Construct bounded temporary-buffer segments without duplicating full metadata arrays. |
| `gpu/Dim.h` | Collect dimensions per worker and flatten into an owned host array before upload. |

The standalone port preserves its existing queue-aware allocation/free calls
and the public `ddla::ddla` CMake target from the LibDDLA API update. It does
not replace `include/RI/global/gpu/GPU_Backend.h` with the older experiment's
adapter. This checkout has no tracked `cal_loop3_GPU.py` generator; the header
is edited directly here.

## Before and after

Previously, packing loops entered shared critical sections for pair-map
insertion, input copying and batch append operations. Other workers waited
while a worker updated shared metadata. Tensor data copying contributed to
the measured packing-loop time.

Now each worker appends operands and dimensions to its own padded chunk and
updates its own pair map. After traversal, upload preparation merges the maps,
assigns one offset per unique pair, and constructs flat device pointer tables.
Input tensor values are copied only after duplicate pairs have been resolved.
Full tensor buffers are not replicated per worker.

Every related operand, temporary and dimension batch is concatenated in the
same worker-ID order. Contribution multiplicity is preserved. Scheduling may
change the order of contributions, so bitwise numerical reproducibility is
not promised for arbitrary workloads.

## Ownership and memory rules

- Input references point into caller-owned tensor maps. Those tensors must
  stay alive and unchanged until `Input::upload()` finishes copying them.
- Global `ptrList` entries own their shape and offset metadata. Worker records
  receive the resolved global offset before device pointers are constructed.
- Consume all references to worker records before clearing their maps and
  releasing their arenas. Clear maps before calling the arena's `release()`.
- `Thread_Batch::release()` frees its backing vectors but retains the batch
  count, because subsequent GEMM calls still use `size()`.
- Flat input values, pointer tables and dimension arrays are member-owned and
  remain alive through queued transfers. A function-local upload source that
  is destroyed on return is insufficient for asynchronous backends.
- Scoped POSIX mappings return released private metadata to the OS instead
  of retaining it in malloc thread arenas. Growing vectors map allocations
  of at least 64 KiB; map arenas use the resource without that threshold.
  Over-aligned allocations and non-POSIX systems use the standard resource.
  Resource identity is preserved when releasing polymorphic vectors.
- Temporary batches store `size_t` sizes (8 bytes on the measured host)
  instead of approximately 64-byte `Pack` records. `Tmp::insert()` returns a
  `Pack` by value; existing block-local `const Pack&` bindings extend that
  temporary's lifetime. The caller's 5 GiB device-temporary cap is preserved;
  an individual tensor larger than the supplied cap throws.

These objects follow the existing one-upload-per-contraction lifecycle;
reuse after release is not a supported new interface. GPU packing requires
C++17 (`std::pmr`). CPU-only library requirements are unchanged.

## Why the input preparation timer increased

The input timer surrounds `Input::upload()`. Its responsibilities changed:

1. Merge worker maps, deduplicate pairs and assign global offsets.
2. Build the unique-copy list and allocate/resize the flat host buffer.
3. Copy unique tensor values in parallel.
4. Allocate device storage, construct pointer tables and enqueue uploads.
5. Release consumed worker metadata.

Part of this work previously ran inside the packing loop. In addition,
`h_data[0].resize(totalSize)` initializes the buffer before copying tensor
values over it: this is a real extra initialization pass in the current code.
Map merging and flat-array construction also remain host work.

The experiment did not separately time these five substeps. Consequently the
increase cannot be attributed quantitatively to copying, initialization,
merging or GPU API waiting. Further optimization needs measurements inside
`Input::upload()`; the current evidence establishes their combined cost only.

## Measured V100 comparison

HPC3, one V100-SXM2-32GB used, 16 physical CPU cores, one MPI rank,
128-carbon Gamma G0W0, six frequencies, Coulomb threshold zero. Three GPUs
were allocated because of the site's CPU/GPU ratio; two were unused.
Original values are the mean of two runs; final values are one corrected run.
Wall seconds measure elapsed scopes; process CPU seconds sum process thread
work and are not elapsed time or core-hours.

| Scope | Original wall (s) | Final wall (s) | Original CPU (s) | Final CPU (s) |
| --- | ---: | ---: | ---: | ---: |
| Packing loops | 629.464 | 15.278 | 9372.731 | 213.563 |
| Input preparation/upload | 49.020 | 184.048 | 68.418 | 402.039 |
| Mul first preparation/upload | 1.842 | 4.869 | 1.836 | 4.857 |
| Mul second preparation/upload | 1.240 | 1.061 | 1.301 | 1.104 |
| Output preparation/upload | 1.476 | 2.730 | 3.852 | 6.085 |
| Temporary preparation/upload | 6.110 | 1.459 | 10.656 | 6.067 |
| Dimension preparation/upload | 1.160 | 5.658 | 1.203 | 5.682 |
| **All preparation above** | **690.313** | **215.103** | **9459.997** | **639.398** |
| **Whole driver** | **1107.792** | **618.551** | **9975.621** | **1141.141** |

Combined preparation improved 3.21x and the whole driver 1.79x. The larger
input timer is already included in both totals. These upload scopes include
host API waiting and must not be described as CUDA-event kernel timings.

Peak process RSS was 21,232,408 / 21,347,104 KiB in the original runs and
21,169,188 KiB in the final run. The single final observation was below both
originals; it does not guarantee lower memory for every workload. Earlier
heap-backed and map-only candidates were rejected for higher production RSS.

## Validation and limits

The production experiment completed in HPC3 job `3937521`, with final build
job `3938035`. All 453 printed QP rows (states 60-512), all eight columns,
were finite and identical to both originals at printed precision. Focused
tests checked all 18 contractions plus their combined sum, real/complex
double, restricted/empty output sectors and signed accumulation against an
independent CPU reference with `1e-12 * (1 + abs(reference))` tolerance.

Packing primitives passed for all four real/complex scalar types at 1, 4 and
16 OpenMP threads. CPU Release and address/undefined-behavior sanitizer checks
covered duplicate keys, ordering, released metadata, allocation thresholds,
zero-size and over-aligned allocation paths. A800 focused job `3836240` also
passed; its full production comparison was still pending when these results
were recorded. HIP, MAGMA and multi-rank execution were not validated by this
packing experiment.

For runnable repository tests, see
[`unittests/gpu_symmetry/README.md`](../unittests/gpu_symmetry/README.md).
Standalone-port validation is recorded separately below; the historical
production timings must not be presented as a fresh standalone benchmark.

## Standalone port validation record (2026-11-06)

All seven implementation headers match the validated candidate after
accounting only for the standalone backend's queue arguments/ownership and
line endings. `git diff --check` and local/remote SHA-256 parity passed.
The standalone checkout's existing backend adapter and contraction-reference
test were preserved. This port adds the two packing tests and their CMake
registrations while retaining public DDLA package discovery.

New standalone build job `3940328` and dependent GPU check `3940325` were
queued on HPC3 V100 partitions when this record was written. They must finish
successfully before describing this exact standalone revision as freshly
tested. Build/check output and the original source archive are retained in
`/data/home/renxg/app/development/libri_repo_validation` (`build.out`,
`gpu.out`, `original.tar`, `source.sha256`). The test scripts use the existing
DDLA 0.0.6 installation from the earlier API validation, not a new dependency
installation. These correctness checks are not production performance runs.
