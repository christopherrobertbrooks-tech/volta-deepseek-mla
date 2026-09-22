# PR #26404 — testing the reviewer's suggested tile-kernel fix

The reviewer asked the PR author to cover the tile kernel rather than gate on
architecture, supplied a patch, and asked for confirmation "by either compiling
the code for Pascal or by temporarily changing the kernel selection logic".

We compiled for Pascal and ran it on real Pascal hardware.

## What was built

Base: PR branch `timkhronos:FA-MMA-GQA5` at `cad375625`.

Two changes on top, both from the review:

1. `ggml/src/ggml-cuda/fattn-tile.cuh` — replace
   `GGML_ABORT("flash-attn tile (192/128): expected GQA ratio multiple of 8")`
   with `launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);`
2. `ggml/src/ggml-cuda/fattn.cu` — remove the
   `gqa_ratio % 8 != 0 && !turing_mma_available(cc)` gate entirely, which the
   reviewer said should then be unnecessary.

Diff: 2 insertions, 6 deletions.

Built with `-DCMAKE_CUDA_ARCHITECTURES="61;70"`, CUDA 12.9.86, GCC 13.3.0.
Verified the binary really contains both: `cuobjdump --list-elf` reports 141
`sm_61` sections and 141 `sm_70` sections, nothing else.

## Result

`test-backend-ops -o FLASH_ATTN_EXT -p "hsk=192"`

| card | compute cap | OK | FAIL |
| :--- | :--- | ---: | ---: |
| **GeForce GTX 1070** (Pascal) | 6.1 | **66** | **0** |
| **Tesla V100-PCIE-32GB** (Volta) | 7.0 | **66** | **0** |

Both report `2/2 backends passed`.

GQA ratios exercised: `nr23=[1,1] [4,1] [5,1] [8,1] [16,1]` — **1, 4 and 5 are
not multiples of 8**, i.e. exactly the cases that previously hit the abort, now
running through the tile kernel on pre-Turing hardware.

Neither card has `turing_mma_available`, so both take the tile path.

## Pre-existing issue, unrelated to the patch

An **unfiltered** `-o FLASH_ATTN_EXT` run aborts on the V100 inside
`ggml_cuda_flash_attn_ext_mma_f16_case<320, 256, 2, 32>` — head size 320/256 in
the *MMA* kernel, not 192/128 in the tile kernel.

**This is not caused by the patch.** Unpatched master (`c550d2f60`) built for
sm_70 crashes identically at the same call. It is why the runs above are
filtered to `hsk=192`.

## What was NOT established

A before/after diff of which individual cases newly pass. The unpatched control
is at a different commit whose test harness emits different case strings
(`kv_view`, `n_kv_max` fields), so the two logs are not directly comparable and
an attempted diff produced meaningless "regressions". The control OK count is
included below for context only, not as a paired comparison.

| log | OK | FAIL |
| :--- | ---: | ---: |
| `pascal-gtx1070-hsk192.log` | 66 | 0 |
| `volta-v100-hsk192.log` | 66 | 0 |
| `control-unpatched-v100-hsk192.log` (master, different commit) | 36 | 0 |
