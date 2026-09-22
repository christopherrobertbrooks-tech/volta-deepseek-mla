# DeepSeek-V2-Lite: enabling flash attention is a large regression, on every GPU tested

**Corrected 2026-09-22.** The first version of this file called it a Volta
finding. It is not — see [the correction](#correction) at the end.

`-fa on` costs **80% of decode on a V100 and 52% on an RTX 4070**, with
byte-identical output and no warning. The cause is the model's attention shape,
not the GPU.

Model: `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` (9.65 GiB loaded, 15.71 B
params, `deepseek2`), upstream llama.cpp `c550d2f60` (b11096), CUDA 12.9.86.

## Measured

`llama-bench -ngl 99 -fa 0,1 -p 512 -n 128`, one card visible at a time.

| card | fa | pp512 | tg128 |
| :--- | :-- | ---: | ---: |
| Tesla V100-PCIE-32GB (sm_70) | **off** | **1836.3 ± 16.5** | **164.6 ± 0.4** |
| Tesla V100-PCIE-32GB | on | 1000.9 ± 5.5 | 33.6 ± 0.04 |
| RTX 4070 (sm_89) | **off** | **2789.3 ± 329.3** | **208.0 ± 0.9** |
| RTX 4070 | on | 1811.6 ± 2.1 | 100.8 ± 0.4 |

Decode loss: **-79.6% on Volta, -51.5% on Ada.** Prefill loss: -45.5% and
-35.1%.

Volta is hit harder, but both cards lose heavily. Correctness is unaffected:
the same prompt at `temp 0` gave byte-identical output either way (155.2 t/s
with FA off against 32.5 t/s with it on, on the V100).

## Mechanism

From the model's own metadata:

```
deepseek2.attention.key_length      192     <- DKQ
deepseek2.attention.value_length    128     <- DV
deepseek2.attention.head_count      16
deepseek2.attention.head_count_kv   16      <- so gqa_ratio = 1
```

In `ggml/src/ggml-cuda/fattn.cu`, the kernel selector has:

```c
const bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && ...;
...
case 192:
    if (V->ne[0] != 128 || !gqa_opt_applies) {
        return BEST_FATTN_KERNEL_NONE;
    }
    if (gqa_ratio % 8 != 0) {
        return BEST_FATTN_KERNEL_NONE;
    }
```

The head-size-192 path was written for models with GQA ratio 8 or 16 — the
source comment names MiMo-V2.5 variants. DeepSeek-V2-Lite has no GQA at all
(16 heads, 16 KV heads, ratio 1), so it fails both gates and **there is no CUDA
flash-attention kernel for this shape**.

There is also no vector kernel available as a fallback:

```c
// 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
const bool can_use_vector_kernel = ... && Q->ne[0] != 192 && ...;
```

Neither gate mentions compute capability. That is why both cards regress.

## Why it still matters

Every llama.cpp quickstart, for these models included, tells you to pass `-fa 1`
or `-fa on`. On this model that advice costs you half to four-fifths of your
decode speed, on any NVIDIA card, and nothing in the output says so. Anyone
benchmarking DeepSeek-V2-Lite with the standard recommended flags is measuring
a heavily degraded path.

With FA off on the V100 this is the **fastest decode of the three models
measured on that card** — 164.6 t/s, ahead of gpt-oss-20b's 143.2.

## Practical

```
# DeepSeek-V2-Lite family, any NVIDIA GPU:
llama-cli -m DeepSeek-Coder-V2-Lite-...-Q4_K_M.gguf -ngl 99 -fa off ...
```

For gpt-oss-20b and Ternary-Bonsai on the V100, keep `-fa on` — measured on the
same card in the same session, FA helps both:

| model | attention | fa off tg128 | fa on tg128 | verdict |
| :--- | :--- | ---: | ---: | :--- |
| DeepSeek-Coder-V2-Lite | MLA, no GQA | **164.6** | 33.6 | FA off |
| gpt-oss-20b MXFP4 | standard MoE | 126.2 | **143.2** | FA on |
| Ternary-Bonsai-2-27B | hybrid linear | 50.4 | **51.2** | FA on |

So this is a per-model decision, not a per-GPU one.

## Correction

The first commit in this repo claimed the regression was Volta-specific and
consistent with FlashMLA's documented Ampere-or-newer floor. That was wrong on
two counts, both caught by running the control rather than reasoning from the
documentation:

1. **The RTX 4070 regresses too** (-52% decode). The effect is not tied to
   compute capability, and the selector code confirms it — the gates that
   reject this shape test `gqa_ratio` and head dims, never `cc`.
2. **The 4070 can serve as a control for this model.** The earlier file said it
   could not. That was carried over from gpt-oss-20b (11.27 GiB), which genuinely
   does not fit; DeepSeek-V2-Lite at 9.65 GiB loads fine.

The documented FlashMLA Ampere+ floor is a real thing, but it is not what is
happening here.

## Status

Unreported. Tracker searched (`ggml-org/llama.cpp`) for prior reports of flash
attention regressing on Volta, on MLA, on `deepseek2`, and for head-size-192
fallbacks; nothing matching found. Closest hits, neither covering this:

- **#28887** — sparse FlashAttention on Volta, but for DeepSeek-V4 CSA layers;
  closed over AI-generated-content policy.
- **#26289** — open, tunes FA fp16 tile configs for head sizes 40-112 on
  P100/V100/Blackwell. Head size 192 is outside its range.

If filed, it should be framed as a shape/dispatch issue affecting all CUDA
GPUs, with the V100 and 4070 numbers as two datapoints — not as a Volta report.

## Environment

Host and toolchain as in the `volta-bonsai` repo's `ENVIRONMENT.md`. Each card
isolated via `CUDA_VISIBLE_DEVICES` set to its UUID.
