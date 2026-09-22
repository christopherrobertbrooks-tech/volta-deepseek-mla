# DeepSeek MLA on a Tesla V100 (sm_70): works, but turn flash attention OFF

MLA (Multi-head Latent Attention) runs correctly on Volta. But **flash
attention, which llama.cpp guides routinely tell you to enable, makes it 4.9x
slower at decode** — silently, with no warning and no error.

Model: `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` (9.65 GiB loaded, 15.71 B
params, `deepseek2` arch), upstream llama.cpp `c550d2f60` (b11096), CUDA 12.9.86,
Tesla V100-PCIE-32GB.

## The finding

`llama-bench -ngl 99 -fa 0,1 -p 512 -n 128 -r 5`

| flash attention | pp512 | tg128 |
| :--- | ---: | ---: |
| **off (`-fa 0`)** | **1836.3 ± 16.5** | **164.6 ± 0.4** |
| on (`-fa 1`) | 1000.9 ± 5.5 | 33.6 ± 0.04 |

Enabling flash attention costs **80% of decode throughput** and 45% of prefill.

Confirmed on the interactive path, not just the benchmark: the same prompt at
`temp 0` produced **byte-identical output** either way, at 155.2 t/s with FA off
against 32.5 t/s with it on.

So this is purely a performance trap. Output is unaffected — which is exactly
what makes it easy to miss.

## It is specific to MLA, not to Volta

Tested the same `-fa 0,1` sweep on two other models on the same card, same
session:

| model | attention | fa off tg128 | fa on tg128 | FA verdict |
| :--- | :--- | ---: | ---: | :--- |
| DeepSeek-Coder-V2-Lite | MLA | **164.6** | 33.6 | **catastrophic, -80%** |
| gpt-oss-20b MXFP4 | standard MoE | 126.2 | **143.2** | helps, +13% |
| Ternary-Bonsai-2-27B | hybrid linear | 50.4 | **51.2** | helps slightly, +1.5% |

Flash attention is the right default on Volta for the other two. Only the MLA
path collapses. That matches the documentation — FlashMLA on CUDA is specified
as Ampere-or-newer, and DeepSeek flash attention as Turing-or-newer (7.5), with
Volta at 7.0 excluded by one step — but the exclusion manifests as a silent 5x
slowdown rather than a refusal or a fallback.

## Why this matters more than the numbers

Every llama.cpp quickstart, including the ones for these models, says to pass
`-fa 1`. On a V100 running a DeepSeek model that advice costs you five times
your decode speed, and nothing in the output tells you. Anyone benchmarking
Volta with the standard flags would conclude MLA is slow on this hardware. It
is not: at 164.6 t/s with FA off it is the **fastest decode of the three models
measured here**, ahead of gpt-oss's 143.2.

## Practical

```
# On a V100, for DeepSeek/MLA models:
llama-cli -m DeepSeek-...-Q4_K_M.gguf -ngl 99 -fa off ...
```

For gpt-oss and Bonsai on the same card, keep `-fa on`.

## Status

Unreported. Worth reporting — the performance cliff is undocumented as far as
the searches went, and "enable flash attention" is near-universal advice.

Before filing: read the whole issue tracker first and check whether it is known
or already addressed. See the near-miss recorded in the `volta-gpt-oss` repo.

## Environment

Host and toolchain as in the `volta-bonsai` repo's `ENVIRONMENT.md`. Card
isolated via `CUDA_VISIBLE_DEVICES` set to the V100's UUID, so the RTX 4070 is
untouched.

Note the 4070 cannot be used as a control for this model at Q4_K_M plus
context alongside the desktop; the V100's 32 GB is what makes these tests
possible on this machine.
