# Throughput of the two pathways

**Unit.** One candidate parameter set = one conductance triple simulated across all 35 injected-current levels, 300 ms each, at a 0.01 ms integration step.

| Pathway | Hardware | Implementation | Candidate parameter sets per second | Range across 5 launches |
|---|---|---|---:|---:|
| Non-differentiable (NEURON) | one processor core | — | **1.1** | — |
| Non-differentiable (NEURON) | 72-core node | spike-count stage | **16.4** | — |
| Non-differentiable (NEURON) | 72-core node | time-warping stage | **16.9** | — |
| Differentiable | one processor core | unbatched (1 set at a time) | **0.089** | — |
| Differentiable | one processor core | batched (1,024 sets) | **20.8** | — |
| Differentiable | RTX 2070 (2018) | batched, unfused | **543** | 542.8 – 543.0 |
| Differentiable | RTX 2070 (2018) | batched, graph-captured | **572** | 571.7 – 571.7 |
| Differentiable | RTX 2070 (2018) | batched, fused | **34,279** | 34,275 – 34,288 |
| Differentiable | RTX 4070 Ti (2023) | batched, unfused | **2,053** | 2,046 – 2,057 |
| Differentiable | RTX 4070 Ti (2023) | batched, graph-captured | **2,806** | 2,790 – 2,809 |
| Differentiable | RTX 4070 Ti (2023) | batched, fused | **63,649** | 63,633 – 63,651 |
| Differentiable | RTX 4070 Ti (2023) | batched, fused, cache-resident batch | **100,443** | 98,001 – 104,926 |

## Memory traffic, before and after fusing the integration step

| Stage | Hardware | Traffic demanded (GB/s) | Device bandwidth, measured (GB/s) | Share of device bandwidth |
|---|---|---:|---:|---:|
| Before fusion | RTX 2070 | 412 – 420 | 393.6 | 105 % – 107 % |
| After fusion | RTX 2070 | 51 – 64 | 393.6 | 13 % – 16 % |
| After fusion | RTX 4070 Ti | 116 | 426.9 | 27 % |

Before fusion the loop demands slightly more bandwidth than the card can supply, so it is bound by memory traffic rather than by arithmetic or capacity: each integration step issues 76 separate kernels, every one of which reads its operands from device memory and writes its result back. Fusing the step so those intermediates are never committed to memory removes that bound and raises throughput on the RTX 2070 from 543 to 34,279 sets per second, a factor of 63.

## Caveats that travel with these figures

1. The non-differentiable rate includes the summary statistics computed for each parameter set; the differentiable rate is simulation only. The comparison therefore flatters the differentiable pathway.
2. Hardware is not normalised: 2018 and 2023 consumer graphics cards against 2020 server processor cores.
3. The biophysics differs slightly: the non-differentiable mechanism carries four state variables against three in the differentiable model.
4. Without batching the advantage reverses — one processor core running the differentiable model one parameter set at a time is 12.5 times slower than the same core running NEURON.
5. The seven graphics-card rates are medians of five independent launches of the same benchmark, with the full range across launches in the spread column; the five processor-core rates are single measurements. At the batch sizes quoted the range is below 0.6 % of the median except for the cache-resident row, where it is 6.9 %.
6. No figure here should be extrapolated to other hardware. Across the two card generations measured, fused throughput rose by a factor of 1.86 while the cards' rated single-precision arithmetic throughput differs by a factor of 5.37, so throughput does not track the rating.

*Sources for every value are in `throughput_table.csv`.*
