# Benchmark — 0.7

Same environment, same 60,000-sample OFDM settings and 8,192-sample radar scene, one warm-up and median of nine runs. Python 3.12/Linux x86-64; BLAS thread count 1. The v0.6 baseline was measured from the previously delivered extracted archive; these are local timings, not a hardware guarantee.

| Operation | v0.6 ms | v0.7 ms | Speedup |
| --- | ---: | ---: | ---: |
| OFDM generation, 60,000 samples | 101.130 | 13.119 | 7.7× |
| OFDM reception, 60,000 samples | 281.714 | 51.187 | 5.5× |
| Waveform/array radar, 8,192 samples | 108.862 | 34.194 | 3.2× |
| Reduce 32,768 spectrum bins | 8.235 | 0.812 | 10.1× |

These measurements cover DSP, excluding HTTP/JSON/browser rendering. Browser results are recorded in BUILD_REPORT.md. The new default communications capture is 20 ms, configurable up to 200 ms in the UI; the table deliberately retains the longer old capture for comparison. Complex arrays, large captures, Python extensions and adaptive processing can lower the achieved display rate.

The historical v0.7 engine targeted 20 spectrum updates/s and 12 radar updates/s. Current releases target 30 Hz for communications and radar live displays, with frame acknowledgment or a single request in flight. Faster scheduling alone is not proof of faster measurements; numerical data and actual capture IDs are tested separately.

## 1.2 spectrum / zero-span release sanity

Release-validation environment: Linux x86-64, Python 3.13, BLAS thread count 1. QPSK capture duration 20 ms, requested spectrum span 10 MHz, RBW 2 kHz, two FFT averages, `max_fft=32768`. Figures are local medians and are not hardware guarantees.

| Source rate | Source regeneration* | Spectrum frame median | Spectrum frame p95 | Live display points | Zero-span median | Zero-span points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 MS/s | 2.92 ms | 0.205 ms | 0.396 ms | 1,024 | 0.247 ms | 900 |
| 10 MS/s | 21.32 ms | 0.684 ms | 1.046 ms | 2,048 | 0.729 ms | 900 |

\*Source regeneration occurs when transmitter configuration changes; steady live spectrum frames reuse the source record.

The 1.2 browser path additionally renders only the traces the user enabled. Trace 1 is live by default and the three hold/average traces start disabled. The communications live target remains 30 Hz; zero span targets 12 Hz and modulation quality 4 Hz. Achieved rates depend on browser, CPU, FFT settings and the rest of the active processing chain.
