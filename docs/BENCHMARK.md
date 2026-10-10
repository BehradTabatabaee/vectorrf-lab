# Benchmarks

Local timings on one desktop, to show where the time goes and what the live views achieve; they are not hardware guarantees. Environment: Windows 10, Python 3.10.6, NumPy 2.2.6, SciPy 1.15.3, a 16-thread AMD desktop CPU, `OPENBLAS_NUM_THREADS=1` (FFTs use `scipy.fft` worker threads).

## DSP workloads

`python examples/benchmark.py`: median of 3 runs after one warm-up, milliseconds, excluding HTTP, JSON and the browser.

| Source | Samples | Generate | Spectrum frame | Receiver (QPSK) | Classifier |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 MS/s, 50 kBd | 60,000 | 4.4 | 12.8 | 19.1 | 178 |
| 5 MS/s, 50 kBd | 300,000 | 8.3 | 27.4 | 25.1 | 136 |
| 2 GS/s, 200 MBd | 2,000,000 | 51 | 28.4 | 35.6 | 284 |

The spectrum frames use a 1/20,000-of-the-rate RBW with a Blackman-Harris window, zero-padded four times. The receiver decimates to 16 samples per symbol before anything else and analyses at most 8,192 symbols, so its cost follows the number of symbols rather than the sample rate.

Spectrum frames at 2 GS/s (two averaged segments; 16-QAM at 200 MBd):

| Window | RBW | Frame |
| ---: | ---: | ---: |
| 32,768 | 91.6 kHz | 15 ms |
| 131,072 | 22.9 kHz | 25 ms |
| 524,288 | 5.7 kHz | 53 ms |
| 1,048,576 | 2.9 kHz | 146 ms |

Other workloads:

| Workload | Samples | Time | Throughput |
| --- | ---: | ---: | ---: |
| Recording window, cf32 / ci16 / cu8 (memory-mapped) | 2,097,152 | 22 / 19 / 24 ms | 95 / 111 / 89 MS/s |
| Hop oscillator (1001-hop grid, off-time and ramps, 125 MS/s) | 8,388,608 | 491 ms | 17.1 MS/s |
| Sweep oscillator (sawtooth, 125 MS/s) | 8,388,608 | 876 ms | 9.6 MS/s |
| Composite: tone + 7.5 MBd QPSK at 60 MS/s, 20 ms | 1,200,000 | 89 ms | 13 MS/s |
| Zoom-FFT frame, 1 MHz span at 60 MS/s, 200 Hz RBW | 1,200,000 | 48 ms | — |
| Array radar experiment (default scene: 8 elements, correlation) | 4,096 | 15 ms | — |

The hop and sweep oscillators are evaluated in closed form for any absolute sample, so their cost does not grow with simulated time.

## Live update rates

Updates per second in a browser (headless Microsoft Edge at 1440 × 900, 1.5× device scale) with the UI defaults — a 50 kBd QPSK carrying PN15 at 40 dB SNR, the receiver channel at 25 dB SNR with a 300 Hz offset, real-time pacing — over 5 s after the view settled, with the payload per update and the browser main thread's busy time:

| Sample rate | Demodulation | Live monitor | Spectrum | Zero span | Modulation quality |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 MS/s | 29 | 30 | 30 | 30 | 30 |
| 10 MS/s | 29 | 30 | 30 | 30 | 30 |
| 100 MS/s | 30 | 30 | 30 | 30 | 30 |
| 2 GS/s | 5 | 30 | 30 | 8 | 14 |

Payloads: spectrum 1 kB (the frequency axis is sent once), zero span 55–58 kB, modulation quality 64–67 kB, live monitor 80 kB, demodulation 130–260 kB. The main thread stays below 25 % busy.

Zero span sweeps 2 ms: at 2 GS/s that is 4 M samples per sweep, generated and transformed whole (one 4 M-point FFT, about 50 ms); a shorter sweep is proportionally faster. With every-sample pacing zero span and modulation quality run at the same rates while sweeping the stream end to end (29–30 updates/s up to 100 MS/s; 8 and 16 at 2 GS/s). A tone and its copy 5 MHz away at 200 MS/s, with 20 ms captures (4 M samples): zero span and modulation quality 30 updates/s in both pacings.

Modulation quality measures a fixed update -- 512 new symbols (1,024 for OQPSK) or 262,144 samples -- in 7–15 ms: 16-QAM at 5 MBd on a 200 MS/s clock and 64-QAM at 1 MBd on 50 MS/s run at 30 updates/s in both pacings (64-QAM ran at 15, losing lock between short blocks and acquiring again). At 2 GS/s a 50 kBd signal has 40,000 samples per symbol: the matched filters' transient plus 32 symbols are 2.6 M samples, whose decimation sets the 14 updates/s.

At 2 GS/s a 50 kBd signal has 40,000 samples per symbol: a real-time capture of 524,288 samples holds 13 symbols, fewer than the matched filters' transient. Views that demodulate symbols therefore ask for captures of 2.6 M samples (the transient plus 32 symbols); generating and decimating them sets their update rate, and real-time pacing drops everything in between. Modulation quality keeps tracking across those captures and measures the most recent 2,048 symbols. A higher symbol rate brings every view back to 30 updates/s.

Nothing processes a multi-GS/s stream in real time: each update analyses one bounded capture, and real-time pacing drops the rest, like an SDR dropping buffers. Achieved rates depend on the CPU, browser, FFT settings, sample rate, capture size and processing chain.
