# Changelog

All notable changes to VectorRF Lab. Versions follow [semantic versioning](https://semver.org/).

## 2.0.2 — 2026-10-10

- A real-time capture prefix of a chirp without its own sweep period is the start of the capture again. The shorter prefix that real-time pacing asks for (for the receiver and modulation quality) used to sweep over the prefix instead of the capture, a different signal from the one the spectrum showed.
- Pulse shaping gives a slice of the stream exactly the samples of any longer slice. BLAS rounds a row of a matrix product differently depending on the shape of the product, so a prefix could differ from the start of its capture by a float32 ulp in a third of the samples; rows are now computed in blocks aligned to the absolute symbol index.
- CI reports failed tests as annotations on the run.

## 2.0.1 — 2026-10-10

- Text being typed into a field is kept when the editor re-renders to answer an earlier edit. When the design plan of a value committed a moment before arrived while the next value was being typed, the editor was rebuilt and the half-typed entry silently replaced, so that edit never happened.
- The live engine no longer logs a browser that disconnects while a product is being sent as a failed product, and a page that closes before its session starts no longer raises an error on the server.
- CI stability: the streaming test follows the client revision each product answers (the server keeps up to two frames in flight), and the browser test waits for conditions instead of fixed delays, so slower runners no longer fail intermittently.

## 2.0.0 — 2026-10-09

A major release: I/Q recordings and operation up to 5 GS/s, composite and frequency-agile transmitters, an instrument-style spectrum analyzer, a receiver that acquires and tracks across the whole carrier-offset range, a classifier that explains itself and a radar studio with an interactive scene and a family of CFAR detectors — on a rewritten, module-based interface, after a full mathematical audit of the DSP. Breaking changes are listed at the end.

### Transmitter

- **Composite signals**: up to 16 independently designed signals summed on one sample clock, each with its own absolute power, offset, phase, payload, burst gating and hopping. The total power accounts for coherent components (a duplicated tone reads +6.02 dB, not +3 dB) and matches what the output measures; a duplicated component is placed where it overlaps nothing.
- **Frequency hopping and sweeps on any signal**, a composite and each component: linear, listed or seeded pseudo-random hop plans, up/down/up-down sweeps, dwell, off-time, raised-cosine ramps and a start delay, from a 64-bit DDS with exact phase at any absolute sample. Phase is continuous (VCO-like) or coherent (each hop on its own free-running oscillator).
- **The transmitter is one stream** indexed by absolute sample: any capture of any length is an exact slice, so captures join seamlessly. Payloads, pulse shaping (an exact polyphase form whose cost does not grow with oversampling), differential encoding, CPFSK/GMSK phase, pulse trains, bursts, hops and noise (counter-based Philox streams) are all functions of the absolute index.
- `power_dbm` is the mean power while the signal is on, a property of the design rather than of one capture.
- The sawtooth chirp is phase-continuous through its flyback, like the FMCW waveform and agility sweeps.
- **RF centre** (`rf_center_hz`): frequency fields accept and show absolute RF values; SigMF exports record it.
- Every frequency a design reaches (offsets, chirps, tones, FSK/FM/PM deviation, hops, components) is validated against ±Fs/2 with a message naming the value, the band and the sample rate that would hold it. The browser keeps the last design the server accepted on air while an edit is invalid, and marks the components at fault.
- The default signal is a 50 kBd QPSK carrying a full-period PN15 sequence: a textbook root-raised-cosine spectrum.
- Example designs (tone + 10 MHz QPSK, FHSS 900 MHz – 1 GHz, pseudo-random FHSS, swept and chirping tones, OFDM with interferers) are files in `examples/designs/`.

### Spectrum analyzer

- **Centre and span** (or start/stop) with a **zoom FFT**: spans up to Fs/4 are digitally down-converted through multistage Kaiser decimation (≥ 110 dB alias rejection per stage), so a narrow span reaches the RBW the capture allows.
- The FFT is zero-padded four times: a tone reads its level within 0.1 dB and shows the same shape wherever it falls between bins (with one bin per resolution, an on-bin tone hid its window sidelobes and an off-bin one lost up to 1.42 dB). The default window is a 4-term Blackman-Harris (−92 dB sidelobes).
- Mouse and keyboard operation: click for markers, drag to select a band (zoom, channel power, classify, receive), wheel zoom, right-drag pan, double-click full span; up to eight normal, delta and noise markers; up to eight draggable **channel-power** windows with density and 99 % occupied bandwidth; four traces, memories and a waterfall.
- **Zero span** tunes and band-limits before AM/PM/FM detection. **Modulation quality** tracks the carrier from capture to capture and measures a record of the 2,048 most recent symbols, so it stays live at any sample rate (at 2 GS/s and 50 kBd a real-time capture holds 13 symbols); its frames shrank from about 600 kB to 60 kB, and a signal without symbols shows its waveform under correctly named panes.

### Receiver

- **Carrier acquisition over ±10 % of the symbol rate** for every linear format: a spectral-centroid estimate arbitrates M-th power estimates at the working rate and at the symbol instants; the APSK branch is identified by the inner ring's fourth-power coherence and differential formats use consecutive-symbol products. 16/32-APSK and π/4-DQPSK used to lose lock beyond 2–3 kHz at 50 kBd.
- **Payloads that use few constellation points** (a repeating text uses 12 of 64-QAM's points) no longer fool the receiver: QAM phase comes from a blind phase search and scale from matching symbol magnitudes to the constellation's radii; 64-QAM read 11–15 % EVM on such captures, now 1 %.
- Carrier estimates run after decimating to 16 samples per symbol, so they see every symbol at any oversampling: at 40,000 samples per symbol they used to see three symbols, and EVM read 50 %.
- Cold timing loops start at the strongest sampling phase; the receiver **tracks across captures**, including captures real time dropped, and re-acquires a block that lost lock.
- Gardner timing for all differential formats; differential detection removes the residual rotation (DBPSK 4.6 → 1.7 % EVM).
- FSK/MSK/GMSK: symbol boundaries on the transmitter's absolute grid or from a non-data-aided search, and tone-offset removal: 4-FSK garbled every capture that began mid-symbol, GMSK failed at 2 kHz.
- OFDM without pilots tracks its common phase from the data.
- **Equalizer**: the reference-aided channel estimate (plotted with the CMA taps) never reached the live receiver; the CMA equalizer keeps its taps across captures and now converges (EVM 30 → 5 % with a 0.3 echo).
- Loop gains refer to unit-RMS symbols and timing gains to symbols, so loops behave the same at any level and oversampling; the carrier page shows damping, natural frequency, lock range and settling. Mueller & Müller timing runs jointly with the carrier loop on carrier-corrected decisions. The headline EVM excludes the loops' acquisition.
- **Synchronized de-hopping** of known schedules; the receiver follows the transmitter (or a chosen composite component) until edited.

### I/Q recordings

- A recording library with streamed uploads; readers for raw interleaved files (cf32/cf64/ci32/ci24/ci16/ci8/cu8, either endianness), WAV/RF64 (including SDR# centre frequencies) and SigMF pairs and archives, memory-mapped for multi-GB files; content heuristics for headerless files and format-contradiction warnings; rate and centre parsed from gqrx and SDR# file names.
- Any recording is the live source of Spectrum, Receiver, Classifier and Radar Studio, with contiguous, real-time or custom-step playback and looping, in dBFS with an optional dBm offset.
- Export of transmitter captures, longer records and recording excerpts to SigMF, WAV/RF64 or raw with peak or unity scaling and clipping reports; SigMF exports carry a power calibration and the design that produced them.

### Classifier

- A **decision path** lists every rule with the measured feature, its threshold and the outcome, with a feature glossary; constellation hypotheses report their fit and the constellation points the data used.
- Drag across an emitter to classify that band only (mix to DC, decimate, band-edge low-pass, in-band SNR against the capture's noise floor). Fractional symbol-period search, derotation per hypothesis, CPFSK and bursty emitters, multiple-emitter reports.

### Radar Studio

- **Detectors**: CA-, GO-, SO- and OS-CFAR along range in every Doppler row, each with its threshold multiplier solved exactly for the requested per-cell false-alarm probability (checked by Monte Carlo), or a fixed threshold. Cells above threshold are grouped and each group reported once with range, velocity, power and margin. CA-CFAR used to run on the Doppler-maximized range profile with an uncalibrated false-alarm rate. The default false-alarm probability is 10⁻⁶ per cell (was 10⁻³).
- Delayed echoes of stored waveforms are band-limited (windowed sinc) instead of linearly interpolated, which was off by several dB near Nyquist; correlation processing applies the Doppler window (−31 dB instead of −13 dB sidelobes).
- An interactive scene: drag targets and their velocity vectors, which keep the position (or the velocity) they did not change; design figures (PRF, duty, bandwidth, compression, range and velocity resolution and ambiguities); one sample clock for waveform and simulation.
- Physics fixes: leakage and clutter are relative to the transmitted signal (dBc) and off by default (the radar-equation scene used to peak at 0 m); Doppler is measured at full resolution; CFAR guard cells cover the waveform's mainlobe.

### Live engine and performance

- **Real-time pacing** (the UI default) follows the wall clock and drops what cannot be processed, like an SDR dropping buffers, so every view runs at about 30 updates/s at any sample rate; a status-bar readout shows the share of each capture processed. "Every sample" pacing processes consecutive captures without gaps.
- A view that demodulates symbols gets captures long enough to hold them at any oversampling.
- Faster DSP throughout: parallel noise, resampling and filtering, complex64 post-processing, an exact polyphase pulse shaper, cached schedules, frames on an ideal 1/fps grid (Windows' 15.6 ms timer capped displays at 21 Hz), a persistent receive task (the old polling made products several times slower) and prefetched captures.
- Plot payloads are rounded to six significant figures in live frames and are JSON-safe (undefined points become gaps).

### Interface

- The browser code is plain ES modules (`web/js/core`, `plots`, `editors`, `workspaces`) without a build step; defaults come from the server.
- Number fields accept engineering notation; in frequency, sample-rate and symbol-rate fields lowercase prefixes work as typed (`10mhz` is 10 MHz, `2.4ghz` 2.4 GHz; `10mhz` used to be read as 10 millihertz). An out-of-band entry is refused with a notice naming the sample rate that would hold it, and the arrow keys and steppers move a frequency visibly from 0 Hz (they moved it by 1 Hz).
- Readable validation messages, hover tooltips instead of instructional prose, a Family/Modulation waveform picker, a calmer instrument look, the logo in the top bar, and displays that resume after the page was hidden.

### Correctness fixes found by the audit

P1 polyphase code order (5 dB worse sidelobes); transmit-power calibration with noise and bursts; zero-padded short-capture spectra (a 0 dBm tone read −74.6 dBm); CPFSK discriminator alignment (25 % noise-free BER at 2 samples/symbol); frame-block symbol offset; EVM normalization bias; float precision of absolute-time phase (−50 dBc spurs at 2 GS/s) and of GHz frequency axes; the channel's fractional delay zeroed 20 samples per capture; OFDM edge-carrier channel estimates and FFT window placement (9 % EVM from a 0.5-sample delay, now 0.7 %); live capture identity (the stream re-processed one record while reporting new capture IDs).

### Security

- The live WebSocket accepts connections from its own origin only (browsers do not apply CORS to WebSockets).

### Breaking changes

- Removed the transmitter's design files and presets (`GET /api/presets`); example designs are files in `examples/designs/`.
- Removed `/api/radar/preview`, `/api/radar/studio` and `/api/radar/scene`; `/api/studio/radar` covers all radar sources, including Radar Studio designs.
- `SpectrumResult.nfft` is the window length; the zero-padded transform is `fft_size` and the frequency axis is spaced `analysis_rate / fft_size`. The default window is Blackman-Harris.
- `RadarProcessing.detector` accepts `ca-cfar`, `go-cfar`, `so-cfar`, `os-cfar` (with `os_rank`) and `manual`; detections carry `velocity_mps` and `margin_db`, one per cluster of cells above threshold.
- Loop gains: `CostasConfig.alpha/beta` refer to unit-RMS symbols (defaults 0.04 and 0.001) and timing gains are in symbols per unit detector output.
- `power_dbm` is the power while on; radar leakage and clutter are dBc and off by default.
- The live WebSocket's default pacing remains `contiguous`; the UI uses `realtime`.

## 1.2.0 — 2026-10-02

The first public release on GitHub.

- Requested spectrum spans control the displayed start/stop frequencies up to 10 MHz; regions outside the source's Nyquist interval are shown as unavailable instead of shrinking the axis. Oversized typed spans clamp in the UI.
- Live spectrum payloads are reduced to a peak-preserving 2,048 points and zero span to 900 points.
- An in-app Help page.
- An installable `vectorrf-lab` command and self-contained wheels with the web assets.
- A conventional `tests/` tree, packaging and static-asset tests, GitHub Actions CI, contribution templates and community documentation.

## 1.1.0

- A 10 MHz maximum spectrum span, baseband analyzer operation, absolute dBm transmitter calibration, spectrum markers and zoom, and generic mixed-modulation TX/RX framing.
- Fixed record-boundary spectrum discontinuities and modulation-quality handling for basic signals.
