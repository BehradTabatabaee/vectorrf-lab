# Architecture

## Sources and captures

Every instrument observes **captures**: bounded windows of complex samples from the active source.

- **Transmitter.** `source_record(config, capture_id)` (`source.py`) generates capture *k* as an immutable array starting at absolute sample *k·N*, where *N* = `round(duration × sample_rate)`. Noise is keyed by the configuration seed and absolute sample index; payloads, symbols and pulse-shaping history are indexed by absolute bit/symbol position. Carrier offset, analog message, multitone, burst gate, hop/sweep schedules, OFDM grid position and custom programs also use absolute sample positions. Changing the capture length does not change the underlying stream.
- **Recording.** `FileSource` (`recordings.py`) describes a recording in the capture library plus playback settings. Capture *k* is the window of `capture_samples` samples starting at `start_sample + k·step` (wrapping when `loop` is set; `step` defaults to the window, i.e. contiguous playback).

`acquire(tx, file, capture_id)` returns either kind as a `GeneratedSignal` whose metadata carries `start_sample`, `capture_samples`, `source_revision`, `source_kind` and the power calibration. A byte-bounded LRU cache (`VECTORRF_CACHE_MB`, default 512 MiB) shares transmitter captures between observers; recordings are memory-mapped, and only the requested window is decoded.

A capture is identified by `(source, capture_id)`. Every observer — live spectrum, receiver/pipeline products, zero span, modulation quality, classifier — that names the same pair sees identical samples.

## Signal design

`TxConfig` (`models.py`) is one transmitter design. Three parts of it compose:

- **`components`** (with `waveform: "composite"`) is a list of up to 16 complete `TxConfig`s. A `mode="before"` validator copies the parent's sample rate, duration and RF centre into every component, so components cannot disagree about the clock; nesting is rejected. `signal._composite` generates each enabled component with a derived seed (`component_seed`), scales it to its own absolute power relative to the strongest component, and sums. The expected total on-power `P = Σᵢ Σⱼ aᵢ aⱼ Re ρᵢⱼ` (`signal.composite_power`) uses the correlation coefficients of the clean components over the first 65,536 samples of the stream: independent components add in power, while identical deterministic copies (a duplicated tone, one text payload twice at one frequency) add as amplitudes, as the output does. The gain `amplitude / √P` is fixed by the design, so bursts and hops never change the scale between captures, and the design plan reports the same total the output measures.
- **`agility`** (`agility.FrequencyAgility`) hops or sweeps a signal. `agility.schedule()` turns the design into an integer schedule (cached): one segment per hop or sweep with on/off sample counts, 64-bit tuning words, the per-segment phase and the raised-cosine ramp. `oscillator(cfg, fs, n, start_sample)` evaluates the local oscillator and the on/off gate for any absolute sample window in 1 Mi-sample chunks, using wrapping `uint64` phase accumulation (a sweep adds the triangular-number term for the linear frequency ramp). Agility multiplies the normalized signal before AWGN, for a single waveform, a whole composite and each component alike.
- **`rf_center_hz`** labels baseband 0 Hz. It changes no sample; it sets axes, absolute-frequency entry and the SigMF capture frequency.

`models.carrier_extent()` computes the frequency range a design reaches (offset, chirp sweep, multitone tones, hop/sweep range, components recursively); validation rejects anything beyond ±Fs/2. `bands.py` derives each component's nominal occupied band from textbook bandwidths (RRC `Rs(1+β)`, Carson's rule for FSK, OFDM layout, radar pulse bandwidth) for the band plan and warnings of `/api/tx/plan`. Example designs live in `examples/designs/*.json`; the test suite validates and generates every one.

## Live engine

`/ws/live` (`engine.py`) schedules frames independently of edits, validates updates, acknowledges accepted configurations and sends JSON metrics followed by a binary `VRF1` PSD frame. Metrics name the capture that produced them (`capture_id`, `capture_start_sample`, `source_kind`, `power_unit`) and the analyzer state (`center_hz`, `span_hz`, `decimation`, `analysis_rate`, `channels`).

Capture advance depends on `pacing` (transmitter source; recordings follow their `FileSource` playback):

- **`realtime`** (the UI default): the stream position is `start + elapsed wall time × Fs`, and each frame uses the capture at that position. Receiver, processing-chain and measurement products do not process the same capture twice; missed captures are skipped. The browser bounds a transmitter live capture by both 524,288 samples and about one display-frame duration; a view that demodulates symbols (receiver pages, modulation quality) gets captures of at least its filters' transient plus 32 symbols (`source.symbolFloor`, 2.6 M samples at 2 GS/s and 50 kBd), so it keeps working, at a lower update rate, at extreme oversampling. The engine measures expensive-product work and limits their processed prefix to roughly 75% of the frame budget, never below that symbol floor (`engine._symbol_floor`); zero span is not limited by that budget: it generates up to the end of its interval plus its filter's settling time and measures the interval whole. A frame may use a nearby prepared capture while the next is prefetched. With only the spectrum displayed, a frame keeps stepping through the current capture while the next one is still being generated. A low-rate recording may use overlapping capture windows to keep the display cadence.
- **`contiguous`** (API default):
  - with a **receiver or processing chain** active, every frame consumes the next complete capture, so state carried by the chain (FIR/lowpass memory, Costas phase/frequency, extension state) spans physically contiguous samples;
  - with a **measurement** (zero span or modulation quality), frames sweep each capture in consecutive segments — zero span of its sweep time, quality of what it processes per frame (measured, never fewer samples than its symbol floor) — and then advance, so every sample is measured;
  - with only the **spectrum**, frames step through the current capture in chunks sized by `spectrum.required_samples` (enough input for the requested RBW after zoom decimation) and then advance to the next capture.

The chain's memory is private to the socket and resets when source, channel or processing settings change.

`_CaptureStream` prefetches the next capture in a worker thread while the current frame is processed. The optional `acknowledge_frames` flag sends a `frame_end` marker after all products; the browser replies with `action: ack`. At most two real-time frames can be in flight, and stateful products execute in capture order. Frames follow an ideal 1/fps grid, because re-arming from "now" adds the coarse timer granularity (15.6 ms on Windows) to every period. Receiver and pipeline products are computed and serialized to JSON in a worker thread. Client revisions on metrics and products keep replies to old edits off the current plots. Receiver frames round their plot traces to six significant figures of each series' peak (`engine.LIVE_DIGITS`, `dsp.display_*`), about halving the payload; HTTP responses keep full precision.

The receiver product applies the channel experiment, then removes the known hop schedules in `RxConfig.dehop` (`pipeline.dehop`, the conjugate of the same oscillator), then runs the processing chain and the demodulator. While the chain keeps the sample clock, the demodulator keeps tracking from capture to capture (`memory['rx_track']`, see below); a reference-aided channel estimate against the transmitted symbols is added for PSK/QAM transmitter sources.

The measurement product (zero span and modulation quality) keeps its own tracking receiver (`memory['quality']`) and the most recent 2,048 tracked symbols of consecutive captures (`engine.QUALITY_SYMBOLS`), so the constellation and EVM describe a stable record even when real time leaves a few symbols per capture. It sends only what the page draws: about 60 kB per frame (it was 450–600 kB) with traces rounded to `LIVE_DIGITS`; `/api/measurements` keeps full precision.

HTTP observers (`/api/rx/loopback`, `/api/measurements`, `/api/blind/analyze`, `/api/studio/pipeline`, `/api/spectrum`) take `tx`, an optional `source` (`FileSource`) and the `capture_id` reported by the stream. Their cadence is independent of the spectrum display.

## Receiver

`rx.demodulate(samples, cfg, start_sample, track)` dispatches by format. The linear receiver (`rx._digital`) works per block:

1. **Decimation** to at most 16 samples per symbol (`WORK_SPS`, polyphase Kaiser β = 10), first, so every estimate below sees thousands of symbols however high the oversampling.
2. **Coarse carrier** (cold start, with `coarse_cfo`): the power spectrum's centroid over the occupied band (`sync.spectral_centre`, unambiguous but biased by a few percent of Rs for payloads with an asymmetric spectrum) arbitrates an M-th power line at the working rate (PSK/QAM) and a fine estimate at the symbol instants (M-th power, or consecutive-symbol products for differential formats). The fine estimate's Rs/M ambiguity is resolved by the coarse estimate; for APSK, whose outer ring is symmetric under one branch per symbol, by the fourth-power coherence of the inner ring.
3. **Matched filter** and symbol-level AGC (loop gains refer to unit-RMS symbols).
4. **Phase and scale**: QAM uses a blind phase search jointly with gains that match the symbol magnitudes to the constellation's radii (`modem.radius_scales`), because a repeating payload can use a dozen of 64-QAM's points with an RMS 16 % off; PSK and APSK use M-th power estimates.
5. **Timing and carrier loops**: Mueller & Müller jointly with a Costas or decision-directed carrier loop (Gardner for differential formats and OQPSK), started at the sampling phase with the most matched-filter energy.
6. **Differential detection** (with the residual rotation removed), optional CMA equalization, slicing and EVM with a decision-directed least-squares gain.

With `track` and the block's absolute `start_sample`, the next block continues instead of re-acquiring: the symbol clock stays on its absolute grid, the CFO correction anchored to absolute time, the carrier loop starts from its predicted phase refined on the new block (the blind phase search for QAM, the inner ring for APSK), and a converged CMA equalizer keeps its taps. A tracked block whose EVM jumps far above its predecessor's is acquired again from scratch.

The FSK/MSK/GMSK demodulator averages a frequency discriminator per symbol, with boundaries on the transmitter's absolute symbol grid (manual timing) or from a non-data-aided search for the boundary that best separates the tones, and removes the tones' common offset (`coarse_cfo`). The OFDM receiver estimates the CFO from the cyclic prefix, places the FFT window inside the ISI-free prefix and equalizes with pilots; without pilots it tracks a common phase per symbol from the QPSK data.

## Spectrum analysis

`spectrum.analyze(x, fs, cfg)` measures the reference level from the original samples, then plans the analysis with `zoom_plan`: when a span of at most a quarter of the sample rate is requested, the centre is mixed to DC and decimated through stages of 8, 4 and 2, each a cached Kaiser low-pass with ≥ 110 dB rejection whose transition band stays outside the span (`ZOOM_OVERSAMPLING`). `down_convert` keeps only fully valid filter outputs. Welch averaging then runs at the analysis rate with each windowed segment zero-padded to four times its length (`FFT_OVERSAMPLING`, fewer for windows beyond 65,536 points): bins lie at a quarter of the window's resolution, so a tone shows the same RBW shape and level wherever it falls between bins (Hann scalloping drops from 1.42 dB to under 0.1 dB). Frequencies are shifted back by the centre, and the result is masked to the requested span. The default window is a 4-term Blackman-Harris (−92 dB sidelobes), close to an analyzer's Gaussian RBW filter. Channel bands (`SpectrumConfig.channels`) are measured on the full-resolution spectrum: calibrated power is the sum of bin powers (edge bins weighted by overlap) divided by the window's equivalent noise bandwidth in (zero-padded) bins; the 99 % occupied bandwidth uses interpolated cumulative power.

## Classifier

`blind.analyze` optionally isolates a band (`isolate_band`: mix to DC, decimate while keeping at least eight times the band width, then a band-edge low-pass), measures the in-band SNR on the original capture, and runs `_analyze`. The classification is a sequence of stages (noise, OFDM, pulsed radar, chirp/FMCW, tone, line spectrum, constant envelope, linear modulation). Each stage records its checks — feature key, measured value, operator, threshold, outcome — in a `_Trace`, and the first stage whose checks all pass decides. `FEATURES` documents every feature key; `/api/blind/features` serves it.

## Recordings

| Module | Responsibility |
| --- | --- |
| `iqfile.py` | Sample encodings (SigMF datatype names), raw/WAV/RF64/SigMF/SigMF-archive readers, memory-mapped `RecordingReader`, writers (WAV/RF64 headers, SigMF metadata and archives), format sniffing, file-name hints |
| `recordings.py` | Capture library folder, name validation, streamed uploads, delete, `FileSource` capture windows, cached readers, whole-file overview |

Requests select recordings by plain file name inside the library folder (`VECTORRF_CAPTURES_DIR`; `<repository>/captures` for source checkouts, otherwise `~/.vectorrf-lab/captures`), never by arbitrary paths — the same pattern as the extension catalog. Uploads are streamed straight to a `.partial` file and renamed on completion (`VECTORRF_MAX_UPLOAD_GB`, default 256). On Windows a memory-mapped file cannot be replaced or deleted, so readers are released before either.

All integer formats decode to a common full scale (a complex sample of magnitude 1 is 0 dBFS). Float data read with the wrong format can contain NaN/Inf or absurd magnitudes; those components are zeroed and counted (`nonfinite_samples_replaced`) so they never reach the DSP or JSON products.

Export (`/api/files/export`) renders a transmitter capture — or a longer contiguous record, validated against `MAX_EXPORT_SAMPLES` through a Pydantic validation context — or converts a recording window, then encodes it to SigMF, WAV/RF64 or raw. SigMF exports of a transmitter capture embed the complete design (`vectorrf:tx_design`) and the RF centre as `core:frequency`.

## Wideband operation

Sample rate is a scale: limits are expressed in samples (`MAX_CAPTURE_SAMPLES`, radar samples × channels, range–Doppler cells), not seconds, so a 5 GS/s capture is simply shorter than a 5 MS/s one. To keep results identical across scales:

- carrier phase uses exact rational start phase (`dsp.phase_cycles` / `dsp.phasor`) and hop/sweep phase exact 64-bit integer accumulation instead of float `f·(t + start_s)`;
- frequency axes and bucket centres stay float64;
- estimators are written in samples/symbols, with no absolute-Hz constants;
- FFTs use batched, threaded `scipy.fft` on complex64.

`tests/test_wideband.py` checks that a 2–5 GS/s experiment reproduces the same experiment at 1 MS/s with all frequencies scaled.

## Python modules

| Module | Responsibility |
| --- | --- |
| `models.py` | Validated configuration models (`TxConfig`, `RxConfig`, `SpectrumConfig`, `ChannelConfig`, …), carrier-extent validation |
| `agility.py` | Hop/sweep schedules and the 64-bit DDS oscillator |
| `bands.py` | Nominal occupied bands, design plans and warnings |
| `signal.py` | Waveform generation, composites, power calibration, agility and impairments |
| `spectrum.py`, `measurements.py` | Zoom FFT/Welch analysis, channel power; zero span |
| `rx.py`, `sync.py`, `modem.py` | Demodulation, timing/carrier/equalizer loops, constellations and slicing |
| `blind.py` | Heuristic single-emitter classification with band isolation and a decision trace |
| `expressions.py` | Restricted AST array expressions with finite/size bounds |
| `programs.py` | Mathematical, control-point, I/Q table and local Python waveforms |
| `ofdm.py` | Resource-grid construction, CP/WOLA synthesis, capture offsets, CP CFO and pilot-LS reception |
| `pipeline.py` | Ordered processing blocks, scopes, stage measurements, live state, channel and de-hopping |
| `source.py`, `engine.py` | Capture identity, bounded reuse, prefetch and scheduled transport |
| `radar_sim.py`, `radar_processing.py` | Full-waveform point-target propagation, array channels, beamforming, correlation, pulse-Doppler/FMCW processing, CFAR detection, angle estimation |
| `radar_studio.py` | Pulse, segment and continuous-wave designer feeding the simulator; design figures |
| `iqfile.py`, `recordings.py` | Recording formats and the capture library |
| `app.py` | FastAPI routes, request models and static assets |

## Browser application

The UI is plain ES modules served from `backend/vectorrf/web/js/` without a build step. `index.html` loads `js/main.js` as a module; everything else is imported explicitly, so a wrong path fails at load time (`tests/test_packaging.py` resolves every import without a browser).

| Folder | Responsibility |
| --- | --- |
| `core/` | `state.js` (the single state object, initialized from `/api/defaults`), `controls.js` (field markup and delegated binding), `format.js` (engineering-notation parse/format), `api.js`, `live.js` (the one live WebSocket), `events.js` (publish/subscribe), `source.js` (active source and request bodies), `ui.js` (status, toasts, dialogs), `persist.js` (the last applied design and recording settings, per browser), `dom.js` |
| `plots/` | Canvas plots (`canvas.js`), the spectrum analyzer with markers, selections and channel windows (`analyzer.js`), band and time plans (`plan.js`), the radar scene (`scene.js`) |
| `editors/` | Reusable editors: waveform, hop/sweep, frame blocks, OFDM grid and shaping, processing chain, structured lists, export |
| `workspaces/` | One module per tab — transmitter, recordings, spectrum, receiver, classifier, radar — each exporting `render`, `start`, `stop`, `resize`, `usesLive` and `payload` (its part of the live-stream request) |

Controls are declared in markup: `data-bind="path"` binds a field to a state path, `data-action`, `data-change-action` and `data-set-path` name handlers, and `bindControls(root, {onChange, actions})` attaches one set of delegated listeners per container. A change that leaves the state unchanged does not re-render. `replaceHTML` re-renders a container while preserving scroll positions, focus and folded sections. Workspaces never call each other; they communicate through the shared state and the event bus (`live:*`, `tx:changed`, `library:changed`, `navigate`). `main.js` switches workspaces, owns the live payload and exposes `window.vectorrf = {state, switchMode}` for debugging and the browser test.

The server registers `text/javascript` for `.js` explicitly (some Windows registries map it to `text/plain`, which browsers refuse for modules) and sends `Cache-Control: no-cache` for the UI shell and modules, so an upgrade never mixes modules from two releases. The waveform/processing Python extension catalog is loaded from the local project `extensions` folder at startup; network requests select installed names, not file paths or uploaded Python source.

## Radar experiment

A request to `/api/studio/radar` carries the source design (program, legacy sequence, OFDM or a recording window), sample rate/size, capture ID/time, target kinematics, array geometry/weights/pattern, link-budget controls and processing blocks. The legacy sequence is re-validated at the experiment's sample rate, so the simulation has a single clock. The source is sampled at current and delayed times. Each point target contributes an echo with delay, radial Doppler and spatial phase at each receive element. Beamforming precedes the editable processing chain. Delay/Doppler correlation uses the transmitted waveform reference. For legacy designs the response includes `design` (`radar_studio.design_summary`): PRF, duty, bandwidth, compression, range and velocity resolution and ambiguities.

A periodic transmit record (a Studio design, an OFDM frame or a recording window) is read at a delayed time with a windowed-sinc fractional delay (`radar_sim.periodic_slice`): band-limited, about −65 dB error up to 0.8 Nyquist, where linear interpolation was off by several dB near Nyquist.

Detection runs along range in every Doppler row of the range–Doppler map (`radar_processing.cfar_threshold`): cell-averaging, greatest-of, smallest-of or ordered-statistic CFAR, each with its threshold multiplier solved exactly for the requested per-cell false-alarm probability in exponential (square-law) noise, or a fixed threshold. The multiplier counts independent training cells (bins per resolution cell from the waveform's autocorrelation, `radar_sim.range_response`), and a second threshold with a guard over the waveform's sidelobes keeps a strong target from raising its own. Detections are mainlobe-sized peaks with a crossing in their mainlobe; weaker peaks within a stronger one's range or Doppler sidelobe envelope are its sidelobes (`radar_processing.detect`). The range-profile threshold drawn in the UI is the one of the Doppler row holding each range cell's maximum, so a profile crossing is a detected cell.

Range/Doppler strongest-cell samples across the array drive the angle map. The result distinguishes measured coordinates from input target truth. The front end advances simulation time explicitly and labels all scopes with their processed capture. It permits only one radar request in flight and rejects results from an older design revision.

## Bounds

Sample rate: 5 GS/s. Capture: 8,388,608 samples (`VECTORRF_MAX_CAPTURE_SAMPLES`); export: 33,554,432 samples (`VECTORRF_MAX_EXPORT_SAMPLES`). Composites: 16 components; hop plans: 262,144 hops (16,384 listed frequencies); receiver de-hop schedules: 2. Source cache: 12 captures and 512 MiB. Spectrum FFT: 1,048,576 points; channel windows: 8; live spectrum frames carry at most 2,048 display points. Radar: 262,144 samples, 4,194,304 channel-samples, 32 elements, 12 targets, 2²³ delay–Doppler cells. Processing chains: 24 blocks. OFDM: FFT up to 4,096, 128 frame symbols and 8,192 overrides. Recording overview: up to 1,024 segments. Waterfall and classifier history remain bounded.

## Security

The server has no authentication and binds to `127.0.0.1` by default. HTTP API requests are JSON bodies (cross-origin pages cannot send them without a CORS preflight, which the server does not grant), and `/ws/live` closes connections whose `Origin` names another host, since browsers do not apply CORS to WebSockets. User expressions are evaluated by a restricted AST interpreter (`expressions.py`: arithmetic, named variables and a fixed list of NumPy functions, no attribute access), and recordings are selected by plain file names inside the library folder. Only local Python files in `extensions/` execute code.

## Transport and performance design

Large HTTP plot products return a JSON response directly instead of repeating recursive framework conversion. Full DSP arrays remain separate from bounded display previews. OFDM grid placement and bit extraction use vectorized arrays. Radar caches only reference-independent Doppler mixers and array steering geometry; echoes and measurement products are recalculated per capture. Filter, waveform or scene edits still change the actual samples.

`channel_samples` applies the receiver channel experiment (or bypasses it when disabled, the default for recordings) per capture. Live processing state stays per socket; convenience HTTP demodulation is capture-local.
