# API and programming guide

Everything the browser does goes through a small HTTP/WebSocket API on the same server, and every DSP block is an importable Python function. This guide covers both, plus the interfaces for custom waveforms, OFDM grids and local Python extensions.

All configuration objects are [Pydantic](https://docs.pydantic.dev/) models in `backend/vectorrf/models.py` (and `agility.py`, `ofdm.py`, `programs.py`, `pipeline.py`, `radar_*.py`). `GET /api/defaults` returns the default value of every editable model, so a client never needs to duplicate them. Invalid requests return HTTP 422 with one readable sentence per problem in `detail` and the raw field locations in `errors`.

## Python quick start

```python
from vectorrf.models import TxConfig, RxConfig, SpectrumConfig
from vectorrf.signal import generate
from vectorrf.engine import calibrated_analyze
from vectorrf.rx import demodulate

tx = TxConfig(waveform="qpsk", sample_rate=1e6, symbol_rate=50e3, duration=0.02,
              data_source="pn", pn_order=15, pn_bits=32767, snr_db=30)
capture = generate(tx)                       # GeneratedSignal: samples (complex64), metadata, tx_bits, tx_symbols

spectrum = calibrated_analyze(capture.samples, capture, SpectrumConfig(rbw_hz=1e3))
print(spectrum.peak_db, spectrum.bandwidth_99_hz)    # dBm per RBW, Hz

rx = RxConfig(modulation="qpsk", sample_rate=1e6, symbol_rate=50e3,
              timing_recovery="mueller-muller", coarse_cfo=True)
result = demodulate(capture.samples, rx, start_sample=0)
print(result.metrics["evm_rms_percent"])
```

`generate(tx, start_sample=a)` returns samples `a … a + N − 1` of the transmitter's stream, with `N = round(duration × sample_rate)`. The stream is a deterministic function of the design and the absolute sample index, so consecutive slices join exactly and the same slice is always the same samples. `vectorrf.source.source_record(tx, capture_id)` returns capture *k* (starting at *k·N*) through a shared cache.

## HTTP endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Version and limits (`max_sample_rate`, `max_capture_samples`, …) |
| `GET /api/catalog` | Modes, waveform families, radar waveforms, spectrum options, file formats, agility limits |
| `GET /api/defaults` | Default value of every editable model |
| `POST /api/tx/plan` | Validate a design and describe it without generating samples (below) |
| `POST /api/tx/preview` | Generate one capture of a `TxConfig` and return a bounded I/Q preview |
| `POST /api/spectrum` | Spectrum of a capture |
| `POST /api/measurements` | Zero span or modulation quality of a capture |
| `POST /api/rx/loopback` | Channel experiment, processing chain and demodulation of a capture |
| `POST /api/blind/analyze` | Classify the emitter in a capture (optionally in one band) |
| `GET /api/blind/features` | Definitions of the classifier's features |
| `POST /api/studio/pipeline` | Run a processing chain on a capture |
| `POST /api/studio/ofdm` | OFDM layout metadata of a design (resource grid, roles) |
| `POST /api/studio/radar` | One capture of a radar experiment |
| `GET /api/studio/extensions` | Installed local Python extensions |
| `GET /api/files`, `PUT /api/files/upload`, `POST /api/files/inspect`, `POST /api/files/overview`, `DELETE /api/files/{name}`, `POST /api/files/export` | The recording library (below) |
| `WS /ws/live` | The live stream (below) |

Observer requests (`/api/spectrum`, `/api/measurements`, `/api/rx/loopback`, `/api/blind/analyze`, `/api/studio/pipeline`) take a transmitter design `tx`, or a recording `source`, and a `capture_id`. Every observer that names the same source and capture sees the same samples, including the live stream.

## Transmitter designs

A `TxConfig` is one complete transmitter. The main groups of fields:

| Group | Fields |
| --- | --- |
| Clock | `sample_rate` (S/s, up to 5 GS/s), `duration` (capture length in s; at most 8,388,608 samples), `rf_center_hz` (the RF frequency baseband 0 Hz represents; labels only) |
| Output | `power_dbm` (mean power while on), `center_hz` (baseband offset), `phase_offset_deg`, `snr_db` (transmitter AWGN, `null` = off), `iq_gain_db`, `iq_phase_deg`, `dc_i`, `dc_q` |
| Waveform | `waveform` and its parameters (`symbol_rate`, `rolloff`, `filter_span_symbols`, `pulse_shape`, `fsk_deviation_hz`, `message_freq_hz`, `tones_hz`, `chirp_start_hz`, `chirp_stop_hz`, `chirp_period_s`, OFDM and radar fields …) |
| Payload | `data_source` (`text`, `pn`, `bits`), `message`, `custom_bits`, `pn_order` (7/9/11/15), `pn_bits`, framing (`frame_enabled`, `frame_mode`, `preamble_bits`, `payload_bits`, `frame_crc`, `frame_blocks`) |
| Gating | `burst_enabled`, `burst_period_s`, `burst_duty`; `agility` (hopping and sweeps) |
| Composite | `components` (with `waveform: "composite"`) |

Every frequency a design reaches — offsets, chirp sweeps, multitone tones, FSK/FM/PM peak deviation, hop and sweep ranges, composite components — must lie inside ±Fs/2 (sweeps may end exactly at the band edge). Designs that would alias are rejected with the frequency and the sample rate that would hold it.

`examples/designs/` holds tested example designs (a PN15 QPSK link, a tone plus a 10 MHz QPSK carrier, FHSS QPSK 900 MHz – 1 GHz, pseudo-random FHSS, swept tones, independently chirping tones, OFDM with interferers); each file is `{"name", "description", "tx"}`.

### Composite signals

```json
{
  "waveform": "composite", "sample_rate": 60e6, "duration": 0.002, "snr_db": 40,
  "components": [
    {"label": "CW tone", "waveform": "tone", "power_dbm": -10, "center_hz": 0},
    {"label": "QPSK 10 MHz", "waveform": "qpsk", "power_dbm": -20, "center_hz": 20e6,
     "symbol_rate": 7.5e6, "rolloff": 0.3333333333333333, "data_source": "pn"}
  ]
}
```

Each component is a complete `TxConfig` (up to 16; `enabled` and `label` optional) with its own absolute `power_dbm`, offset, phase, payload, burst gating and hopping. The parent's `sample_rate`, `duration` and `rf_center_hz` are copied into every component, and components cannot be composites themselves. The parent's `center_hz`, `phase_offset_deg`, `agility`, burst settings and impairments apply to the sum. Components get distinct derived seeds, so two copies never share noise or PN data; identical deterministic content (a duplicated tone) adds coherently, and the reported total power accounts for it.

### Hopping and sweeps

`agility` is accepted by any design and by each component:

| Field | Meaning |
| --- | --- |
| `enabled` | Apply the schedule |
| `mode` | `hop` or `sweep` |
| `plan` | Hops: `linear` (start → stop in `step_hz`), `list` (`frequencies_hz`, up to 16,384) or `random` (seeded permutation of the linear grid) |
| `start_hz`, `stop_hz`, `step_hz` | Grid or sweep range, as baseband offsets from `rf_center_hz` |
| `order` | `up`, `down` or `up-down` |
| `dwell_s`, `off_s` | On time per hop or sweep, then gated-off time |
| `ramp_s` | Raised-cosine rise/fall at each dwell edge (at most half the dwell) |
| `offset_s` | Delay before the schedule starts |
| `phase` | `continuous` (VCO-like) or `coherent` (each hop at the phase of its own oscillator since t = 0); sweeps are always continuous |
| `seed` | Seed of the random plan |

QPSK hopping 900 MHz → 1 GHz in 100 kHz steps (1001 hops), 100 µs dwell, 20 µs off-time:

```json
{
  "waveform": "qpsk", "sample_rate": 125e6, "rf_center_hz": 950e6, "symbol_rate": 80e3, "rolloff": 0.2,
  "agility": {"enabled": true, "mode": "hop", "plan": "linear", "start_hz": -50e6, "stop_hz": 50e6,
              "step_hz": 100e3, "dwell_s": 100e-6, "off_s": 20e-6, "ramp_s": 2e-6}
}
```

To receive a hopping signal, put its schedule in `RxConfig.dehop` (up to two `agility` objects, outermost first; the UI fills it when the receiver follows the transmitter). The receiver multiplies by the conjugate of the same oscillator at the same absolute samples before the processing chain.

### The design plan

`POST /api/tx/plan` validates a design (any JSON body) and returns, without generating samples: the normalized `config`, `capture_samples`, the nominal occupied `band_hz` and `carrier_extent_hz`, `total_power_dbm`, per-component bands and powers, hop/sweep summaries (segments, period, duty, hop rate and the first segments of the plan) and aliasing `warnings`. A rejected composite also returns `components`: one message or `null` per component, so an editor can mark the components at fault.

## Instruments

### Spectrum

`POST /api/spectrum` takes `tx` or `source`, `capture_id` and `spectrum`, a `SpectrumConfig`:

| Field | Meaning |
| --- | --- |
| `center_hz`, `span_hz` | Analyzed band; spans up to Fs/4 are digitally down-converted (zoom FFT) |
| `rbw_hz` | Requested resolution bandwidth (the window's equivalent noise bandwidth) |
| `window` | `blackmanharris` (default), `hann`, `flattop`, `kaiser`, `rect` |
| `detector`, `averaging` | `average`, `peak` or `sample` over `averaging` 50 %-overlapped segments |
| `max_fft` | Longest window (256 … 1,048,576) |
| `channels` | Up to eight `{"label", "center_hz", "width_hz"}` integration bands |

The response holds `freq_hz`, `psd_db` (dBm per RBW for transmitter captures, dBFS for recordings unless calibrated), `nfft` (window length), `fft_size` (the zero-padded transform, four times the window up to 65,536 points), `rbw_actual_hz`, `rbw_limited_by_capture`, `peak_hz`, `peak_db`, `bandwidth_3db_hz`, `bandwidth_99_hz`, `decimation`, `analysis_rate` and one result per channel: `power_db`, `density_db_hz`, `obw99_hz`, `peak_hz`, `peak_db`, `unit` and `covered` (whether the analyzed span covers the whole band).

### Zero span and modulation quality

`POST /api/measurements` takes `tx` or `source`, `capture_id`, the interval `offset_s` and `duration_s`, optional zero-span tuning `center_hz` and `bandwidth_hz` (mix to DC and band-limit before detection), and `quality`.

Without `quality` the response holds the zero-span traces (`time_s`, `i`, `q`, `amplitude`, `phase_deg`, `frequency_hz`, `valid`, `frequency_valid`) with the achieved `bandwidth_actual_hz`, `analysis_rate` and `decimation`, and whether the interval was clipped to the capture.

With `quality` it holds `quality`: for linear formats and OFDM `kind: "evm"` with `metrics` (`evm_rms_percent`, `evm_peak_percent`, `mer_db`, `symbol_count`, …), error-vs-symbol traces (`evm_time_percent`, `magnitude_error_percent`, `phase_error_deg`) and `symbols`; otherwise `kind: "basic"` with the waveform's envelope, phase and instantaneous frequency. The live stream's version of this product tracks the carrier from capture to capture and measures the most recent 2,048 tracked symbols; an HTTP request measures one capture.

### Receiver

`POST /api/rx/loopback` takes `tx` or `source`, `capture_id`, a `channel` (`ChannelConfig`: `snr_db`, `freq_offset_hz`, `phase_offset_deg`, `timing_offset_samples`, `echo_gain`, `echo_delay_symbols`, I/Q imperfections), an optional processing `pipeline` and `rx` (`RxConfig`):

| Field | Meaning |
| --- | --- |
| `modulation`, `symbol_rate`, `rolloff`, `filter_span_symbols`, `pulse_shape` | The format to decode |
| `tuner_hz` | Mixed to 0 Hz before demodulation |
| `timing_recovery`, `timing_gain`, `timing_phase` | `manual`, `mueller-muller` (joint with the carrier loop) or `gardner`; the gain is in symbols per unit detector output |
| `coarse_cfo` | Coarse carrier acquisition (spectral centroid, M-th power and symbol-rate fine estimates) |
| `carrier_algorithm`, `costas` | `costas`, `decision-directed` or `off`; loop gains `alpha`, `beta` refer to unit-RMS symbols |
| `equalizer`, `equalizer_taps`, `equalizer_gain` | `off` or `cma` |
| `frame_enabled`, `frame_blocks`, energy and preamble settings | Generic frame acquisition and block decoding |
| `dehop` | Known hop/sweep schedules to remove |

The response holds `metrics` (EVM excluding the loops' acquisition, `evm_all_percent`, `evm_settle_symbols`, symbol counts), `symbols`, `raw_symbols` (before carrier recovery), decoded `bits`, `hex` and `text`, and `telemetry`: coarse CFO (`coarse_cfo_hz`, `spectral_cfo_hz`), loop traces, eye diagram, EVM traces and, for PSK/QAM transmitter captures, a reference-aided channel estimate (`channel_estimate_taps`, `channel_estimate_phase_deg`) of the symbols before any equalizer.

### Classifier

`POST /api/blind/analyze` takes `tx` or `source`, `capture_id` and optionally `band_center_hz`/`band_width_hz` to isolate one emitter. The result names the `modulation`, `family`, a heuristic `confidence`, bandwidths, symbol rate and SNR, `candidates` (constellation hypotheses with `evm_percent`, `score`, `points_used`, `penalized`) and `trace`: one entry per rule stage (`key`, `title`, `question`, `outcome`, `detail`) with its `checks` (`feature`, `label`, `value`, `op`, `threshold`, `unit`, `passed`). `GET /api/blind/features` returns `{"features": {key: {"label", "description"}}}`.

## The live stream

`/ws/live` serves one live session per connection. The client sends JSON messages:

```jsonc
{"action": "start", "tx": {/* TxConfig */}, "source": null, "spectrum": {/* SpectrumConfig */}, "fps": 30, "pacing": "realtime",
 "rx": null, "pipeline": null, "channel": {/* ChannelConfig */}, "measurement": null,
 "acknowledge_frames": true, "client_revision": 1}
```

`update` replaces any part of the configuration, `ack` acknowledges a frame (`{"action": "ack", "seq": n}`) and `stop` pauses. The server answers with `hello`, `meta` (when the source changes), then for each frame a `metrics` message (capture identity, analyzer state, RBW, peak, channel results; the frequency axis only when it changes) followed by a binary PSD frame (`VRF1` header: magic, type, sequence number, payload length, then float32 dB values), then at most one product — `receiver`, `pipeline` or `measurement`, by priority of what the request enables — and, with acknowledgements, `frame_end`. Errors arrive as `error` messages with a `scope` (`request`, `tx`, `rx`, `pipeline`, `measurement`, `product`); a bad receiver, chain or measurement setting disables only that product. Messages carry the `client_revision` they answer, so a client can ignore replies to superseded edits.

`pacing` selects how the stream advances: `contiguous` (the API default) consumes every capture in order, so filter and loop state stay physically continuous and simulated time may run slower than real time; `realtime` follows the wall clock and drops what cannot be processed in time, like an SDR dropping buffers. The server accepts connections from its own origin only.

## Recordings and file formats

Any request that takes a transmitter configuration also accepts a `source` naming a recording in the capture library (`VECTORRF_CAPTURES_DIR`); `tx` is then ignored for sampling.

```json
{
  "source": {
    "kind": "file", "name": "capture_2Msps.iq", "format": "ci16_le", "sample_rate": 2000000,
    "center_frequency_hz": 433920000, "header_bytes": 0, "start_sample": 0, "capture_samples": 65536,
    "step_samples": null, "loop": true, "power_offset_db": 0, "power_unit": "dBFS"
  },
  "capture_id": 3
}
```

Capture `k` covers `capture_samples` samples from `start_sample + k·step_samples` (`step_samples` defaults to `capture_samples`, i.e. contiguous playback). `format`, `sample_rate` and `center_frequency_hz` override what the file declares; raw files need `sample_rate` (and usually `format`). Sample formats use SigMF names: `cf32_le`, `cf64_le`, `ci32_le`, `ci24_le`, `ci16_le`, `ci8`, `cu8`, big-endian `_be` variants and real `r…` types.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/files` | Library folder and recordings with detected format, rate, duration |
| `PUT /api/files/upload?name=…` | Stream a request body into the library (never overwrites unless `overwrite=true`) |
| `POST /api/files/inspect` | Full metadata for a `FileSource`, ranked format candidates and contradiction warnings |
| `POST /api/files/overview` | Sampled whole-file spectrogram, power vs time and average spectrum |
| `DELETE /api/files/{name}` | Delete a recording (both halves of a SigMF pair) |
| `POST /api/files/export` | Write a transmitter capture or a recording window to SigMF, WAV/RF64 or raw |

Export body:

```json
{
  "tx": {"waveform": "qpsk", "sample_rate": 2000000000, "symbol_rate": 200000000, "duration": 0.0001},
  "capture_id": 0, "format": "ci16_le", "container": "sigmf", "scaling": "peak", "headroom_db": 1,
  "destination": "download"
}
```

`duration_s` or `samples` request a longer contiguous record (up to `VECTORRF_MAX_EXPORT_SAMPLES`). With `source` instead of `tx`, the window `[start_sample, start_sample + samples)` of a recording is converted. Integer formats are peak-normalized with `headroom_db` below full scale (or `scaling: "unity"`, which refuses to clip); the response headers report `X-VectorRF-Clipped`, `X-VectorRF-Scale` and `X-VectorRF-Samples`. SigMF downloads are `.sigmf` archives (an uncompressed tar holding the meta/data pair); saving to the library writes the pair directly. SigMF exports of a transmitter capture record the RF centre as `core:frequency`, a power calibration (`vectorrf:full_scale_dbm`) and the complete design (`vectorrf:tx_design`). WAV stores integer rates up to 4,294,967,295 S/s and switches to RF64 beyond 4 GiB.

Reading files directly in Python:

```python
from vectorrf.iqfile import inspect, RecordingReader

info = inspect("capture.cs16", sample_rate=20e6)   # or a .wav / .sigmf-meta / .sigmf
reader = RecordingReader(info)
window = reader.read(start=1_000_000, count=65536)   # complex64, 0 dBFS = magnitude 1
```

`iqfile.sniff_raw(path)` ranks plausible encodings of a headerless file; treat its result as a suggestion to confirm.

## Radar experiments

`POST /api/studio/radar` takes a `RadarExperiment`:

| Group | Fields |
| --- | --- |
| Transmit waveform | `source`: `program` (`program`), `legacy` (a Radar Studio pulse, segment or continuous design in `legacy`), `ofdm` (`ofdm`, a `TxConfig`) or `file` (`file`, a recording window) |
| Capture | `sample_rate`, `samples` (≤ 262,144), `capture_id`, `time_s`, `carrier_hz` |
| Scene | `targets` (≤ 12: `position`, `velocity`, `acceleration` in m and m/s, `rcs_m2`, `gain_db`, `fluctuation`) |
| Array | `antenna`: `geometry`, `columns`, `rows`, `spacing_lambda`, custom `positions`, complex taper weights, element pattern, steering, scan, `beamformer` (`conventional` or `mvdr`) |
| Link | `power_model` (`relative` or `radar-equation`), `power_w`, gains, `noise_figure_db`, `noise_db`, `loss_db`, `clutter_db`, `leakage_db` |
| Search | `max_range_m`, `max_velocity_mps`, `doppler_bins`, `cfar_guard`, `cfar_train`, `cfar_pfa` |
| Processing | `processing` and an optional processing `pipeline` |

`processing` takes `method` (`correlation`, `pulse-doppler`, `fmcw`), `pri_s`, `pulse_width_s`, `chirp_slope_hz_s`, `clutter` (`off`, `mean`, `two-pulse`, `three-pulse`), `range_window` and `doppler_window` (boxcar, Hann, Hamming, Blackman), `min_range_m`, `detector` and `os_rank`, and `threshold_db`. Detectors are `ca-cfar`, `go-cfar`, `so-cfar` and `os-cfar` (the `os_rank` fraction of the training cells sets the noise level), each with its multiplier solved for `cfar_pfa` per cell, or `manual` (a fixed `threshold_db`). Coherent processing starts at the next known interval boundary within each capture and needs four complete intervals; processing settings must match the source timing, while correlation handles arbitrary timing.

The result holds the transmitted, received and processed previews, the spectrum, the range profile and its threshold, the range–Doppler map, `detections` (`range_m`, `velocity_mps`, `power_db`, `margin_db` above threshold; one per cluster of cells above threshold), the strongest cell with its angle estimate, the array pattern, the angle map, the target `truth`, `warnings` and, for legacy designs, `design`: PRF, duty, bandwidth, compression ratio, range resolution, blind and unambiguous range, unambiguous velocity and velocity resolution per sequence item.

## Custom waveforms

### Expressions

Transmitter → Basic → Custom and the radar waveform source share one mathematical program (`WaveProgram`). Expressions return one complex sample per requested time. They support arithmetic, comparisons, `&`/`|` masks, and these functions:

`sin cos tan exp sqrt abs angle real imag conj log log10 floor ceil sign tanh sinc minimum maximum where clip`

| Variable | Meaning |
| --- | --- |
| `t` | Time modulo the program's period, in seconds |
| `time` | Absolute simulation time, in seconds |
| `T` | The period, in seconds |
| `n` | Sample index within the requested capture |
| `fs` | Sample rate in samples/second |
| `frame` | Capture number |
| `j`, `pi` | Imaginary unit and π |
| Named parameters | Values from the program's `parameters` dictionary |

```python
exp(j*2*pi*25000*time)                                    # a tone with continuous phase
where(t < width, exp(j*2*pi*(f0*t + slope*t*t/2)), 0)     # a gated chirp (parameters f0, slope, width)
exp(j*(2*pi*20000*time + 2*sin(2*pi*30*time)))            # phase modulation
0.7*exp(j*2*pi*15000*time) + 0.3*exp(-j*2*pi*35000*time)  # two components
```

The interpreter evaluates array arithmetic over a parsed syntax tree; it never executes Python. Attribute access, imports, assignments, subscripts and other function calls are rejected. Expressions are bounded to 8,000 characters and 512 nodes, exponents to scalars of magnitude ≤ 32, and the result to finite samples of magnitude ≤ 10⁹. Use a local Python extension for algorithms that need loops, FFTs, state machines or other libraries.

### Control points and I/Q tables

Control-point times are in microseconds and must increase. Frequency is integrated piecewise-linearly, while amplitude and extra phase interpolate; samples outside the point interval are zero. The program repeats at its period.

An I/Q table defines arbitrary complex samples played at its own `table_rate` with `linear` or `hold` interpolation; I `[1, 0, -1, 0]`, Q `[0, 1, 0, -1]` is a four-sample complex rotation.

### OFDM resource grids

The carrier coordinate is the signed FFT index (DC `0`), spaced `sample_rate / ofdm_fft`. An empty active-carrier list selects the guarded allocation; an explicit list can select separated subbands. Per-cell overrides apply after the default data/pilot assignment:

```json
[
  {"symbol": 0, "carrier": -3, "kind": "pilot", "i": 1, "q": 0},
  {"symbol": 0, "carrier": 2, "kind": "value", "i": 0.5, "q": -0.5},
  {"symbol": 1, "carrier": 4, "kind": "null"},
  {"symbol": 2, "carrier": 6, "kind": "data", "modulation": "64qam", "gain": 0.7}
]
```

`symbol` is zero-based within the frame. Known values support pilot channel estimation; mixed data cells are sliced with their own mapping and gain. A list of cyclic-prefix lengths cycles across symbols; WOLA (`window_samples`, `window_kind` `raised-cosine`, `linear` or `custom` with `window_ramp`) must fit in the shortest prefix. `output_filter` (`off`, `lowpass` with `filter_cutoff_hz`, `filter_length`, `filter_window`, or `custom` taps `filter_i`/`filter_q`) is applied after CP/WOLA, with the preceding samples rendered so the convolution crosses capture boundaries. The receiver knows the frame position from the absolute sample index (`ofdm_timing_offset` skips a known delay); it does not search for unknown frame boundaries.

## Local Python extensions

Python files in the project's `extensions/` folder are loaded at server start and listed by `/api/studio/extensions`. They run with the server's permissions: only install extensions you trust.

A **waveform** extension provides `waveform(context)`:

```python
import numpy as np

def waveform(context):
    f = context["parameters"].get("frequency_hz", 25000)
    return np.exp(2j * np.pi * f * context["time"])
```

The context holds `t`, `time`, `fs`, `frame`, `parameters` and a deterministic NumPy `rng`. Return a one-dimensional finite complex array of `len(context["time"])` samples. Radar evaluates sources at delayed times as well as the current time, so pure functions of time and parameters are strongly preferred. `extensions/polynomial_chirp.py` is an example.

A **processing** extension provides `process(samples, sample_rate, context)`:

```python
import numpy as np
from scipy.signal import lfilter

def process(samples, sample_rate, context):
    memory = context["state"]
    taps = np.array([0.25, 0.5, 0.25])
    zi = memory.get("zi", np.zeros(len(taps) - 1, complex))
    out, memory["zi"] = lfilter(taps, [1], samples, zi=zi)
    return out, sample_rate
```

The context holds `time_s`, `frame`, `parameters` and a mutable `state`. Return an array, or `(array, new_sample_rate)` for rate-changing algorithms. In the live receiver chain the state persists across captures until the source, channel or chain changes (or **Reset state**); a radar chain runs once per simulated capture. `extensions/iir_notch.py` is a stateful notch filter.

## Units and conventions

- Frequencies are complex-baseband offsets in Hz relative to `rf_center_hz`; the UI shows absolute RF values when an RF centre is set.
- Transmitter power is dBm while on (a pulsed generator's level setting); spectra read dBm per RBW, recordings dBFS (a complex sample of magnitude 1 is 0 dBFS) plus an optional offset.
- Radar velocity is positive for receding targets; Doppler is `−2v/λ`.
- Loop gains refer to unit-RMS symbols and timing gains to symbols, so they behave the same at any level and oversampling.
- Numeric UI fields accept engineering notation: `2.4G`, `2.4 GHz`, `2.4ghz`, `100k`, `20us`, `1e6`, `−3 dBm`. In frequency, sample-rate and symbol-rate fields `m` means mega (`10mhz` is 10 MHz).
