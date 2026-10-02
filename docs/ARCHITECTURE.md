# Architecture

## Capture and processing flow

Transmitter owns the communications configuration. `source_record(config, capture_id)` produces a deterministic immutable complex array, with a configuration revision and simulation time. A bounded cache allows different observers to retrieve the same capture. Capture IDs advance in the WebSocket engine; source randomness is seeded by the ID. This replaces the old repeated immutable-record observer loop.

`/ws/live` schedules spectrum captures independently of edits, validates updates, acknowledges accepted configurations and sends JSON metrics followed by binary `VRF1` PSD. It can also process an optional Receiver chain on each full capture and send a `pipeline` product. Its processing-memory dictionary is private to the socket and reset when DSP settings change. Configuration signatures exclude the probe selector.

Receiver convenience demodulation, quality and Classifier HTTP requests include the displayed source capture ID. Their cadence is independent of the spectrum display; their results identify which capture was actually processed. They can lag the newest spectrum frame. Browser reconnect preserves the capture counter, while DSP memory on a new socket starts empty.

## Programmable modules

| Module | Responsibility |
| --- | --- |
| `expressions.py` | Restricted AST array expressions with finite/size bounds |
| `programs.py` | Mathematical, control-point, I/Q table and local Python waveforms |
| `ofdm.py` | Resource-grid construction, CP/WOLA synthesis, capture offsets, CP CFO and pilot-LS reception |
| `pipeline.py` | Ordered processing blocks, scopes, stage measurements and live state |
| `radar_sim.py` | Full-waveform point-target propagation, array channels, beamforming, correlation and angle estimation |
| `source.py`, `engine.py` | Capture identity, bounded reuse and scheduled transport |
| `signal.py`, `rx.py`, `sync.py`, `modem.py` | Existing communications generation/demodulation and recovery algorithms |
| `radar_studio.py` | Existing pulse, segment and continuous-wave designer feeding the new simulator |
| `backend/vectorrf/web/app.js` | Shared instrument shell, plots and existing instrument workflows |
| `backend/vectorrf/web/studio.js` | Programmable editors, OFDM grid/receiver, processing bench and radar experiment UI |

The static scripts load in order: app.js → studio.js → workbench.js. The final script installs the general receiver, organized controls and custom steppers before initialization. The waveform/processing Python extension catalog is loaded from the local project `extensions` folder at startup; network requests select installed names, not file paths or uploaded Python source.

## Radar experiment

A request to `/api/studio/radar` carries source design, sample rate/size, capture ID/time, target kinematics, array geometry/weights/pattern, link-budget controls and processing blocks. The full source is sampled at current and delayed times. Each point target contributes an echo with delay, radial Doppler and spatial phase at each receive element. Beamforming precedes the editable processing chain. Delay/Doppler correlation uses the transmitted waveform reference.

Range/Doppler strongest-cell samples across the array drive the angle map. The result distinguishes measured coordinates from input target truth. The front end advances simulation time explicitly and labels all scopes with their processed capture. It permits only one radar request in flight and rejects results from an older design revision.

Legacy `/api/radar/studio`, `/api/radar/scene` and preview endpoints remain for compatibility. The current Radar UI uses `/api/studio/radar`, including full legacy sequence waveforms.

## Bounds

Communications source captures: two million samples. Source cache: 12 identities. Radar input: 32,768 samples, 32 elements and 12 targets. Radar processed output: 65,536 samples. Processing chains: 24 blocks. OFDM: FFT up to 4,096, 128 frame symbols and 8,192 overrides. Display products are reduced after DSP; spectrum frames contain at most 1,600 display bins. Waterfall and classifier history remain bounded.

## Transport and performance design

The optional WebSocket `acknowledge_frames` flag sends a `frame_end` marker after all products; the browser sends `action: ack`. The server waits before generating another capture, with a bounded two-second recovery timeout. Legacy clients without this flag retain the original protocol. This bounds queuing rather than manufacturing intermediate samples.

Large HTTP plot products return a JSON response directly instead of repeating recursive framework conversion. Full DSP arrays remain separate from bounded display previews. OFDM grid placement and bit extraction use vectorized arrays. Radar caches only reference-independent Doppler mixers and array steering geometry; echoes and measurement products are recalculated per capture. Filter, waveform or scene edits still change the actual samples.

`radar_processing.py` implements explicit coherent pulse/FMCW processing. `channel_samples` shares deterministic communications channel noise/echo behavior across receiver paths. Live processing state stays per socket; convenience HTTP demodulation is capture-local.
