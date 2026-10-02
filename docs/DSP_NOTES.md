# DSP notes — 0.7

## Communications capture model

Each communications capture uses a seed derived from configuration seed and capture ID. AWGN and PN sources therefore evolve reproducibly. Receiver channel noise advances too, and its CFO phase uses absolute capture time. Mathematical programs evaluate absolute time directly. OFDM tracks absolute sample offset, CP position and grid row across captures.

Built-in non-OFDM digital waveforms still construct framing and shaping within each capture. They are not an arbitrary continuous hardware stream with all encoder/filter state preserved. Local waveform modules can provide time-indexed generation. The live processing chain carries FIR/lowpass/Costas/custom-Python state; timing, resampling and CMA remain capture-local. The convenience demodulators reacquire per capture.

## OFDM

A complex frequency-domain grid uses signed FFT indices. Default data/pilot placement is overridden by per-cell null, pilot, explicit-value or modulation/gain entries. A known mixed mapping is decoded by the corresponding per-cell constellation. Uniform BPSK, QPSK, 8-PSK, 16-QAM and 64-QAM are supported.

IFFT normalization is unitary. Prefix lengths cycle by symbol; optional WOLA uses cyclic suffixes and complementary sine-squared ramps within the shortest prefix. Absolute capture position includes an earlier symbol when needed to preserve overlap tails. Turning off Transmitter RMS normalization preserves exact designed sample amplitudes apart from the selected scalar gain.

The receiver knows frame/resource positions, discards an initial partial symbol, and uses complete symbols. CP correlation estimates fractional CFO modulo subcarrier spacing, approximately within ±Δf/2. Pilot LS estimates a complex channel per OFDM symbol and interpolates across signed frequency. Deep nulls, poor pilot spacing, delay exceeding the CP, large CFO and time variation can invalidate this simple estimator. Arbitrary unknown frame synchronization and MIMO are future work.

EVM is decision-directed. With pilot scale available, OFDM measures errors against the absolute known constellation scale; nonuniform text-driven QAM no longer acquires a false RMS normalization penalty. Error trajectories are exposed. Wrong decisions can still make EVM optimistic; it is not BER or standards compliance.

## Radar source and propagation

The simulator accepts mathematical/control-point/sample-table/Python programs, the complete legacy sequence or continuous design, and a complete OFDM frame. Stored legacy and OFDM designs repeat; mathematical programs can use absolute time for nonrepeating behavior. Delayed evaluations use the same source definition.

For target range R and receding radial velocity v, echo delay is `2R/c` and carrier Doppler is `-2v/λ`. Spatial phase is computed separately at every receive element from its position in wavelengths and the target direction. Target position includes configured velocity and acceleration across simulation time. Range migration and acceleration within one capture are neglected (stop-and-hop approximation).

Relative mode uses amplitude proportional to `sqrt(RCS) / R²`, referenced to 3 km. Radar-equation mode uses `Pt Gt Gr λ² σ / ((4π)³ R⁴)` power, with configured losses and extra echo gain. Analytic TX-array and TX/RX-element responses multiply the echo. Thermal noise uses `k T Fs F` at 290 K. Array weights are normalized; this is a system-level approximation rather than a calibrated hardware front end. Swerling-I draws one exponential power multiplier per target/capture.

The clutter control currently adds spatially common complex Gaussian samples. It is a noise-like clutter experiment, not a terrain or Doppler-spectrum clutter model. TX leakage is a coherent scaled copy of the transmitted samples.

## Arrays and estimation

The coordinate convention is X forward, Y lateral, Z up, with a front-hemisphere angle search. Arrays can be linear, planar or arbitrary element positions. Element patterns are isotropic, cosine, Gaussian or a tabulated azimuth cut. Complex tapers and steering influence the actual receive samples. Scanning sweeps azimuth sinusoidally through ±60°; the rate field controls its angular phase rate.

Conventional beamforming uses the configured steering/taper. MVDR estimates sample covariance, adds 1% trace-based diagonal loading and normalizes the steering response. It can suppress desired signals outside the chosen look direction. This is not a robust adaptive beamformer under all model mismatch.

Processing uses full-waveform delay/Doppler correlation over the displayed search interval. It works for arbitrary known references without silently substituting a preset pulse. Range sampling is `c/(2Fs)` at the processing rate. Velocity bins are the configured search grid; useful Doppler resolution still depends on coherent observation duration. Periodic signals can have delay/velocity ambiguities and chirps can exhibit range–Doppler coupling. Oversampled search grids do not create physical resolution.

The strongest measured range/Doppler cell is coherently extracted from the raw array channels and scanned with conventional steering vectors. Reported angles/XYZ come from this array response, not target truth. Linear arrays do not determine elevation uniquely. Multiple targets sharing a cell, grating lobes, low SNR, receiver-chain delay and ambiguous waveforms can yield incorrect or nonunique strongest-cell estimates.

CA-CFAR uses training and guard cells on the Doppler-maximized range profile. Its usual independent-noise scaling is applied per range test; sample correlation and Doppler maximization make the overall false-alarm probability uncalibrated. Detections and strongest-cell readouts are separate products.

## What a live scope means

All radar live plots use the latest processed capture and expose its ID/time. Receive noise, moving targets, scan weights and time-dependent waveforms change the numerical data. Fixed waveform correlation or a fixed beam pattern may be invariant; they are recomputed/current reference properties, not artificially animated measurements. Pause freezes the simulation; Step acquires one new capture.

Spectrum and zero span calculate at acquisition rate before bounded display reduction. The spectrum display span may be wider than the source sample-rate Nyquist interval; in that case the UI marks the unavailable outer region instead of inventing data, while the DSP spectrum itself remains limited to sampled frequencies. Near-zero phase/frequency samples are masked. Extremely narrow events can be lost by display sampling. The communications spectrum uses an explicit simulator calibration: transmitter average power is the configured dBm value and FFT bins are reported in dBm. A bin-centered CW therefore reads the configured transmit power at its peak; modulated signals distribute the same average power across bins. This is an internal baseband power calibration, not an RF front-end/impedance/antenna model. Classifier hypotheses and scores remain heuristic and do not identify arbitrary new protocols automatically.

## 0.7 processing additions

Pulse-Doppler processing uses fast-time matched filtering per transmitted pulse and slow-time Doppler FFT. FMCW uses `TX * conj(RX)` dechirping, fast-time beat-frequency FFT and slow-time Doppler FFT. Signs follow positive receding velocity. Configured PRI, pulse width and chirp slope must match the waveform. Doppler shifts bias FMCW range. Range weighting trades resolution for sidelobes; MTI and coherent mean removal suppress zero-Doppler clutter and may suppress desired targets. No tracking or calibrated CFAR global-Pfa model is implied.

Manual detector thresholds use the current processor's normalized correlation/FFT power in dB, not calibrated dBm. Gate edges and range spacing are derived from the chosen processor. Native slow-time FFT bins may be fewer than the requested display-bin count; interpolation/zero padding does not create independent measurements.

OFDM output FIR filters are causal. The source provides prehistory for convolution; arbitrary filters may violate cyclic-prefix assumptions. Custom overlap ramps use a complementary falling ramp but need not have favorable spectral properties. Tests verify their actual sample effects and delay tuning.
