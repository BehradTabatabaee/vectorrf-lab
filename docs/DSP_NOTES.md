# DSP notes and model assumptions

## Communications capture model

Capture *k* of a transmitter configuration starts at absolute sample *k·N* (N samples per capture). Transmitter AWGN is keyed by absolute sample index. PN payloads, framing, symbols, pulse-shaping history and differential phase are indexed by absolute bit/symbol position, so changing the capture length does not change the transmitted stream. Analog messages, carrier phase, hops, bursts, mathematical programs and OFDM grid position also use absolute sample time. Adjacent captures are exact adjacent slices of one deterministic stream.

Receiver channel noise advances with the capture, and its CFO phase uses absolute sample time; the channel experiment's fractional-delay filter and echo still see capture edges. The live processing chain carries FIR/lowpass/Costas/custom-Python state across contiguous captures, and live demodulation carries carrier estimates across captures. Gaps under real-time pacing do not claim to preserve unobserved processing-chain samples. The convenience HTTP demodulators reacquire per capture.

## Time and phase

Carrier offsets, the receiver CFO experiment, the processing-chain mixer, analog messages, multitone components and burst gating are computed from integer sample indices. The phase at the first sample is `frac(f · start_sample / fs)` evaluated with exact rational arithmetic (`fractions.Fraction`); the in-capture increment is a float64 fraction per sample, reduced modulo one before scaling by 2π. Evaluating `f · (t + start_s)` in float64 instead rounds `t + start_s` at ~10⁻¹² s after hours of simulated time — at 2 GS/s that is about −50 dBc of phase-error spurs. User expressions that use `time` are evaluated in float64 as written.

Analog modulations use closed forms for the sinusoidal message m(t) = sin 2πf_m t: SSB is the exact analytic signal ∓j·e^{±j2πf_m t} (no finite-record Hilbert ringing) and FM uses the exact phase integral (Δf/f_m)(1 − cos 2πf_m t).

The basic chirp is a sawtooth: within each sweep period T = P/fs the phase is `2π(f₀τ + ½kτ²)` with k = (f₁ − f₀)/T, and sweep r starts with the phase of r whole sweeps, `r·(f₀ + f₁)T/2` cycles, so the flyback is a frequency step without a phase step (like a VCO; the FMCW radar waveform and agility sweeps are continuous in the same way).

## Frequency hopping and sweeps

A hop/sweep schedule is a periodic sequence of segments: a dwell of `N_on = round(dwell·fs)` samples (minimum 1) followed by `N_off = round(off·fs)` gated-off samples, during which the oscillator holds its last frequency. Raised-cosine ramps of `N_r` samples (at most `N_on/2`) shape the start and end of every dwell; the start delay shifts the whole schedule.

The oscillator is a 64-bit direct digital synthesizer. Each frequency is a tuning word `W = round(f/fs · 2⁶⁴)` (two's complement for negative frequencies) and the phase is an unsigned 64-bit accumulator, so phase arithmetic is exact modulo 2⁶⁴ and the frequency resolution is fs/2⁶⁴. The phase of absolute sample *k* is evaluated in closed form — the phase accumulated by all complete periods, then by the complete segments of the current period, then within the current segment — so any capture can be generated independently and consecutive captures join exactly. Only the final conversion `2π·φ/2⁶⁴` to cos/sin is float64 (phase error ≈ 10⁻¹⁵ rad).

- **Continuous** hop phase integrates the frequency through every segment, including off-times, like a VCO: no phase steps at hop boundaries.
- **Coherent** hop phase is `W_i · k` for absolute sample *k*: each hop has the phase its own oscillator would have had since t = 0, like a bank of coherent synthesizers; there are phase steps at hop boundaries, but a hop's phase does not depend on the hop order.
- **Sweeps** add a per-sample increment `ΔW = round((f₁ − f₀)/fs · 2⁶⁴ / N_on)`; within a segment the phase is `u·W₀ + ΔW·u(u−1)/2` (the triangular-number term, also exact modulo 2⁶⁴). Sweeps are always continuous.

A linear plan visits `start, start + step, …` up to `stop` (inclusive within a 10⁻⁹ step tolerance); a random plan is a seeded permutation of the same grid, so every frequency is used once per period. Agility multiplies the clean, normalized signal before transmitter AWGN, so the noise floor stays during off-times and the configured power is the power while on.

The receiver's synchronized de-hopper multiplies by the conjugate of the same oscillator (same schedule, same absolute sample index). Off-times and ramps remain in the de-hopped signal: EVM in modulation quality is reported only for continuous dwells, otherwise the quality page shows the de-hopped trajectory.

## Composite signals

Every enabled component *i* is generated as a complete signal normalized to unit clean RMS, then scaled by `aᵢ = 10^((Pᵢ − P_ref)/20)` with `P_ref = maxᵢ Pᵢ`. The sum is multiplied by `g = A/√P` (A is the composite amplitude) and the calibration maps RMS `g` to `P_ref` dBm, so component *i* reads exactly `Pᵢ`. The expected total `P = Σᵢ Σⱼ aᵢ aⱼ Re ρᵢⱼ` (relative to `P_ref`) uses the correlation coefficients ρ of the clean components, measured over the first 65,536 samples of the stream: components with distinct derived seeds (noise, PRBS) and distinct carriers are uncorrelated and the total is `10·log₁₀ Σ 10^(Pᵢ/10)`, while identical deterministic content adds coherently — a duplicated tone at the same frequency and phase reads +6.02 dB, at 90° +3.01 dB, in antiphase it cancels. Because the gain is fixed by the design, bursting or hopping components and off-times never rescale the others between captures; while a component is off, the total power is lower than the nominal sum. Composite-level agility, burst gating and impairments apply to the sum.

The combined carrier extent — offsets, chirp sweep, multitone tones, hop/sweep ranges and components, recursively — must lie within ±Fs/2 (exactly ±Fs/2 is accepted); designs that would alias are rejected. Modulated bandwidth beyond the carrier extent (filter skirts) is reported by the band plan as a warning, not rejected.

## Absolute power

Transmit power (dBm) is the average power of the **clean signal** — after RMS normalization and before AWGN, I/Q imbalance, DC and burst gating. The configured SNR is relative to the same clean power, so impairments add on top of it; a burst reads the configured power while on (like a pulsed signal generator's level). The spectrum calibration maps that reference RMS to the configured dBm, so a bin-centred complex tone reads its configured power at the peak and modulated signals distribute the same power over their occupied bins. This is an internal baseband calibration, not an RF front-end/impedance/antenna model.

Recordings use dBFS: a complex tone of magnitude 1 (|I + jQ| = 1) is 0 dBFS. Integer samples scale by 2^(bits−1); cu8 subtracts the RTL-SDR midpoint 127.5 and 8-bit WAV PCM subtracts 128. An optional offset converts dBFS to dBm when the receiver calibration is known. VectorRF SigMF exports record `vectorrf:full_scale_dbm`, the dBm value of full scale after export scaling, which restores absolute power on re-import.

## Spectrum estimation

Averaged periodograms use 50 % overlapping windowed segments. Without calibration the output is a PSD in dBFS/Hz, |X|²/(fs·Σw²). With calibration it is power per resolution bandwidth, |X|²/(Σw)² relative to the reference RMS: a bin-centred tone reads its full power and white noise reads σ²·RBW/fs. RBW is the equivalent noise bandwidth: ENBW_bins · fs / N (Hann 1.5, Blackman–Harris 2.00, flat top 3.77, Kaiser β=8.6 1.72, rectangular 1.0 bins).

A window never extends past the captured samples. When the capture is shorter than the window the requested RBW implies, the analyzer uses the largest power-of-two window that fits and reports the achieved RBW (`rbw_limited_by_capture`). A short capture placed under a long window would sit under its leading edge — a 0 dBm tone in a 4 ms capture at 10 Hz RBW would read −74.6 dBm with an RBW 30× too narrow.

Each windowed segment is zero-padded to four times its length before the FFT (`FFT_OVERSAMPLING`; twice or once for windows of more than 65,536 or 131,072 points). The resolution is still the window's, but bins lie at a quarter of it, so a tone shows the same RBW shape wherever it falls and loses less than 0.1 dB of peak level between bins (1.42 dB with Hann at one bin per window resolution, where an on-bin tone also hid its sidelobes in the window's nulls and looked unlike an off-bin one). Zero padding leaves the power per bin unchanged and multiplies the bins per RBW, so channel power divides by the window's ENBW in padded bins. The default window is a 4-term Blackman-Harris (−92 dB sidelobes), close to an analyzer's Gaussian RBW filter; flat top remains the window for amplitude accuracy. The −3 dB width interpolates the crossings on both sides of the peak and the 99 % occupied bandwidth interpolates the cumulative power across bin edges.

### Zoom FFT and channel power

For spans of at most Fs/4 the analyzer down-converts before estimating: the centre is mixed to DC and decimated through stages of 8, 4 and 2 while the output rate stays at least 1.25 × span (`ZOOM_OVERSAMPLING`). Each stage is a Kaiser-window FIR with pass band ±span/2 and ≥ 110 dB rejection (`ZOOM_REJECTION_DB`) from `f_out − span/2` upward: anything that could alias into the span is at least 110 dB down, and transition-band content folds outside it. Only outputs computed from complete filter support are kept. RBW is then `ENBW_bins · f_analysis / N`; the capture length needed for a requested RBW is `required_samples()`, which the live engine uses to size its chunks.

Channel power integrates the full-resolution spectrum, not the display. With calibration each bin holds the power in one RBW, so summing bins over-counts by the window's ENBW in bins: `P = Σ wₖ Pₖ / ENBW_bins`, where `wₖ` is the fraction of bin *k* inside the band. Without calibration `P = Σ wₖ PSDₖ · Δf`. Density is `P` divided by the band width; the 99 % occupied bandwidth interpolates the cumulative power inside the band.

Spectrum calculates from acquired samples before bounded display reduction. Tuned zero span mixes at the absolute capture sample index, anti-alias decimates toward the requested bandwidth, then filters and demodulates at the analysis rate. It reports the achieved bandwidth and analysis rate; a capture too short to resolve the requested bandwidth uses a wider achievable bandwidth instead. Live zero span clips its interval to the acquired prefix and reports that clipping. The display span may be wider than the source's Nyquist interval; the UI marks the unavailable region instead of inventing data. Near-zero phase/frequency samples are masked. Extremely narrow events can be lost by display sampling.

## Modulation, slicing and EVM

PSK/QAM constellations are unit-average-power and Gray coded (adjacent points differ by one bit); APSK uses natural labels. The RRC filter is unit-energy and symmetric, so matched filtering gives a Nyquist (zero-ISI) raised cosine up to truncation.

EVM and slicing fit the received symbols to the constellation by decision-directed least squares: the complex gain g = ⟨d, z⟩/⟨d, d⟩ to the current decisions d is removed and the decisions are refreshed until they stop changing. The iteration starts from RMS normalization; when that fit is poor (mean squared error above a quarter of the squared half-spacing) on a constellation with several radii, it also starts from the gains that match the symbol magnitudes to the radii (`radius_scales`: the best local minima of the mean squared radial error) and keeps the start with the smallest error. Magnitudes do not depend on the carrier phase or on how a payload uses the constellation: a repeating text can use a dozen of 64-QAM's points with an RMS 16 % above the constellation's, and an RMS start then converges to the wrong decisions. EVM removes the complex gain (scale and common phase); slicing removes only its magnitude, so residual rotation still produces decision errors instead of being silently corrected. Ideal symbols read 0 % EVM for any finite data mix (RMS normalization alone left 0.03 % on 16-QAM) and noise power does not bias the scale. EVM is decision-directed: at low SNR wrong decisions make it optimistic, and a rotation large enough to move points across decision boundaries is a carrier-recovery problem no decision-directed metric can undo. MER = −20·log₁₀(EVM_rms).

CPFSK demodulation averages the frequency discriminator over each symbol. The phase difference between samples k−1 and k measures the frequency of sample k (indexing it by k−1 mixes 1/sps of the next symbol into every decision: 25 % noise-free BER at 2 samples/symbol). Symbol boundaries lie on the transmitter's absolute symbol grid (manual timing, a capture can start anywhere in a symbol) or come from a non-data-aided search for the boundary that maximizes the spread of the per-symbol averages (a window straddling two symbols averages their tones towards each other). With `coarse_cfo` the tones' common offset is fitted — starting midway between the extreme tones seen, then alternating nearest-tone decisions with a least-squares offset — and removed before the decisions.

## Synchronization loops and EVM settling

Before timing and carrier recovery the matched-filter output is divided by its symbol RMS (`symbol_rms`: the RMS at the best sampling phase), so loop gains refer to unit-RMS symbols and behave the same at any signal level, oversampling and pulse shape. The second-order carrier loop updates `f ← f + β·e`, `φ ← φ + f + α·e`; with a phase detector of slope `K_d` (√2 for the QPSK Costas detector on unit-RMS symbols, 1 for BPSK) its natural frequency is `ω_n = √(β·K_d)` and damping `ζ = α·K_d/(2ω_n)` in radians per symbol, and it settles in about `4/(ζ·ω_n)` symbols. The defaults α = 0.04, β = 0.001 give ζ ≈ 0.75 and ω_n ≈ 0.038 rad/symbol (about 140 symbols to settle at QPSK).

Linear modulations are demodulated at no more than 16 samples per symbol. Higher oversampling is first reduced by a rational polyphase resampler (`resample_poly`, Kaiser β = 10, about 100 dB stop band, pass band about ±0.68 of the new Nyquist ≈ ±5 Rs, so carrier offsets up to about ±4 Rs pass intact), so the cost follows the number of symbols, results do not depend on the input rate (`test_demodulation_does_not_depend_on_oversampling`) and the carrier estimates below see every symbol of the block (131,072 input samples are three symbols at 40,000 samples per symbol).

### Carrier acquisition

With `coarse_cfo`, a cold block estimates the carrier in up to three stages:

1. **Spectral centroid** (`spectral_centre`): the band of width Rs(1+β) with the most power above the noise floor is located in a Welch periodogram and its power centroid taken, within ±max(Rs(1+β), 5 % of Fs) of the tuning so a stronger neighbouring signal is ignored. It has no ambiguity, but a payload whose spectrum is not symmetric biases it: under 1 % of Rs for PRBS, up to about 4 % for a short repeating text.
2. **M-th power line** at the working rate (M = 2, 4, 8 for BPSK, QPSK/QAM, 8-PSK; range ±Fs/2M), kept only when it agrees with the centroid within 5 % of Rs. A periodic payload can make a false line stronger than the carrier's (16-QAM carrying a 12-character text has one at +Rs/8); APSK and π/4-DQPSK, whose symbol-clock harmonics outweigh the carrier line, skip this stage.
3. **Fine estimate at the symbol instants** (when stage 2 was rejected or absent): the M-th power of matched-filter symbols at the strongest sampling phase, or for differential formats the M-th power of consecutive-symbol products, `Δf = arg(±E[(zₙz*ₙ₋₁)^M]) · Rs/(2πM)`, which does not depend on the carrier phase (DBPSK ±Rs/4, DQPSK and π/4-DQPSK ±Rs/8). Its candidates repeat every Rs/M (every Rs/8 for QAM, whose false lines sit half a branch away); turning by one branch per symbol is a symmetry of PSK, QAM and π/4-DQPSK, so the candidate nearest the coarse estimate is taken. For APSK it is a symmetry of the outer ring only (30° per symbol for 16-APSK, 22.5° for 32-APSK): the inner ring's four points keep a coherent fourth power only on the right branch, `|E[(z/|z|)⁴ e^{−j8πΔf n/Rs}]|` over inner-ring symbols, which selects among the plausible candidates. Once the branch is settled, an agreeing stage-2 estimate replaces the fine one (it is more precise).

The receiver then derotates by the estimate anchored to absolute time, so the correction is continuous across blocks, and filters again when the fine stage moved it by more than 1 % of Rs.

### Phase, scale and timing start

QAM phase comes from a blind phase search: rotations over one 90° sector are scored by the mean squared distance to the nearest constellation point, jointly with the radius-matched gain candidates, and refined by a parabola. The fourth-power moment is data dependent for a repeating payload (it can point anywhere). APSK phase comes from its outer ring's M-th power with the branch chosen by decisions; PSK from the M-th power. A cold timing loop starts at the sampling phase with the most matched-filter energy, which is within a fraction of a sample of the symbol instant; from an arbitrary phase a decision-directed loop on dense QAM can take hundreds of symbols to converge, or lock falsely.

### Tracking across blocks

With a tracking state and absolute block positions, a block continues from the previous one: the symbol clock on its absolute grid (the first symbol instant after the previous block's last), the carrier phase predicted from the loop's final phase and frequency across the gap and refined on the new block's first symbols (blind phase search near the prediction for QAM, the inner ring's fourth power for APSK, M-th power otherwise, always on the branch nearest the prediction), the residual frequency updated from the phase drift over the gap, and a converged CMA equalizer's taps. A tracked block whose EVM exceeds max(2 × the previous block's + 1 %, 6 %) has lost lock and is acquired again from scratch. Modulation quality accumulates the most recent 2,048 tracked symbols of consecutive captures, so it measures a stable record even when each capture holds a few symbols; a fresh acquisition starts a fresh record.

Timing loops correct the sampling instant by `sps · clip(μ·e, ±¼)` samples per symbol, so the gain μ is in symbols per unit detector output and the dynamics do not depend on oversampling (a correction of μ·e samples would make the loop sps times slower at higher oversampling). The detector slopes for unit-RMS root-raised-cosine symbols, measured from their S-curves, are K ≈ 1.6–1.95 for Mueller & Müller and K ≈ 2.9·β for Gardner, independent of oversampling; a first-order loop converges as (1 − μK)ᵏ and settles in about 4/(μK) symbols (157 for the default μ = 0.015 with M&M).

Mueller & Müller timing is decision-directed, so it runs jointly with the carrier loop: each symbol is interpolated at the current timing estimate, derotated by the current carrier phase and sliced; the timing error `e = Re{d*ₖ₋₁·zₖ − d*ₖ·zₖ₋₁}` and the carrier error come from that one decision, and the sampling instant advances by `sps · (1 + clip(μ·e, ±¼))` with linear interpolation. Gardner timing is decision-free and runs before the carrier loop; it replaces Mueller & Müller for the differential formats (whose sign decisions are ambiguous on the axes) and OQPSK (whose staggered rails bias M&M). Differential detection removes the residual rotation of the products, estimated with the differential estimator above.

The headline EVM excludes the acquisition: the first `max(4/(μK) for timing, 4/(ζω_n) for the carrier loop, ⌈1/μ_CMA⌉ for the equalizer)` symbols of the active loops, capped at half the capture. The EVM over all symbols and the number of excluded symbols are reported alongside, and the excluded span is shaded in the EVM traces.

The channel experiment's fractional delay is a 41-tap Hamming-windowed sinc applied centred (error about −65 dB for content within 0.8·Nyquist). The channel is applied per capture, so the 20 samples nearest each capture edge, and the echo's first samples, see zeros beyond the edge.

## Generic frames

Frame blocks are pulse shaped with a centred (zero-delay) RRC so block boundaries stay exact; the first and last symbol of each block therefore see truncated-pulse ISI from the missing tails. The block demodulator supplies one transmit-filter delay of context around each linear block, which places symbol 0 where the causal-delay demodulator samples it.

## OFDM

A complex frequency-domain grid uses signed FFT indices. Default data/pilot placement is overridden by per-cell null, pilot, explicit-value or modulation/gain entries. A known mixed mapping is decoded by the corresponding per-cell constellation. Uniform BPSK, QPSK, 8-PSK, 16-QAM and 64-QAM are supported.

IFFT normalization is unitary. Prefix lengths cycle by symbol; optional WOLA uses cyclic suffixes and complementary sine-squared ramps within the shortest prefix. Absolute capture position includes an earlier symbol when needed to preserve overlap tails. Turning off Transmitter RMS normalization preserves exact designed sample amplitudes apart from the selected scalar gain.

The receiver knows frame/resource positions, discards an initial partial symbol, and uses complete symbols. CP correlation estimates fractional CFO modulo subcarrier spacing, approximately within ±Δf/2. The FFT window starts midway through the ISI-free part of each cyclic prefix (after any transmit windowing ramp), so a fractional or small timing error spreads none of the next symbol into it; advancing the window by *a* samples multiplies carrier *k* by `exp(−j2πka/N)`, which is removed exactly. Pilot LS estimates the channel per OFDM symbol; log-magnitude and unwrapped phase are interpolated linearly between pilots and extrapolated linearly beyond the outermost pilots, so a gain and a delay (linear phase) are reproduced exactly on every carrier. Pilots *D* carriers apart resolve delays up to about `N/(2D)` samples before the phase unwraps wrongly (2.3 samples for the default 64-point FFT with pilots every 14 carriers). Deep nulls, poor pilot spacing, delay exceeding the CP, large CFO and time variation can invalidate this simple estimator. Arbitrary unknown frame synchronization and MIMO are future work.

EVM is decision-directed. With pilot scale available, OFDM measures errors against the absolute known constellation scale. Error trajectories are exposed. Wrong decisions can still make EVM optimistic; it is not BER or standards compliance.

OFDM output FIR filters are causal. The source provides prehistory for convolution; arbitrary filters may violate cyclic-prefix assumptions. Custom overlap ramps use a complementary falling ramp but need not have favorable spectral properties.

## Radar codes

Phase codes follow Levanon & Mozeson, *Radar Signals* (2004), with group (frequency step) as the slow index: Frank φ = 2π·g·k/M; P1 φ = −(π/M)[M − (2g+1)][gM + k]; P2 φ = −(π/2M)(2k + 1 − M)(2g + 1 − M); P3 φ = πn²/N; P4 φ = πn²/N − πn; Zadoff–Chu with the odd/even-length definitions and gcd(u, N) = 1. (Flattening the P1 array sample-major instead produces a different sequence about 5 dB worse in peak sidelobe level.) Frank and P1 share their aperiodic autocorrelation magnitude; Barker codes have unit sidelobes; Costas arrays use the exponential Welch construction (order p−1 for the largest prime p ≤ requested + 1, minimum 4).

## Radar source and propagation

The simulator accepts mathematical/control-point/sample-table/Python programs, the complete legacy sequence or continuous design, a complete OFDM frame, or a recording window. Stored legacy, OFDM and recording designs repeat; mathematical programs can use absolute time for nonrepeating behavior. Delayed evaluations use the same source definition: programs are evaluated at the delayed time; a stored record is read on its own sample grid and moved by the remaining fraction of a sample with a 41-tap windowed sinc (about −65 dB error up to 0.8 Nyquist), as an ideal delay of the sampled waveform would be. A recording at a sample rate different from the simulation is linearly interpolated, which attenuates content near Nyquist (a warning is shown).

For target range R and receding radial velocity v, echo delay is `2R/c` and carrier Doppler is `−2v/λ`. Spatial phase is computed separately at every receive element from its position in wavelengths and the target direction. Target position includes configured velocity and acceleration across simulation time. Range migration and acceleration within one capture are neglected (stop-and-hop approximation).

Relative mode uses amplitude proportional to `sqrt(RCS) / R²`, referenced to 3 km. Radar-equation mode uses `Pt Gt Gr λ² σ / ((4π)³ R⁴)` power, with configured losses and extra echo gain. Analytic TX-array and TX/RX-element responses multiply the echo. Thermal noise uses `k T Fs F` at 290 K. Array weights are normalized; this is a system-level approximation rather than a calibrated hardware front end. Swerling-I draws one exponential power multiplier per target/capture.

The clutter control adds spatially common complex Gaussian samples. It is a noise-like clutter experiment, not a terrain or Doppler-spectrum clutter model. TX leakage is a coherent scaled copy of the transmitted samples.

## Arrays and estimation

The coordinate convention is X forward, Y lateral, Z up, with a front-hemisphere angle search. Arrays can be linear, planar or arbitrary element positions. Element patterns are isotropic, cosine, Gaussian or a tabulated azimuth cut. Complex tapers and steering influence the actual receive samples. Scanning sweeps azimuth sinusoidally through ±60°; the rate field controls its angular phase rate.

Conventional beamforming uses the configured steering/taper. MVDR estimates sample covariance, adds 1% trace-based diagonal loading and normalizes the steering response. It can suppress desired signals outside the chosen look direction. This is not a robust adaptive beamformer under all model mismatch.

Processing uses full-waveform delay/Doppler correlation over the displayed search interval (with the Doppler window tapering the capture, normalized to unit coherent gain: Hann sidelobes are −31 dB instead of the rectangular window's −13 dB), pulse-Doppler (fast-time matched filter per transmitted pulse, slow-time Doppler FFT) or FMCW (`TX · conj(RX)` dechirp, fast-time beat FFT, slow-time FFT; Doppler biases FMCW range). Signs follow positive receding velocity. Range sampling is `c/(2Fs)` at the processing rate. Periodic signals can have delay/velocity ambiguities and chirps can exhibit range–Doppler coupling. Oversampled search grids and zero padding do not create physical resolution.

The strongest measured range/Doppler cell is coherently extracted from the raw array channels and scanned with conventional steering vectors. Reported angles/XYZ come from this array response, not target truth. Linear arrays do not determine elevation uniquely. Multiple targets sharing a cell, grating lobes, low SNR, receiver-chain delay and ambiguous waveforms can yield incorrect or nonunique strongest-cell estimates.

Detection compares every cell of the range–Doppler power map with a multiple α of a noise estimate from `T` training cells on each side along range, outside `G` guard cells (the guard is widened to the compressed mainlobe). For square-law detected complex Gaussian noise the cells are exponential, and α is solved exactly for the requested per-cell false-alarm probability:

| Detector | Noise estimate | Pfa(α) |
| --- | --- | --- |
| CA (cell averaging) | mean of the N = 2T cells | (1 + α/N)^−N |
| SO (smallest of) | smaller of the two sides' means | 2 Σ_{k=0}^{T−1} C(T−1+k, k) (2 + α/T)^−(T+k) |
| GO (greatest of) | greater of the two sides' means | 2 (1 + α/T)^−T − Pfa_SO(α) |
| OS (ordered statistic) | k-th smallest of the N cells (k = rank·N) | Π_{i=0}^{k−1} (N − i)/(N − i + α) |

(Gandhi & Kassam 1988; Rohling 1983.) GO holds the false-alarm rate at clutter edges, SO resolves closely spaced targets, OS tolerates interfering targets in the training cells; Monte Carlo tests check each against its nominal rate. The default per-cell rate is 10⁻⁶: a range–Doppler map has a few thousand cells, so 10⁻³ would add several noise detections to every frame. Near the ends of the range axis the cells that exist are used (cell averaging when one side is missing). Neighbouring cells of a correlation or FFT output are not independent and real clutter is not exponential, so the per-cell rate is nominal for noise, not a calibrated end-to-end false-alarm probability. Cells above threshold are grouped into 8-connected clusters, each reported once at its strongest cell, so a target's mainlobe and sidelobes are one detection. Manual thresholds use the processor's normalized power in dB, not calibrated dBm.

## Classifier

The classifier is a heuristic observer of one emitter. Its pipeline: (1) noise, OFDM (high PAPR, many spectral components and a Gaussian envelope, E|x|⁴/(E|x|²)² ≈ 2, which single-carrier PSK/QAM — at most ~1.6 — does not have), pulsed radar and chirp rules; (2) sparse line spectra; (3) coarse tuning of the occupied band's power centroid to DC; (4) a constant-envelope test on the envelope CV in excess of the noise contribution √(1/(2(SNR+1))); (5) for constant-envelope signals, a symbol-synchronous CPFSK test — band-limit to the occupied band, take the symbol rate from the cyclostationary line of the discriminator's transition energy (stepping down from a harmonic only while the level count and clustering quality are preserved), integrate-and-dump the instantaneous frequency per symbol, accept 2 or 4 well-separated levels, call deviation ≈ Rs/4 MSK and neighbour-dependent symbol averages GMSK, and treat a strictly alternating level sequence as periodic analog modulation; (6) for linear signals, cyclostationary symbol rate, M-th-power residual CFO/phase removal per hypothesis, decision-directed EVM, and rejection of hypotheses that leave constellation points unused (nested constellations fit denser grids equally well).

The symbol period is searched fractionally (the cyclostationary estimate ± half a symbol of drift, plus the nearest integer period), with the sampling phase refined parabolically, so oversampling ratios that are not integers no longer misread QAM. Each constellation hypothesis is fitted after both a residual-CFO derotation (M-th power) and a phase-only derotation, keeping the better EVM; hypotheses that leave many constellation points unused are penalized. That assumes random data: a structured payload (repeated text) can use only a subset of a dense constellation (9 of 16 points of 16-QAM, a dozen of 64-QAM) and then reads as a sparser one — 8-PSK for 16- and 64-QAM carrying the default text. APSK is not among the hypotheses and reads as the nearest QAM.

Every rule records its feature value, operator, threshold and outcome; the trace is returned with the result and shown as the decision path. Bursty emitters (on 30–90 % of the capture) are classified from their longest burst. When several separate occupied bands carry significant power, the result says so and its score is lowered. An analysis band isolates one emitter: the band is mixed to DC, decimated while keeping at least eight times its width (so the symbol-rate and constellation estimators keep enough samples per symbol), and low-pass filtered at the band edges; its SNR is measured in-band against the noise floor of the original capture. Scores measure agreement with these rules, not calibrated probabilities. `tests/test_blind.py` holds the regression matrices.

## Recordings

Recordings are memory-mapped and decoded per capture window. The Recordings overview reads a fixed number of evenly spaced FFT segments, so it is a sampled view: short events between segments can be missed (the examined fraction is reported). Headerless formats are ranked by content heuristics (float plausibility, integer mean/spread and lag-1 correlation); they are suggestions, and the UI warns when a file is being read with a format its content contradicts. Float components that are non-finite or exceed 10¹⁸ in magnitude (wrong-format data) are zeroed and counted.

## Wideband operation

Limits are in samples, so a higher rate means a shorter capture rather than more memory. Every DSP path is written in samples/symbols and uses exact sample-index phase and float64 frequency axes, so an experiment at 2–5 GS/s reproduces the same experiment at kS/s–MS/s rates with frequencies scaled (`tests/test_wideband.py`). Live instruments process bounded capture windows. Real-time pacing advances with wall time, skips captures and bounds expensive products to a measured per-frame work budget; every-sample pacing consumes complete consecutive captures and can run slower than wall time. At low recording rates, overlapping windows can provide fresh samples for each display frame.

## What a live scope means

All radar live plots use the latest processed capture and expose its ID/time. Receive noise, moving targets, scan weights and time-dependent waveforms change the numerical data. Fixed waveform correlation or a fixed beam pattern may be invariant; they are recomputed/current reference properties, not artificially animated measurements. Pause freezes the simulation; Step acquires one new capture.
