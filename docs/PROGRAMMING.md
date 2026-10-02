# Programming waveforms and processing — 0.7

## Expressions

Transmitter → Basic → CUSTOM and Radar → Waveform designer expose the same mathematical program. Expressions return one complex sample per requested time. They support arithmetic, comparisons, `&`/`|` masks, and these functions:

`sin cos tan exp sqrt abs angle real imag conj log log10 floor ceil sign tanh sinc minimum maximum where clip`

| Variable | Meaning |
| --- | --- |
| `t` | Time modulo the specified waveform period, in seconds |
| `time` | Absolute simulation time, in seconds |
| `T` | Specified period, in seconds |
| `n` | Sample index within the requested capture |
| `fs` | Requested sample rate in samples/second |
| `frame` | Deterministic capture ID |
| `j`, `pi` | Imaginary unit and π |
| Named parameters | Values from the program's numeric JSON dictionary |

A tone with continuous phase:

```python
exp(j*2*pi*25000*time)
```

A gated linear chirp, using numeric parameters `f0`, `slope` and `width`:

```python
where(t < width, exp(j*2*pi*(f0*t + slope*t*t/2)), 0)
```

A time-varying phase/frequency experiment:

```python
exp(j*(2*pi*20000*time + 2*sin(2*pi*30*time)))
```

A multicomponent signal:

```python
0.7*exp(j*2*pi*15000*time) + 0.3*exp(-j*2*pi*35000*time)
```

The interpreter evaluates array arithmetic rather than executing Python. Attribute access, imports, assignments and arbitrary function calls are unavailable. Expressions are bounded to 8,000 characters/512 AST nodes and finite sample values. Use a local Python extension for algorithms requiring loops, FFTs, state machines or additional libraries.

## Control points and I/Q tables

Control-point times are in microseconds and must increase. Frequency is integrated piecewise-linearly, while amplitude and extra phase interpolate. Samples outside the point interval have zero amplitude. The program repeats at its specified period; an inconsistent endpoint can create an intended or unintended discontinuity.

An I/Q table defines arbitrary finite complex samples. I and Q arrays must have equal lengths. Its own sample rate and interpolation determine playback; the table's duration controls its repetition. Example: I `[1,0,-1,0]`, Q `[0,1,0,-1]` is a four-sample complex rotation.

Radar preserves the program's sample amplitude. Transmitter's **Normalize RMS** toggle controls whether samples are RMS-normalized or simply multiplied by Output gain.

## OFDM resource grids

The FFT carrier coordinate is signed, with DC `0`, negative frequencies to the left and positive frequencies to the right. The spacing is `sample_rate / FFT_length`. An empty active-carrier list selects the guarded allocation; an explicit list can select separated subbands.

Per-cell overrides are applied after default data/pilot assignment. Example:

```json
[
  {"symbol":0,"carrier":-3,"kind":"pilot","i":1,"q":0},
  {"symbol":0,"carrier":2,"kind":"value","i":0.5,"q":-0.5},
  {"symbol":1,"carrier":4,"kind":"null"},
  {"symbol":2,"carrier":6,"kind":"data","modulation":"64qam","gain":0.7}
]
```

`symbol` is zero-based within the configured frame. The grid can define training symbols or arbitrary frequency-domain values without changing the synthesizer. Explicit known values can also support pilot channel estimation. Mixed data-cell constellations are sliced with their configured mapping and gain. Explicit values/pilots are reference cells, not payload bits.

A CP list cycles across successive OFDM symbols. The overlap window must fit in the shortest CP. WOLA extends each symbol with a cyclic suffix and complementary ramps. The transmitter computes its position from absolute sample offset; a frame larger than one capture is still fully transmitted. Receiver uses that known position, skips an initial partial symbol, and processes complete symbols. This is known-boundary reception, not blind frame synchronization.

## Local Python waveform modules

Create a `.py` file in the project's `extensions` folder and restart the server. The extension catalog appears in both custom waveform editors. These are local trusted Python modules loaded by your server process; they have the same access as that process.

```python
import numpy as np

def waveform(context):
    time = context['time']
    f = context['parameters'].get('frequency_hz', 25000)
    return np.exp(2j*np.pi*f*time)
```

The context contains `t`, `time`, `fs`, `frame`, `parameters`, and a deterministic NumPy `rng`. Return a one-dimensional finite complex array with exactly `len(context['time'])` samples. The function may use NumPy/SciPy or additional dependencies you install into `.venv`. It is called at delayed time coordinates as well as current time during radar propagation, so pure functions of time/parameters are strongly preferred for deterministic waveforms. A stateful mutable source must account for out-of-order delayed evaluations itself.

The included `polynomial_chirp.py` is an editable polynomial instantaneous-frequency law with amplitude gating. Its parameters are `f0`, `f1`, `f2`, `f3`, `width_s`, and `amplitude`.

## Local Python processing modules

A module can provide `process(samples, sample_rate, context)`:

```python
import numpy as np
from scipy.signal import lfilter

def process(samples, sample_rate, context):
    memory = context['state']
    taps = np.array([0.25, 0.5, 0.25])
    zi = memory.get('zi', np.zeros(len(taps)-1, complex))
    out, memory['zi'] = lfilter(taps, [1], samples, zi=zi)
    return out, sample_rate
```

Context contains `time_s`, `frame`, `parameters`, and mutable `state`. Return an array, or `(array, new_sample_rate)` for rate-changing algorithms. Output must be finite, one-dimensional and within the engine's size bound. The included `iir_notch.py` supplies a stateful notch filter with `frequency_hz` and `Q` parameters.

In **Receiver → Processing chain**, the WebSocket engine invokes the chain for each successive capture. Python state, FIR delay state and Costas phase/frequency persist. Probe-only changes preserve state; source, channel or processing changes reset it. **Reset state** increments an explicit reset epoch. Connection loss also restarts processing memory.

Radar's processing chain runs independently per simulated capture, including its Python state. Its output feeds waveform correlation, with reference resampling when the output sample rate changes. Algorithms that introduce uncompensated delay can shift measured range. The scope probe only chooses the intermediate array displayed; it does not alter the final processing output.

## Reproducible experiments

A source configuration plus capture ID identifies a deterministic communications capture. Radar additionally includes simulation time, the entire waveform design, scene, array and processor settings. Use Pause/Step and the same settings to reproduce a radar result. Setup controls and traces are session-local; persistent project save/load is future work.

## 0.7 OFDM shaping fields

`OFDMDesign.window_kind` is `raised-cosine`, `linear` or `custom`; `window_ramp` defines a custom rising overlap ramp. `output_filter` is `off`, `lowpass` or `custom`. Lowpass fields are `filter_cutoff_hz`, `filter_length` and `filter_window`. Custom taps are `filter_i` and optional same-length `filter_q`. The output filter is applied after CP/WOLA and before transmitter scaling/impairments. Preceding raw OFDM samples are rendered to preserve convolution across capture boundaries. It is one full-band output filter, not a separate subband filter bank or FBMC implementation.

Receiver `ofdm_timing_offset` skips a known number of delayed samples while preserving the known source frame position. Changing receiver FFT/resource parameters disables UI following. Processing chains that alter sample rate also alter the rate passed to demodulation; you must configure the demodulator accordingly. There is no automatic arbitrary-chain delay estimator.

## 0.7 radar processing object

`RadarExperiment.processing` takes `method` (`correlation`, `pulse-doppler`, `fmcw`), `pri_s`, `pulse_width_s`, `chirp_slope_hz_s`, `clutter` (`off`, `mean`, `two-pulse`, `three-pulse`), `range_window`, `doppler_window`, `min_range_m`, `detector` (`ca-cfar`, `manual`), and `threshold_db`. Windows are boxcar, Hann, Hamming or Blackman.

Coherent processing starts at the next known interval boundary within each capture. It requires four complete intervals before cancellation. MTI uses finite differences across intervals, dropping one or two initial rows. A separate actual transmitted reference is used for each pulse/chirp. Processing settings must match source timing; arbitrary timing remains available via correlation. FFT bin spacing and physical resolution are distinct, especially with zero padding.
