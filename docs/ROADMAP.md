# Roadmap

The programmable interfaces now permit numerical waveform and receiver extensions. These additional capabilities still require concrete implementations and validation:

- Continuous built-in framing/filter/timing/resampler state across every communications source and block; backpressure and long-duration resource profiling.
- Unknown-boundary OFDM synchronization, robust pilot layouts, MIMO/resource ports and reference-aligned BER/EVM.
- Persistent setup save/load, named experiments, undo/redo and repeatable parameter-sweep comparisons.
- Radar fractional-delay fidelity, wideband arrays, range migration, bistatic geometry, propagation/multipath and calibrated clutter.
- Multi-cell angle estimation, detection association, tracking and quantified false-alarm calibration.
- Electromagnetic antenna geometry/mutual-coupling or mesh scattering through an external solver integration.
- Statistically validated blind-modulation classes, protocol decoders and multi-emitter channelization.
- More OS/browser coverage and independent RF/DSP review.

No full-wave antenna solver, generic MATLAB compatibility or arbitrary hardware integration is implied by the current extension API.
