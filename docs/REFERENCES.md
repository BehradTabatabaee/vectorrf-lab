# References

The literature and product documentation behind VectorRF Lab's algorithms and workflows. A product listed here informed a design decision; it does not imply compatibility or feature parity. Everything in the lab is implemented independently.

## Signal processing

| Reference | Used for |
| --- | --- |
| F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform", *Proc. IEEE* 66(1), 1978 | Window ENBW, scalloping and sidelobe figures; the analyzer's RBW definition |
| P. D. Welch, "The use of fast Fourier transform for the estimation of power spectra", *IEEE Trans. Audio Electroacoust.* 15(2), 1967 | Averaged, overlapped periodograms |
| R. E. Crochiere, L. R. Rabiner, *Multirate Digital Signal Processing*, Prentice Hall, 1983 | Multistage decimation of the zoom FFT and polyphase resampling |
| A. J. Viterbi, A. M. Viterbi, "Nonlinear estimation of PSK-modulated carrier phase with application to burst digital transmission", *IEEE Trans. Inf. Theory* 29(4), 1983 | M-th power carrier phase and frequency estimation |
| T. Pfau, S. Hoffmann, R. Noé, "Hardware-efficient coherent digital receiver concept with feedforward carrier recovery for M-QAM constellations", *J. Lightwave Technol.* 27(8), 2009 | Blind phase search for QAM |
| K. H. Mueller, M. Müller, "Timing recovery in digital synchronous data receivers", *IEEE Trans. Commun.* 24(5), 1976 | Decision-directed timing recovery |
| F. M. Gardner, "A BPSK/QPSK timing-error detector for sampled receivers", *IEEE Trans. Commun.* 34(5), 1986 | Non-data-aided timing recovery |
| D. N. Godard, "Self-recovering equalization and carrier tracking in two-dimensional data communication systems", *IEEE Trans. Commun.* 28(11), 1980 | The constant-modulus equalizer |
| U. Mengali, A. N. D'Andrea, *Synchronization Techniques for Digital Receivers*, Plenum, 1997 | Loop analysis, differential frequency estimation, acquisition |
| W. A. Gardner, "Exploitation of spectral redundancy in cyclostationary signals", *IEEE Signal Processing Magazine* 8(2), 1991 | Symbol-rate lines used by the classifier |
| T. M. Schmidl, D. C. Cox, "Robust frequency and timing synchronization for OFDM", *IEEE Trans. Commun.* 45(12), 1997; J.-J. van de Beek et al., "ML estimation of time and frequency offset in OFDM systems", *IEEE Trans. Signal Process.* 45(7), 1997 | Cyclic-prefix frequency estimation |
| J. K. Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", *SC '11*, 2011 | Counter-based (Philox) noise streams indexed by absolute sample |
| [PySDR: synchronization](https://pysdr.org/content/sync) | Showing timing and carrier acquisition symbol by symbol |

## Radar

| Reference | Used for |
| --- | --- |
| N. Levanon, E. Mozeson, *Radar Signals*, Wiley, 2004 (ch. 6) and B. L. Lewis, F. F. Kretschmer, "A new class of polyphase pulse compression codes and techniques", *IEEE Trans. Aerosp. Electron. Syst.* 17(3), 1981 | Frank and P1–P4 codes and their ordering |
| J. P. Costas, "A study of a class of detection waveforms having nearly ideal range–Doppler ambiguity properties", *Proc. IEEE* 72(8), 1984; S. W. Golomb, H. Taylor, "Constructions and properties of Costas arrays", *Proc. IEEE* 72(9), 1984 | Costas frequency hopping and the Welch construction |
| M. A. Richards, *Fundamentals of Radar Signal Processing*, McGraw-Hill, 2014 | Pulse-Doppler processing, MTI, the radar equation, CFAR |
| P. P. Gandhi, S. A. Kassam, "Analysis of CFAR processors in nonhomogeneous background", *IEEE Trans. Aerosp. Electron. Syst.* 24(4), 1988 | Exact false-alarm probabilities of CA-, GO- and SO-CFAR |
| H. Rohling, "Radar CFAR thresholding in clutter and multiple target situations", *IEEE Trans. Aerosp. Electron. Syst.* 19(4), 1983 | Ordered-statistic CFAR |
| H. L. Van Trees, *Optimum Array Processing*, Wiley, 2002 | Conventional and MVDR beamforming |
| [MathWorks: waveform-level radar synthesis](https://www.mathworks.com/help/radar/ug/radar-design-part-II-from-radar-data-to-IQ-signal-synthesis.html), [building and processing a radar data cube](https://www.mathworks.com/company/technical-articles/building-and-processing-a-radar-data-cube.html), [joint radar/communication with PMCW and OFDM](https://www.mathworks.com/help/phased/ug/joint-radar-communication-using-pmcw-and-ofdm-waveforms.html) | Propagating waveform samples through targets and arrays before measurement; fast-time and slow-time processing; OFDM as a radar waveform |
| [RadarSimX: arbitrary waveform design](https://radarsimx.com/2021/05/10/arbitrary-waveform/) | Frequency-time control points alongside mathematical and table waveforms |
| [Rohde & Schwarz Pulse Sequencer](https://www.rohde-schwarz.com/products/test-and-measurement/signal-generator-software/rs-pulse-sequencer_63493-374528.html) | Structured pulse sequences, separating waveform design from target experiments |

## Files and formats

| Reference | Used for |
| --- | --- |
| [SigMF specification](https://github.com/sigmf/SigMF/blob/main/sigmf-spec.md) | Datatype names (`cf32_le`, `ci16_le`, `cu8`, …), global and capture metadata, the `.sigmf` archive |
| EBU Tech 3306, *MBWF / RF64* | WAV files beyond 4 GiB (`ds64` chunk) |
| SpectraVue / SDR# `auxi` WAV chunk | Centre frequency stored in I/Q WAV recordings |

## Instruments and workflows

| Source | Design takeaway |
| --- | --- |
| [Signal Hound VSG](https://www.signalhound.com/sigdownloads/VSG60/docs/html/index.html) and [Spike](https://signalhound.com/sigdownloads/Spike/docs/html/index.html) documentation | Explicit source, modulation and impairment setup; distinct measurement modes, trace behaviour and acquisition controls |
| [Keysight 89600 VSA: digital demodulation troubleshooting](https://helpfiles.keysight.com/csg/89600B/Webhelp/Subsystems/digdemod/content/digdemod_troubleshooting.htm) | Demodulation quality depends on the format, filters and synchronization; quality displays state their assumptions |
| [SDRangel spectrum documentation](https://github.com/f4exb/sdrangel/blob/master/sdrgui/gui/spectrum.md) | Spectrum and waterfall controls, independent traces |
| [RadioInspector](https://www.radioinspector.com/) and [Wavecom W-SPECTRA](https://www.wavecom.ch/content/ext/MonitoringSystemOnlineHelp/worddocuments/wspectra.htm) | Classification and detailed inspection as separate stages, with the evidence shown |
| [GNU Radio flowgraphs](https://wiki.gnuradio.org/index.php/Handling_Flowgraphs) and [embedded Python blocks](https://wiki.gnuradio.org/index.php/Embedded_Python_Block) | Explicit processing order, intermediate probes and custom Python blocks |
| [MathWorks: OFDM modulator](https://www.mathworks.com/help/comm/ref/comm.ofdmmodulator-system-object.html), [custom resource grids](https://www.mathworks.com/help/wireless-hdl/ug/ofdm-custom-resource-grid.html), [F-OFDM](https://www.mathworks.com/help/comm/ug/f-ofdm-vs-ofdm-modulation.html), [OFDM transmitter and receiver](https://www.mathworks.com/help/comm/ug/ofdm-transmitter-and-receiver.html) | FFT, guard, DC, pilot, prefix and windowing controls; an editable resource grid; filtering after OFDM formation |
