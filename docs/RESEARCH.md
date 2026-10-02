# Research and design references — 0.7

These primary product/documentation sources informed workflow decisions. The release implements the features documented in its README and DSP notes; a reference here does not imply compatibility or feature parity.

| Source | Design takeaway used in this release |
| --- | --- |
| [Signal Hound VSG documentation](https://www.signalhound.com/sigdownloads/VSG60/docs/html/index.html) | A generator needs explicit source, modulation and impairment setup. This informed the plot-free Transmitter editor and exclusive payload source. |
| [Signal Hound Spike documentation](https://signalhound.com/sigdownloads/Spike/docs/html/index.html) | Distinct measurement modes, trace behavior and acquisition controls informed Spectrum sweep, zero span and quality pages. |
| [Keysight 89600 digital-demod troubleshooting](https://helpfiles.keysight.com/csg/89600B/Webhelp/Subsystems/digdemod/content/digdemod_troubleshooting.htm) | Demodulation depends on correct format, filtering and synchronization. Quality displays state their known-format and decision-directed assumptions. |
| [SDRangel spectrum documentation](https://github.com/f4exb/sdrangel/blob/master/sdrgui/gui/spectrum.md) | Spectrum/waterfall controls informed display range, independent trace behavior and the bounded waterfall arrangement. |
| [RadioInspector](https://www.radioinspector.com/) | Monitoring benefits from observation history and supporting evidence. Classifier gained history, thresholding and feature views. |
| [Wavecom W-SPECTRA documentation](https://www.wavecom.ch/content/ext/MonitoringSystemOnlineHelp/worddocuments/wspectra.htm) | Classification and detailed inspection are separate stages. Candidate evidence is visible and no protocol decoding is implied by a modulation label. |
| [Rohde & Schwarz Pulse Sequencer](https://www.rohde-schwarz.com/nl/products/test-and-measurement/signal-generator-software/rs-pulse-sequencer_63493-374528.html) | Structured pulse sequencing and scenario design informed progressive radar controls and separation of waveform design from target-return experiments. |
| [PySDR synchronization](https://pysdr.org/content/sync) | Intermediate timing/carrier state is useful for understanding acquisition. Receiver retains actual symbol-by-symbol recovery replay. |

The principal UI choice is progressive disclosure: give each instrument a small set of task pages, keep common controls visible, and reveal specialized parameters in their relevant context. The existing dark instrument palette is preserved, with larger workspace tabs and consistent plot-header/body spacing.


## Programmable-studio research for 0.6

| Primary source | Implementation decision |
| --- | --- |
| [MathWorks comm.OFDMModulator](https://www.mathworks.com/help/comm/ref/comm.ofdmmodulator-system-object.html) | Expose FFT/guard/DC/pilot/CP/window controls and a frequency-domain resource grid. |
| [MathWorks custom OFDM resource grid](https://www.mathworks.com/help/wireless-hdl/ug/ofdm-custom-resource-grid.html) | Treat carrier/symbol cells as editable construction data rather than a fixed QPSK preset. |
| [MathWorks waveform-level radar synthesis](https://www.mathworks.com/help/radar/ug/radar-design-part-II-from-radar-data-to-IQ-signal-synthesis.html) | Propagate waveform samples through target/array models before measurement and angle estimation. |
| [RadarSimX arbitrary waveform design](https://radarsimx.com/2021/05/10/arbitrary-waveform/) | Provide frequency-time control points alongside mathematical and sample-table designs. |
| [GNU Radio flowgraphs](https://wiki.gnuradio.org/index.php/Handling_Flowgraphs) and [Python blocks](https://wiki.gnuradio.org/index.php/Embedded_Python_Block) | Make processing order, intermediate probes and custom Python interfaces explicit. |

These references informed the architecture; this project does not reproduce the full feature set or model fidelity of MATLAB/Simulink, GNU Radio or RadarSimPy.

## 0.7 references

- [MathWorks joint radar/communication using PMCW and OFDM](https://www.mathworks.com/help/phased/ug/joint-radar-communication-using-pmcw-and-ofdm-waveforms.html): supports keeping OFDM as an explicitly advanced sensing option alongside pulse/CW construction.
- [MathWorks F-OFDM vs OFDM](https://www.mathworks.com/help/comm/ug/f-ofdm-vs-ofdm-modulation.html): time-domain filtering after OFDM formation and the need to assess spectrum alongside demodulation quality.
- [MathWorks OFDM transmitter and receiver](https://www.mathworks.com/help/comm/ug/ofdm-transmitter-and-receiver.html): general receive filtering/frequency correction, with format-specific OFDM demodulation.
- [MathWorks radar data cube](https://www.mathworks.com/company/technical-articles/building-and-processing-a-radar-data-cube.html): explicit fast-time pulse compression and slow-time Doppler processing.

Implementation here is independently coded and bounded; product examples are design references, not claims of feature parity.
