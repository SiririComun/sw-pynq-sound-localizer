# Real-Time Acoustic Kinematics, Doppler Tracking & Sound Localizer on PYNQ-Z2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hardware Overlay](https://img.shields.io/badge/Hardware-hw--xadc--dma--overlays%20v1.5.3-orange.svg)](https://github.com/SiririComun/hw-xadc-dma-overlays)
[![Package Version](https://img.shields.io/badge/Version-v1.3.1-blue.svg)](https://github.com/SiririComun/sw-pynq-sound-localizer/releases/tag/v1.3.1)
[![Board Support](https://img.shields.io/badge/Board-PYNQ--Z2-green.svg)](https://tul.com.tw/ProductsPYNQ-Z2.html)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

An FPGA-accelerated acoustic processing platform for the **PYNQ-Z2 board (`xc7z020clg400-1`)**. Provides **true simultaneous dual-ADC parallel sampling ($0.00\,\mu\text{s}$ inter-channel skew)**, **continuous Hilbert phase demodulation**, **real-time metric distance inversion ($r = k_A / A$ and $r = \sqrt{k_E / E}$)**, **drift-free differential Doppler velocimetry**, and **direction of arrival (AoA) bearing localization**.

---

## 🏛 System Architecture

The package automatically pulls its pre-compiled hardware bitstream (`v1.5.3`) and metadata from GitHub Releases into local cache and encapsulates dual DMA receivers, XADC parallel sequencers, hardware decimators, pulse generators, and direct-energy telemetry engines into a clean Python API:

```
 [ MAX4466 Mic 1 ] ─────────────────────────> [ PYNQ-Z2 Pin A0 (Vaux1) ]
 [ MAX4466 Mic 2 ] ─────────────────────────> [ PYNQ-Z2 Pin A1 (Vaux9) ]
 [ 2N2222A Buzzer ] <───────────────────────── [ PYNQ-Z2 Pin AR2 (Pin U13) ]
                                                            │
                                             (XADC Dual Continuous Sequencer)
                                             (1 MSPS Interleaved Stream, 0.00 µs Skew)
                                                            ▼
                                                  [ axis_decimator IP ]
                                           (Programmable M = 1, 10, 20, 50)
                                                            │ (50 kSPS / 500 kSPS)
                              ┌─────────────────────────────┴─────────────────────────────┐
                              ▼                                                           ▼
                   [ AXI DMA 0 (Time Stream) ]                                 [ FPGA FFT Core + CORDIC ]
                              │ (Raw 12-bit ADC Voltages)                                 │ (32-bit Phase/Magnitude)
                              ▼                                                           ▼
                      [ DDR Time Buffer ]                                         [ AXI DMA 1 (Polar FFT) ]
                              │                                                           │
                              └─────────────────────────────┬─────────────────────────────┘
                                                            ▼
                                               [ MicrophoneArrayOverlay ]
                                      ├── .capture_spectral_frame()    (Lock-Step Dual-DMA Frame)
                                      ├── .capture_pulsed_toa_frame()  (Direct-Energy & ToA Range)
                                      ├── .record_differential_flight()(1D Air-Track Doppler Tracker)
                                      ├── .play_audio()                (Jupyter Audio Playback)
                                      └── .kinematics_dashboard()      (4-Tab Rolling Live GUI)
```

---

## 🔬 Metrological Capabilities & Validated Experiments

### 1. 1D Air-Track Multi-Mass Differential Doppler Kinematics (`exp03`)
Opposing microphones at $x = 0$ (Mic 1) and $x = L$ (Mic 2) eliminate buzzer oscillator thermal and battery drift through exact common-mode cancellation:
$$v_{\text{diff}}(t) = c(T) \cdot \left(\frac{f_2(t) - f_1(t)}{f_1(t) + f_2(t)}\right), \qquad f_{0,\text{common}}(t) = \frac{f_1(t) + f_2(t)}{2}$$

* **Newton's Second Law Verification:** A 3-mass campaign ($25\,\text{g}, 50\,\text{g}, 100\,\text{g}$, $N=3$ trials each) across an air track confirmed $a = \frac{m}{M+m}g$, experimentally extracting **$g_{\text{exp}} = 9.37\,\text{m/s}^2$ ($4.2\%$ error** against local $g = 9.78\,\text{m/s}^2$).
* **Elastic Bumper Turnaround Dynamics:** Harmonic shock-absorber contact analysis extracts bumper stiffness ($k_{\text{bumper}} \approx 2.2\,\text{N/m}$) and restitution ($e$).

### 2. Quasi-Anechoic Direct-Path Calibration & Ranging (`exp01`)
Acoustic pulse gating ($K = 3$ carrier periods $\approx 1.15\,\text{ms}$) freezes energy accumulation before room reflections arrive, reducing room reverberation to zero ($c_{\text{room}} \to 0$):
* **Pressure Inversion:** $A(r) = k_A \cdot \frac{1}{r} \implies r(t) = \frac{k_A}{A(t)}$ ($R^2 \ge 0.997$)
* **Intensity Inversion:** $E(r) = k_E \cdot \frac{1}{r^2} \implies r(t) = \sqrt{\frac{k_E}{E(t)}}$ ($R^2 \ge 0.998$)

### 3. 3-Method Direction of Arrival (AoA) Benchmark (`exp02`)
Comprehensive protractor characterization ($-60^\circ \text{ to } +60^\circ$ at $r = 30.0\,\text{cm}$, baseline $d = 5.0\,\text{cm}$):
* **Direct Autodyne Phase ($\theta_{\text{phase}}$):** Coherent single-bin Fourier projection achieves **$\text{MAE} \approx 4.02^\circ$** across the entire sector and $< 1.0^\circ$ at broadside.
* **TDOA Leading-Edge Threshold ($\theta_{\text{TDOA}}$):** Sub-sample zero-crossing interpolation achieves **$\text{MAE} \approx 7.86^\circ$**.
* **Gain-Normalized Energy Monopulse ($\theta_{\text{energy}}$):** Station median filtering eliminates multipath spikes, dropping error to **$\mathbf{18.46^\circ}$**.

---

## 🔌 Hardware Setup & Physical Pin Constraints

Connect two analog electret microphones (MAX4466) and the active buzzer driver to the PYNQ-Z2 Arduino Headers (`J1` and Digital):

```
 PYNQ-Z2 BOARD
 ┌────────────────────────────────────────────────────────────────────────┐
 │  [Power Header]                   [Analog Header]     [Digital Header] │
 │   • 3.3V ──────────────┐           • A0 ───────────┐   • Pin 2 (AR2) ─┐│
 │   • GND Pin 1 (Clean) ─┼───┐       • A1 ─────────┐ │                  ││
 │   • GND Pin 2 (Noisy) ─┼─┐ │                     │ │                  ││
 └────────────────────────┼─┼─┼─────────────────────┼─┼──────────────────┼┘
                          │ │ │                     │ │                  │
 ═════════════════════════╪═╪═╪═════════════════════╪═╪══════════════════╪════
 BRANCH 1: SENSITIVE ANALOG │                     │ │                  │
 • Mic 1 & 2 VCC ─────────┘ │                     │ │                  │
 • Mic 1 & 2 GND ───────────┘                     │ │                  │
 • Mic 1 OUT (Vaux1 / E17-D18) ───────────────────┼─┘                  │
 • Mic 2 OUT (Vaux9 / E18-E19) ───────────────────┘                    │
 ════════════════════════════════════════════════════════════════════════╪════
 BRANCH 2: NOISY BUZZER ACTUATOR                                       │
 • External 5V (+) ──► Buzzer (+)                                     │
 • Buzzer (-) ───────► Transistor Collector (Pin 3)                    │
 • AR2 (Pin U13) ────► 1kΩ Resistor ──► Transistor Base (Pin 2) ───────┘
 • Transistor Emitter (Pin 1) ──► External 5V GND (-) AND PYNQ GND Pin 2
```

| Signal Port | PYNQ-Z2 Connection | Pin Location | Description |
| :--- | :--- | :--- | :--- |
| **`VCC`** (Both Mics) | **`3.3V`** | Power Header | Clean analog supply voltage |
| **`GND`** (Both Mics) | **`GND Pin 1`** | Power Header | Common system analog ground |
| **`OUT` (Mic 1)** | **Header `J1` Pin A0** | `E17` / `D18` | Channel 1 Analog Input (`Vaux1`) |
| **`OUT` (Mic 2)** | **Header `J1` Pin A1** | `E18` / `E19` | Channel 2 Analog Input (`Vaux9`, $0.00\,\mu\text{s}$ skew) |
| **`buzzer_pulse_out`** | **Digital Pin AR2** | `U13` (LVCMOS33) | Hardware Pulse Trigger Output to Transistor Base |

---

## 🚀 Installation & Getting Started

### 1. Install Package
```bash
pip install git+https://github.com/SiririComun/sw-pynq-sound-localizer.git
```

### 2. Deploy Official Notebooks to Jupyter Workspace
```bash
pynq-localizer-notebooks
```

This deploys:
* **Interactive Labs:** `01_realtime_kinematics_telemetry.ipynb` to `06_planar_tdoa_sound_localizer.ipynb`
* **Experimental Harvesters:** `experiments/exp01_distance_energy_characterization.ipynb` to `exp03_air_track_kinematics_harvesting.ipynb`
* **Persistent Data Storage:** `experiments/data/` (preserves all generated `.xlsx` and `.csv` files)

---

## 💻 Python API Usage

### 1. 1D Air-Track Continuous Doppler Recording
```python
from pynq_localizer import MicrophoneArrayOverlay

ol = MicrophoneArrayOverlay()

# Record 6.0 seconds of dual-channel flight data along a 1.80 m track
flight = ol.record_differential_flight(
    duration_sec=6.0,
    track_length_m=1.80,
    f0_nominal=2580.0,
    temperature_c=20.0
)

print(f"Max Forward Velocity: {flight['velocity_cmps'].max():.1f} cm/s")
print(f"Estimated Restitution: e = {flight['kinematics_summary']['mean_coefficient_of_restitution']:.3f}")
```

### 2. Quasi-Anechoic Direct-Pulse Ranging
```python
# Fire a 10 ms pulse on AR2, capture direct wave, and invert distance
res = ol.capture_pulsed_toa_frame(
    pulse_width_ms=10.0,
    f_target=2580.0,
    temperature_c=20.0
)

print(f"ToA Radial Distance : {res['distance_cm']:.1f} cm")
print(f"Direct Energy Inversion: {res['dist_energy_m1_cm']:.1f} cm")
print(f"Incident Bearing    : θ = {res['theta_phase_deg']:+.1f}°")
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.