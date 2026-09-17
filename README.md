# Real-Time Acoustic Kinematics, Doppler Tracking & Sound Localizer on PYNQ-Z2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hardware Overlay](https://img.shields.io/badge/Hardware-hw--xadc--dma--overlays%20v1.5.1-orange.svg)](https://github.com/SiririComun/hw-xadc-dma-overlays)
[![Board Support](https://img.shields.io/badge/Board-PYNQ--Z2-green.svg)](https://tul.com.tw/ProductsPYNQ-Z2.html)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

A high-performance FPGA-accelerated acoustic processing platform for the **PYNQ-Z2 board (`xc7z020clg400-1`)**. Provides **true simultaneous dual-ADC parallel sampling ($0.00\,\mu\text{s}$ inter-channel skew)**, **sub-Hertz fundamental pitch tracking ($20\,\text{Hz} - 20\,\text{kHz}$)**, **real-time single-channel metric distance inversion ($r = k(f_0)/A_{\text{true}}$)**, **flagship 3-row rolling telemetry curves**, and continuous multi-second flight recording for Doppler kinematics.

---

## 🏛 System Architecture

The package automatically pulls its pre-compiled hardware bitstream (`v1.5.1`) and metadata from GitHub Releases into local cache and encapsulates dual DMA receivers, XADC parallel sequencers, hardware decimators, and telemetry engines into a clean Python API:

```
 [ MAX4466 Mic 1 ] ─────────────────────────> [ PYNQ-Z2 Pin A0 (Vaux1) ]
 [ MAX4466 Mic 2 ] ─────────────────────────> [ PYNQ-Z2 Pin A1 (Vaux9) ]
                                                            │
                                             (XADC Dual Continuous Sequencer)
                                             (1 MSPS Interleaved Stream, 0.00 µs Skew)
                                                            ▼
                                                  [ axis_decimator IP ]
                                                (FPGA Anti-Aliasing M=10)
                                                            │ (50 kSPS Decimated Stream)
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
                                      ├── .capture_spectral_frame() (Lock-Step Dual-DMA Frame)
                                      ├── .capture_quadruple()      (BFP-Immune (f, A, φ, t))
                                      ├── .record_continuous()      (Multi-Second Flight Logger)
                                      ├── .play_audio()             (Jupyter Audio Playback)
                                      └── .kinematics_dashboard()   (3-Row Rolling Live GUI)
```

---

## 🎛 Real-Time 3-Row Telemetry Dashboard

The **`KinematicsDashboard`** uses a decoupled two-thread architecture ($100\,\text{Hz}$ background DSP worker + $30\,\text{FPS}$ Plotly rendering) to display real-time physical telemetry without browser lag:

* **Row 1 (Amplitude vs. Time):** Physical in-band amplitude envelope $A_{\text{true}}(t)$ in physical **mV**, extracted via single-bin coherent Fourier projection. **100% immune to FPGA Block Floating Point (BFP) bit-shift scaling**.
* **Row 2 (Frequency vs. Time):** Sub-Hertz fundamental pitch trajectory $f_0(t)$ ($20\,\text{Hz} - 20\,\text{kHz}$) using exact DFT sinc peak ratio interpolation ($\pm 0.01\,\text{Hz}$ precision) with moving-median reflection rejection and active noise squelch gating.
* **Row 3 (Distance vs. Time):** Real-time inverted metric distance $r(t) = \frac{k(f_0)}{A_{\text{true}}(t)}$ in **cm** with dynamic confidence error bands ($\pm \delta r(t)$).
* **Live Status Readouts:** Real-time frequency and distance tracking directly in the header bar (`A0: 35.2mV (1000.5Hz → 45.2cm)`).
* **3-Tab Synchronized View:** Dedicated **Mic 1 (A0)**, **Mic 2 (A1)**, and **Dual Comparison Overlay**.
* **Clean CSV Export:** Exports synchronized time, amplitude, pitch, distance, and uncertainty columns without `NaN` values directly to disk.

---

## 📐 Acoustic Calibration Protocol & Distance Inversion

Spherical acoustic wave propagation dictates:

$$V_{\text{RMS}}(r_i,\, f_j) = k(f_j) \cdot \left(\frac{1}{r_i}\right) + c_{\text{room}}(f_j)$$

* **`AcousticCalibrationProtocol`:** Ingests multi-sample ($N=30$) observations across an acoustic grid (e.g., 14 distance stations $\times$ 26 carrier frequencies), applies **Dynamic Boundary Pruning** using log-log power-law penalty scoring ($\frac{d\ln V}{d\ln r} \approx -1.0$) to reject near-field saturation clipping and far-field echo floors, and solves **Weighted Least Squares (WLS)** regressions ($w_i = 1/\sigma_i^2$).
* **`AcousticProfile`:** Portable JSON profile artifact storing continuous interpolated $k(f)$, measurement uncertainties $\delta k(f)$, certified operating bounds $[r_{\text{min}}(f),\, r_{\text{max}}(f)]$, and environmental metadata.
* **`DistanceEstimator`:** Real-time distance solver with dynamic uncertainty propagation:

$$\delta r(t) = r \cdot \sqrt{\left(\frac{\delta k}{k}\right)^2 + \left(\frac{\delta A}{A}\right)^2}$$

---

## 🔌 Hardware Setup & Physical Pin Constraints

Connect two analog electret microphones (such as Adafruit **MAX4466** or MAX9814) to the PYNQ-Z2 Arduino Header **`J1`**:

| Microphone Pin | PYNQ-Z2 Connection | Header Location | Description |
| :--- | :--- | :--- | :--- |
| **`VCC`** (Both Mics) | **`3.3V`** | Power Header | Clean analog supply voltage |
| **`GND`** (Both Mics) | **`GND`** | Power Header | Common system analog ground |
| **`OUT` (Mic 1)** | **Header `J1` Pin A0** | Pin 6 (Bottom) | Channel 1 Analog Input (`Vaux1`, pins `E17`/`D18`) |
| **`OUT` (Mic 2)** | **Header `J1` Pin A1** | Pin 5 (2nd from Bottom) | Channel 2 Analog Input (`Vaux9`, pins `E18`/`E19`) |

---

## 🚀 Installation & Getting Started

### 1. Install from GitHub
```bash
pip install git+https://github.com/SiririComun/sw-pynq-sound-localizer.git
```

### 2. Copy Example Notebooks to Jupyter Workspace
```bash
pynq-localizer-get-notebooks
```

---

## 💻 Python API Usage

### 1. Launch the Live 3-Row Telemetry Dashboard
```python
from pynq_localizer import MicrophoneArrayOverlay

# Auto-downloads and loads the pinned v1.5.1 bitstream
ol = MicrophoneArrayOverlay()

# Launch the live interactive 3-row dashboard (auto-loads calibrated profile if present)
app = ol.kinematics_dashboard()
```

### 2. Direct Clean Telemetry Handoff in Python
```python
# Extract clean non-NaN telemetry arrays directly from the dashboard
t_sec, amp_v, freq_hz, dist_cm, disterr_cm = app.get_clean_data(channel=1, return_distance=True)

print(f"Captured {len(t_sec)} clean motion points!")
print(f"Observed Doppler span : {freq_hz.min():.1f} Hz -> {freq_hz.max():.1f} Hz")
print(f"Distance trajectory   : {dist_cm.min():.1f} cm -> {dist_cm.max():.1f} cm (±{disterr_cm.mean():.1f} cm)")
```

### 3. Capture Single-Shot BFP-Immune Spectral Quadruple
```python
# Returns (f0, A_true, phase, timestamp) with zero BFP scaling error
data = ol.capture_quadruple(source="A0", f_min=100.0, f_max=10000.0)
q = data["quadruple"]

print(f"Pitch (f0)  : {q['frequency_hz']:.2f} Hz")
print(f"Amplitude   : {q['amplitude_v']*1000:.2f} mV RMS (Coherent In-Band)")
print(f"Phase (φ)   : {q['phase_deg']:.1f}°")
print(f"Timestamp   : {q['timestamp_sec']:.6f} s")
```

### 4. Continuous Multi-Second Flight Recording
```python
# Record 4.0 seconds of continuous 50 kSPS dual-channel flight data
t_axis, v_mic1, v_mic2 = ol.record_continuous(duration_sec=4.0)

# Listen to captured audio directly in Jupyter
ol.play_audio(channel=1, custom_data=v_mic1)
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.