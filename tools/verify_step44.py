from unittest.mock import MagicMock
import numpy as np
from pynq_localizer.array import MicrophoneArrayOverlay
from pynq_localizer.kinematics import KinematicAnalytics

# 1. Instantiate overlay mock
ol = MicrophoneArrayOverlay.__new__(MicrophoneArrayOverlay)
ol.fs_per_ch = 50000.0
ol.trigger = MagicMock()
ol.axi_dma_0 = MagicMock()
ol.axi_dma_0.recvchannel.idle = True
ol.axi_timer_0 = MagicMock()
ol.axi_timer_0.mmio.read.return_value = 100000000

fs = 50000.0
c = KinematicAnalytics.speed_of_sound(20.0)  # 343.2145 m/s
f0 = 2609.73
d = 0.05

# 2. Target physical parameters: r = 60.0 cm, theta = +25.0 deg
r_true = 0.600
theta_true = 25.0
theta_rad = np.radians(theta_true)

t_flight = r_true / c
t_start0 = t_flight
t_start1 = t_flight + (d * np.sin(theta_rad) / c)

tau_rise = 0.001140 / np.log(2.0)
n_samples = 3000
t_axis = np.arange(n_samples) / fs

# Create exponential step burst envelopes
env0 = np.where(t_axis >= t_start0, 1.0 - np.exp(-(t_axis - t_start0) / tau_rise), 0.0)
env1 = np.where(t_axis >= t_start1, 1.0 - np.exp(-(t_axis - t_start1) / tau_rise), 0.0)

v0 = 0.150 * env0 * np.cos(2.0 * np.pi * f0 * t_axis)
v1 = 0.150 * env1 * np.cos(2.0 * np.pi * f0 * t_axis)

# 3. Simulate real XADC 12-bit unipolar sampling (Biased at 1.65V DC)
v_dc = 1.65
code0 = np.clip((v_dc + v0) * (4095.0 / 3.3), 0, 4095).astype(np.uint16)
code1 = np.clip((v_dc + v1) * (4095.0 / 3.3), 0, 4095).astype(np.uint16)

raw_interleaved = np.empty(n_samples * 2, dtype=np.uint16)
raw_interleaved[0::2] = code0 << 4
raw_interleaved[1::2] = code1 << 4

ol._buf_time = raw_interleaved
ol._buf_fft = None

# 4. Measure system offset at r = 0 cm (Buzzer rise + Bandpass filter lag)
v_cal = 0.150 * (1.0 - np.exp(-t_axis / tau_rise)) * np.cos(2.0 * np.pi * f0 * t_axis)
cal_offset_ms = KinematicAnalytics.detect_pulse_arrival_time(v_cal, fs, f0)["t_arrival_sec"] * 1000.0

# 5. Execute capture_pulsed_toa_frame
res = ol.capture_pulsed_toa_frame(
    pulse_width_ms=5.0,
    mic_distance_m=d,
    calibrated_offset_ms=cal_offset_ms
)

print(f"Target Distance : {r_true*100:.1f} cm | Estimated Distance : {res['distance_cm']:.2f} cm")
print(f"Target Bearing  : {theta_true:+.1f}° | Estimated TDOA Angle: {res['theta_tdoa_deg']:+.2f}°")
print(f"Flight Time     : {res['t_flight_sec']*1000:.4f} ms")
print(f"Status          : {res['status']}")

assert abs(res['distance_m'] - r_true) < 0.010, f"Distance error: {res['distance_m']}"
assert abs(res['theta_tdoa_deg'] - theta_true) < 0.50, f"TDOA angle error: {res['theta_tdoa_deg']}"
assert res['status'] == "ACTIVE_VALID"

print("🎉 Step 4.4 PASSED: MicrophoneArrayOverlay.capture_pulsed_toa_frame() verified on PC")
