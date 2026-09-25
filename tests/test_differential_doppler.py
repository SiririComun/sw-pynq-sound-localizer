"""
tests/test_differential_doppler.py: Strict Unit Verification Suite for Dual-Ended Differential Doppler Tracker.
"""

import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pytest

from pynq_localizer.kinematics import KinematicAnalytics, DifferentialDopplerTracker

def generate_synthetic_doppler_stereo_frame(
    v_mps: float,
    f0: float = 2609.73,
    amplitude_v: float = 0.100,
    fs: float = 50000.0,
    n_samples: int = 2048,
    temperature_c: float = 20.0,
    carrier_drift_hz: float = 0.0,
    snr_db: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Synthesizes dual-microphone audio for an air track glider moving at velocity v_mps.
    Mic 1 is at x=0 (left), Mic 2 is at x=L (right).
    Sign convention:
      - v > 0 => Moving toward Mic 2 (Mic 2 is Blue, Mic 1 is Red)
      - v < 0 => Moving toward Mic 1 (Mic 1 is Blue, Mic 2 is Red)
    """
    c = KinematicAnalytics.speed_of_sound(temperature_c)
    f0_actual = f0 + carrier_drift_hz

    # Observed frequencies at opposing ends
    f1 = f0_actual * (1.0 - (v_mps / c))
    f2 = f0_actual * (1.0 + (v_mps / c))

    t = np.arange(n_samples) / fs
    v_a0 = amplitude_v * np.cos(2.0 * np.pi * f1 * t)
    v_a1 = amplitude_v * np.cos(2.0 * np.pi * f2 * t)

    if snr_db is not None:
        noise_sigma = amplitude_v / (10.0 ** (snr_db / 20.0)) / np.sqrt(2.0)
        v_a0 += np.random.normal(0, noise_sigma, n_samples)
        v_a1 += np.random.normal(0, noise_sigma, n_samples)

    return v_a0, v_a1, f1, f2

class TestDifferentialDopplerEngine:

    def test_stationary_glider_zero_velocity(self):
        """Verify that a stationary glider produces exactly 0.00 m/s with f1 = f2 = f0."""
        v0, v1, f1_true, f2_true = generate_synthetic_doppler_stereo_frame(v_mps=0.0)
        tracker = DifferentialDopplerTracker(nominal_f0_hz=2609.73)
        res = tracker.process_stereo_frame(v0, v1, fs=50000.0)

        assert abs(res["velocity_mps"]) < 1e-4, f"Stationary glider velocity not zero: {res['velocity_mps']}"
        assert abs(res["delta_f_hz"]) < 0.05, f"Delta f not zero: {res['delta_f_hz']}"
        assert res["motion_state"] == "STATIONARY"
        assert res["status"] == "ACTIVE_VALID"

    def test_bidirectional_velocity_accuracy(self):
        """
        Verify velocity inversion accuracy across a wide speed sweep from -1.20 m/s to +1.20 m/s.
        Tolerance: error < 0.003 m/s (3 mm/s).
        """
        test_speeds_mps = [-1.20, -0.80, -0.45, -0.15, -0.05, 0.05, 0.15, 0.45, 0.80, 1.20]
        tracker = DifferentialDopplerTracker(nominal_f0_hz=2609.73)

        for v_true in test_speeds_mps:
            v0, v1, _, _ = generate_synthetic_doppler_stereo_frame(v_mps=v_true)
            res = tracker.process_stereo_frame(v0, v1, fs=50000.0)
            v_est = res["velocity_mps"]
            err_mps = abs(v_est - v_true)

            expected_state = "TOWARD_MIC2" if v_true > 0 else "TOWARD_MIC1"
            assert err_mps < 0.003, f"Speed error too high: True={v_true} m/s, Est={v_est} m/s, Err={err_mps*1000:.2f} mm/s"
            assert res["motion_state"] == expected_state

    def test_common_mode_thermal_drift_immunity(self):
        """
        Verify that a massive +25.0 Hz carrier thermal/battery drift produces zero velocity bias.
        """
        v_true = 0.500  # 50 cm/s
        # Simulate +25 Hz drift on the buzzer oscillator
        v0, v1, _, _ = generate_synthetic_doppler_stereo_frame(v_mps=v_true, carrier_drift_hz=25.0)

        tracker = DifferentialDopplerTracker(nominal_f0_hz=2609.73)
        res = tracker.process_stereo_frame(v0, v1, fs=50000.0)

        err_mps = abs(res["velocity_mps"] - v_true)
        print(f"\n[Thermal Drift Test] True v = {v_true} m/s | Est v = {res['velocity_mps']} m/s | Bias = {err_mps*1000:.4f} mm/s")
        assert err_mps < 0.003, f"Thermal drift corrupted velocity: {err_mps} m/s"
        assert abs(res["f0_common_hz"] - (2609.73 + 25.0)) < 0.20, "Common-mode drift tracker failed!"

    def test_minimum_detectable_velocity_threshold(self):
        """
        Verify sub-centimeter velocity sensitivity: detects a glider creeping at 4.0 mm/s (0.004 m/s).
        """
        v_slow = 0.004  # 4 mm/s
        v0, v1, _, _ = generate_synthetic_doppler_stereo_frame(v_mps=v_slow)

        tracker = DifferentialDopplerTracker(nominal_f0_hz=2609.73, velocity_deadband_mps=0.001)
        res = tracker.process_stereo_frame(v0, v1, fs=50000.0)

        assert abs(res["velocity_mps"] - v_slow) < 0.0015, f"Sub-centimeter tracking failed: {res['velocity_mps']}"
        assert res["motion_state"] == "TOWARD_MIC2"

    def test_pneumatic_blower_noise_robustness(self):
        """
        Verify tracking accuracy remains intact under realistic pneumatic air blower hiss (SNR = 18 dB).
        """
        np.random.seed(42)
        v_true = -0.350
        v0, v1, _, _ = generate_synthetic_doppler_stereo_frame(v_mps=v_true, snr_db=18.0)

        tracker = DifferentialDopplerTracker(nominal_f0_hz=2609.73)
        res = tracker.process_stereo_frame(v0, v1, fs=50000.0)

        err_mps = abs(res["velocity_mps"] - v_true)
        print(f"\n[Blower Noise Test] True v = {v_true} m/s | Est v = {res['velocity_mps']} m/s | Err = {err_mps*1000:.2f} mm/s")
        assert err_mps < 0.025, f"Noise error too high: {err_mps} m/s"
        assert res["motion_state"] == "TOWARD_MIC1"

    def test_glider_kinematic_analysis_drag_and_restitution(self):
        """
        Verify trajectory analysis: extracts viscous damping gamma and bumper restitution e.
        """
        # Synthesize a 3-second glider flight:
        # Segment 1: Coasting forward with viscous damping: v(t) = 0.60 * exp(-0.15 * t)
        # Impact at t = 1.5s: Glider reaches v_impact = 0.60 * exp(-0.15 * 1.5) ≈ 0.479 m/s
        # Rebound with restitution e = 0.90: v_after = -0.90 * v_impact ≈ -0.431 m/s
        t = np.linspace(0, 3.0, 300)
        v = np.zeros_like(t)

        mask1 = t < 1.5
        mask2 = t >= 1.5

        # Coasting segment
        v[mask1] = 0.60 * np.exp(-0.15 * t[mask1])

        # True velocity right before impact at t = 1.5s
        v_impact = 0.60 * np.exp(-0.15 * 1.5)

        # Rebound segment with e = 0.90
        v[mask2] = -v_impact * 0.90 * np.exp(-0.15 * (t[mask2] - 1.5))

        metrics = DifferentialDopplerTracker.analyze_glider_kinematics(time_sec=t, velocity_mps=v)

        assert metrics["total_collisions_detected"] == 1
        assert abs(metrics["mean_coefficient_of_restitution"] - 0.90) < 0.02
        assert abs(metrics["viscous_drag_gamma"] - 0.15) < 0.03
        assert metrics["status"] == "ANALYSIS_COMPLETE"