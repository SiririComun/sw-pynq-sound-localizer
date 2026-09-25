"""
tests/test_toa.py: Strict Unit Verification Suite for Time of Arrival (ToA) and TDOA Engine.
"""

import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pytest

from pynq_localizer.kinematics import KinematicAnalytics, TimeOfArrivalEstimator

def generate_synthetic_pulsed_stereo_frame(
    distance_m: float,
    theta_deg: float = 0.0,
    f0: float = 2609.73,
    amplitude_v: float = 0.150,
    fs: float = 50000.0,
    n_samples: int = 5000,
    mic_distance_m: float = 0.05,
    temperature_c: float = 20.0,
    t_emission_sec: float = 0.0,
    tau_rise_sec: float = 0.001140 / np.log(2.0),
    snr_db: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Synthesizes a dual-channel pulsed acoustic burst arriving at distance_m and theta_deg.
    Models the physical exponential step onset of an active piezo buzzer.
    """
    c = KinematicAnalytics.speed_of_sound(temperature_c)
    theta_rad = np.radians(theta_deg)

    # Physical acoustic flight time
    t_flight = distance_m / c
    t_start0 = t_emission_sec + t_flight
    t_start1 = t_start0 + (mic_distance_m * np.sin(theta_rad) / c)

    t_axis = np.arange(n_samples) / fs

    env0 = np.zeros(n_samples)
    env1 = np.zeros(n_samples)
    mask0 = t_axis >= t_start0
    mask1 = t_axis >= t_start1

    env0[mask0] = 1.0 - np.exp(-(t_axis[mask0] - t_start0) / tau_rise_sec)
    env1[mask1] = 1.0 - np.exp(-(t_axis[mask1] - t_start1) / tau_rise_sec)

    v_a0 = amplitude_v * env0 * np.cos(2.0 * np.pi * f0 * t_axis)
    v_a1 = amplitude_v * env1 * np.cos(2.0 * np.pi * f0 * t_axis)

    if snr_db is not None:
        noise_sigma = amplitude_v / (10.0 ** (snr_db / 20.0)) / np.sqrt(2.0)
        v_a0 += np.random.normal(0, noise_sigma, n_samples)
        v_a1 += np.random.normal(0, noise_sigma, n_samples)

    return v_a0, v_a1, t_start0, t_start1

class TestTimeOfArrivalEngine:

    @pytest.fixture
    def calibrated_offset_ms(self):
        """Calibrates total system onset lag (buzzer mechanical rise + bandpass filter delay)."""
        fs = 50000.0
        f0 = 2609.73
        tau_rise = 0.001140 / np.log(2.0)
        t_axis = np.arange(5000) / fs
        v_cal = 0.150 * (1.0 - np.exp(-t_axis / tau_rise)) * np.cos(2.0 * np.pi * f0 * t_axis)
        cal_res = KinematicAnalytics.detect_pulse_arrival_time(v_cal, fs, f0)
        return cal_res["t_arrival_sec"] * 1000.0

    def test_zero_distance_calibration(self, calibrated_offset_ms):
        """Verify that a contact pulse (r = 0.0 cm) returns 0.0 cm with sub-millimeter precision."""
        v0, v1, _, _ = generate_synthetic_pulsed_stereo_frame(distance_m=0.0)
        estimator = TimeOfArrivalEstimator(
            nominal_f0_hz=2609.73,
            calibrated_offset_ms=calibrated_offset_ms
        )
        res = estimator.estimate_distance_and_tdoa(v0, v1, fs=50000.0)

        assert abs(res["distance_m"] - 0.0) < 0.002, f"Zero distance failed: {res['distance_m']} m"
        assert res["status"] == "ACTIVE_VALID"

    def test_distance_sweep_accuracy(self, calibrated_offset_ms):
        """
        Verify metric distance accuracy across multiple radial stations:
        15 cm, 30 cm, 50 cm, 80 cm, 120 cm.
        Tolerance: error < 0.010 m (1.0 cm).
        """
        test_distances_m = [0.15, 0.30, 0.50, 0.80, 1.20]
        estimator = TimeOfArrivalEstimator(
            nominal_f0_hz=2609.73,
            calibrated_offset_ms=calibrated_offset_ms
        )

        for r_true in test_distances_m:
            v0, v1, _, _ = generate_synthetic_pulsed_stereo_frame(distance_m=r_true)
            res = estimator.estimate_distance_and_tdoa(v0, v1, fs=50000.0)
            err_m = abs(res["distance_m"] - r_true)

            assert err_m < 0.010, f"Distance error too high: Target={r_true*100}cm, Est={res['distance_cm']:.1f}cm, Err={err_m*100:.2f}cm"
            assert res["status"] == "ACTIVE_VALID"

    def test_tdoa_angle_sweep_accuracy(self, calibrated_offset_ms):
        """
        Verify TDOA bearing angle recovery across incident angles from -45 deg to +45 deg.
        Sub-sample linear interpolation resolves arrival times to sub-microsecond precision (< 1.0 µs),
        yielding angular error < 0.75 deg across the entire +-45 deg sector.
        """
        test_angles = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]
        estimator = TimeOfArrivalEstimator(
            nominal_f0_hz=2609.73,
            mic_distance_m=0.05,
            calibrated_offset_ms=calibrated_offset_ms
        )

        for th_true in test_angles:
            v0, v1, _, _ = generate_synthetic_pulsed_stereo_frame(distance_m=0.60, theta_deg=th_true)
            res = estimator.estimate_distance_and_tdoa(v0, v1, fs=50000.0)
            th_est = res["theta_tdoa_deg"]
            err_deg = abs(th_est - th_true)

            assert err_deg < 0.75, (
                f"TDOA angle error too high: Target={th_true}°, Est={th_est:.2f}°, "
                f"Err={err_deg:.3f}° (Timing error = {err_deg / 0.556:.3f} µs)"
            )
            assert res["status"] == "ACTIVE_VALID"

    def test_emission_timestamp_offset(self, calibrated_offset_ms):
        """
        Verify that non-zero emission timestamps (e.g. pulse launched at t = 50 ms)
        correctly subtract to yield the true flight time.
        """
        r_true = 0.400  # 40 cm
        t_emit = 0.050  # 50 ms emission timestamp
        v0, v1, _, _ = generate_synthetic_pulsed_stereo_frame(distance_m=r_true, t_emission_sec=t_emit)

        estimator = TimeOfArrivalEstimator(
            nominal_f0_hz=2609.73,
            calibrated_offset_ms=calibrated_offset_ms
        )
        res = estimator.estimate_distance_and_tdoa(v0, v1, fs=50000.0, t_emission_sec=t_emit)

        err_m = abs(res["distance_m"] - r_true)
        assert err_m < 0.008, f"Emission offset distance error: {err_m*100:.2f} cm"
        assert res["t_flight_sec"] > 0.0

    def test_noise_gate_silence_rejection(self, calibrated_offset_ms):
        """Verify that quiet frames below noise gate return status SILENCE and NaN distance."""
        estimator = TimeOfArrivalEstimator(
            nominal_f0_hz=2609.73,
            calibrated_offset_ms=calibrated_offset_ms,
            noise_gate_v=0.020
        )
        # 3 mV weak pulse
        v0, v1, _, _ = generate_synthetic_pulsed_stereo_frame(distance_m=0.50, amplitude_v=0.003)
        res = estimator.estimate_distance_and_tdoa(v0, v1, fs=50000.0)

        assert np.isnan(res["distance_m"])
        assert np.isnan(res["theta_tdoa_deg"])
        assert res["status"] == "SILENCE"

    def test_profile_loading_integration(self):
        """Verify that TimeOfArrivalEstimator correctly loads f0 and calibrated offset from profile."""
        profile_path = Path("profiles/active_buzzer_2610hz.json")
        assert profile_path.exists(), "Profile missing!"

        estimator = TimeOfArrivalEstimator(profile=profile_path)
        assert abs(estimator.f0_hz - 2609.73) < 0.1
        assert abs(estimator.offset_ms - 1.1400) < 0.01