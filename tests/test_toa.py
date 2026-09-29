"""
tests/test_toa.py: Strict Unit Verification Suite for Time of Arrival (ToA),
TDOA Bearing, and Exact 2D Cartesian Multilateration Engine.
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pytest

from pynq_localizer.kinematics import KinematicAnalytics, TimeOfArrivalEstimator

def generate_synthetic_2d_stereo_frame(
    x_m: float,
    y_m: float,
    d_m: float = 0.050,
    f0: float = 2660.0,
    amplitude_v: float = 0.150,
    fs: float = 50000.0,
    n_samples: int = 5000,
    temperature_c: float = 20.0,
    t_emission_sec: float = 0.0,
    tau_rise_sec: float = 0.001808 / np.log(2.0),
    snr_db: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Synthesizes exact near-field stereo microphone signals for a sound emitter at (x_m, y_m).

    Geometry:
      Mic 1 (Left / A0)  at (-d/2, 0)
      Mic 2 (Right / A1) at (+d/2, 0)
    """
    c = KinematicAnalytics.speed_of_sound(temperature_c)

    # Exact near-field Euclidean ranges
    r1 = np.sqrt((x_m + d_m / 2.0) ** 2 + y_m ** 2)
    r2 = np.sqrt((x_m - d_m / 2.0) ** 2 + y_m ** 2)

    # Physical acoustic arrival times
    t_start0 = t_emission_sec + (r1 / c)
    t_start1 = t_emission_sec + (r2 / c)

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

    return v_a0, v_a1, r1, r2, t_start0, t_start1

class TestTimeOfArrivalEngine:

    @pytest.fixture
    def calibrated_offset_ms(self):
        """Calibrates system onset lag (buzzer rise + filter delay) from a reference onset burst."""
        fs = 50000.0
        f0 = 2660.0
        tau_rise = 0.001808 / np.log(2.0)
        t_axis = np.arange(5000) / fs
        v_cal = 0.150 * (1.0 - np.exp(-t_axis / tau_rise)) * np.cos(2.0 * np.pi * f0 * t_axis)
        cal_res = KinematicAnalytics.detect_pulse_arrival_time(v_cal, fs, f0)
        return cal_res["t_arrival_sec"] * 1000.0

    # =========================================================================
    # 1. Pure Analytical Multilateration Unit Tests
    # =========================================================================

    def test_multilateration_broadside_center(self):
        """Verify (x=0, y=40cm) broadside localization: theta = 0.0 deg."""
        d = 0.050
        x_true, y_true = 0.0, 0.400
        r1_true = np.sqrt((x_true + d / 2.0) ** 2 + y_true ** 2)
        r2_true = np.sqrt((x_true - d / 2.0) ** 2 + y_true ** 2)

        res = KinematicAnalytics.solve_2d_multilateration(r1_true, r2_true, d)

        assert abs(res["x_m"] - 0.0) < 1e-4
        assert abs(res["y_m"] - 0.400) < 1e-4
        assert abs(res["range_m"] - 0.400) < 1e-4
        assert abs(res["theta_deg"] - 0.0) < 1e-3
        assert res["status"] == "ACTIVE_VALID"

    def test_multilateration_right_sector_positive_angle(self):
        """Verify (x=+15cm, y=30cm): theta > 0 deg (closer to Mic 2)."""
        d = 0.050
        x_true, y_true = 0.150, 0.300
        r1_true = np.sqrt((x_true + d / 2.0) ** 2 + y_true ** 2)
        r2_true = np.sqrt((x_true - d / 2.0) ** 2 + y_true ** 2)
        expected_theta = np.degrees(np.arctan2(x_true, y_true))

        res = KinematicAnalytics.solve_2d_multilateration(r1_true, r2_true, d)

        assert abs(res["x_m"] - 0.150) < 1e-4
        assert abs(res["y_m"] - 0.300) < 1e-4
        assert abs(res["range_m"] - np.sqrt(x_true**2 + y_true**2)) < 1e-4
        assert abs(res["theta_deg"] - expected_theta) < 1e-3
        assert res["status"] == "ACTIVE_VALID"

    def test_multilateration_left_sector_negative_angle(self):
        """Verify (x=-15cm, y=30cm): theta < 0 deg (closer to Mic 1)."""
        d = 0.050
        x_true, y_true = -0.150, 0.300
        r1_true = np.sqrt((x_true + d / 2.0) ** 2 + y_true ** 2)
        r2_true = np.sqrt((x_true - d / 2.0) ** 2 + y_true ** 2)
        expected_theta = np.degrees(np.arctan2(x_true, y_true))

        res = KinematicAnalytics.solve_2d_multilateration(r1_true, r2_true, d)

        assert abs(res["x_m"] - (-0.150)) < 1e-4
        assert abs(res["y_m"] - 0.300) < 1e-4
        assert abs(res["theta_deg"] - expected_theta) < 1e-3
        assert res["status"] == "ACTIVE_VALID"

    def test_multilateration_triangle_inequality_gate(self):
        """Verify that |r1 - r2| > d flags GEOMETRIC_OUT_OF_BOUNDS and clamps gracefully."""
        d = 0.050
        r1, r2 = 0.500, 0.420
        res = KinematicAnalytics.solve_2d_multilateration(r1, r2, d, wrap_modulo_lambda=False)

        assert res["status"] == "GEOMETRIC_OUT_OF_BOUNDS"
        assert np.isfinite(res["x_m"])
        assert np.isfinite(res["theta_far_deg"])

    def test_cycle_slip_parity_reconstruction(self):
        """Verify that delta_r exceeding baseline is repaired via modulo-lambda wrapping."""
        d = 0.050
        f0 = 2660.0
        c_sound = 343.21

        r1, r2 = 0.500, 0.420
        res = KinematicAnalytics.solve_2d_multilateration(
            r1, r2, d, f0=f0, c_sound=c_sound, wrap_modulo_lambda=True
        )

        assert res["status"] == "ACTIVE_VALID"
        assert abs(res["theta_far_deg"]) <= 90.0
        assert np.isfinite(res["x_m"])
        assert res["y_m"] > 0.0

    # =========================================================================
    # 2. End-to-End Waveform Localization Tests
    # =========================================================================

    def test_2d_waveform_localization_broadside(self, calibrated_offset_ms):
        """Verify waveform end-to-end 2D solver at (x=0, y=40cm)."""
        x_t, y_t, d = 0.0, 0.400, 0.050
        v0, v1, r1, r2, _, _ = generate_synthetic_2d_stereo_frame(x_m=x_t, y_m=y_t, d_m=d, f0=2660.0)

        estimator = TimeOfArrivalEstimator(
            nominal_f0_hz=2660.0,
            mic_distance_m=d,
            calibrated_offset_ms=calibrated_offset_ms
        )
        res = estimator.estimate_distance_and_tdoa(v0, v1, fs=50000.0)

        assert abs(res["x_cm"] - 0.0) < 0.5
        assert abs(res["y_cm"] - 40.0) < 1.0
        assert abs(res["theta_deg"] - 0.0) < 0.75
        assert res["status"] == "ACTIVE_VALID"

    def test_2d_waveform_localization_right_sector(self, calibrated_offset_ms):
        """Verify waveform end-to-end 2D solver at (x=+15cm, y=30cm)."""
        x_t, y_t, d = 0.150, 0.300, 0.050
        v0, v1, _, _, _, _ = generate_synthetic_2d_stereo_frame(x_m=x_t, y_m=y_t, d_m=d, f0=2660.0)

        estimator = TimeOfArrivalEstimator(
            nominal_f0_hz=2660.0,
            mic_distance_m=d,
            calibrated_offset_ms=calibrated_offset_ms
        )
        res = estimator.estimate_distance_and_tdoa(v0, v1, fs=50000.0)

        assert abs(res["x_cm"] - 15.0) < 0.6
        assert abs(res["y_cm"] - 30.0) < 1.0
        assert abs(res["theta_deg"] - 26.56) < 0.8
        assert res["status"] == "ACTIVE_VALID"

    def test_2d_waveform_localization_left_sector(self, calibrated_offset_ms):
        """Verify waveform end-to-end 2D solver at (x=-15cm, y=30cm)."""
        x_t, y_t, d = -0.150, 0.300, 0.050
        v0, v1, _, _, _, _ = generate_synthetic_2d_stereo_frame(x_m=x_t, y_m=y_t, d_m=d, f0=2660.0)

        estimator = TimeOfArrivalEstimator(
            nominal_f0_hz=2660.0,
            mic_distance_m=d,
            calibrated_offset_ms=calibrated_offset_ms
        )
        res = estimator.estimate_distance_and_tdoa(v0, v1, fs=50000.0)

        assert abs(res["x_cm"] - (-15.0)) < 0.6
        assert abs(res["y_cm"] - 30.0) < 1.0
        assert abs(res["theta_deg"] - (-26.56)) < 0.8
        assert res["status"] == "ACTIVE_VALID"

    # =========================================================================
    # 3. Robustness, Emission Offsets, and Profile Integration
    # =========================================================================

    def test_emission_timestamp_offset(self, calibrated_offset_ms):
        """Verify non-zero pulse emission timestamp subtraction."""
        x_t, y_t, d = 0.0, 0.400, 0.050
        t_emit = 0.050
        v0, v1, _, _, _, _ = generate_synthetic_2d_stereo_frame(
            x_m=x_t, y_m=y_t, d_m=d, f0=2660.0, t_emission_sec=t_emit
        )

        estimator = TimeOfArrivalEstimator(
            nominal_f0_hz=2660.0,
            mic_distance_m=d,
            calibrated_offset_ms=calibrated_offset_ms
        )
        res = estimator.estimate_distance_and_tdoa(v0, v1, fs=50000.0, t_emission_sec=t_emit)

        assert abs(res["y_cm"] - 40.0) < 1.0
        assert res["t_flight_sec"] > 0.0

    def test_noise_gate_silence_rejection(self, calibrated_offset_ms):
        """Verify weak signals below noise gate return status SILENCE and NaN coordinates."""
        v0, v1, _, _, _, _ = generate_synthetic_2d_stereo_frame(
            x_m=0.0, y_m=0.50, f0=2660.0, amplitude_v=0.003
        )
        estimator = TimeOfArrivalEstimator(
            nominal_f0_hz=2660.0,
            calibrated_offset_ms=calibrated_offset_ms,
            noise_gate_v=0.020
        )
        res = estimator.estimate_distance_and_tdoa(v0, v1, fs=50000.0)

        assert np.isnan(res["x_cm"])
        assert np.isnan(res["y_cm"])
        assert np.isnan(res["theta_deg"])
        assert res["status"] == "SILENCE"

    def test_profile_loading_integration(self):
        """Verify profile parameter resolution and calibrated offset alignment."""
        profile_path = Path("profiles/active_buzzer_profile.json")
        if not profile_path.exists():
            profile_path = Path("profiles/active_buzzer_2610hz.json")
        assert profile_path.exists(), "Profile file missing!"

        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
        expected_offset = float(profile_data.get("calibrated_toa_offset_m1_ms", profile_data.get("calibrated_toa_offset_ms", 1.8080)))
        expected_f0 = float(profile_data.get("f_res_hz", 2660.0))

        estimator = TimeOfArrivalEstimator(profile=profile_path)
        assert abs(estimator.f0_hz - expected_f0) < 0.1
        assert abs(estimator.offset_ms - expected_offset) < 0.001