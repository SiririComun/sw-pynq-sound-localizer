"""
tests/test_distance.py: Verification Suite for AcousticProfile and DistanceEstimator Engine.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from pynq_localizer.kinematics import AcousticProfile, DistanceEstimator


class TestAcousticProfile:

    def test_profile_interpolation(self):
        """Verify continuous k(f) interpolation across discrete frequency points."""
        frequencies = [500.0, 1000.0, 2000.0, 4000.0]
        k_values = [0.030, 0.050, 0.080, 0.120]  # k in V*m
        r2_values = [0.98, 0.99, 0.97, 0.96]
        k_err = [0.001, 0.002, 0.003, 0.004]

        profile = AcousticProfile(
            frequencies_hz=frequencies,
            k_values=k_values,
            r_squared=r2_values,
            k_uncertainty=k_err,
            system_metadata={"speaker_volume": 0.75, "mic_gain": "+35dB"},
            name="SpeakerA_Calibrated"
        )

        # Exact grid evaluation
        k_1000, err_1000 = profile.evaluate(1000.0)
        assert abs(k_1000 - 0.050) < 1e-6
        assert abs(err_1000 - 0.002) < 1e-6

        # Intermediate interpolated frequency (1500 Hz)
        k_1500, err_1500 = profile.evaluate(1500.0)
        assert 0.050 < k_1500 < 0.080
        assert 0.002 < err_1500 < 0.003

    def test_profile_json_roundtrip(self):
        """Verify export to JSON and reload preserving metadata and operational bounds."""
        frequencies = [800.0, 1500.0, 3000.0]
        k_values = [0.045, 0.065, 0.095]
        r2_values = [0.992, 0.985, 0.978]
        bounds = {
            "800.0": {"r_min_m": 0.20, "r_max_m": 0.90, "v_sat_v": 0.22, "v_min_v": 0.005},
            "1500.0": {"r_min_m": 0.15, "r_max_m": 1.20, "v_sat_v": 0.43, "v_min_v": 0.004},
            "3000.0": {"r_min_m": 0.25, "r_max_m": 0.80, "v_sat_v": 0.38, "v_min_v": 0.006},
        }
        metadata = {"speaker_volume": 0.80, "mic_gain": "+35dB", "environment": "test_lab"}

        profile_orig = AcousticProfile(
            frequencies_hz=frequencies,
            k_values=k_values,
            r_squared=r2_values,
            operational_bounds=bounds,
            system_metadata=metadata,
            name="JSON_Test_Profile"
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            profile_orig.to_json(tmp_path)
            profile_loaded = AcousticProfile.from_json(tmp_path)

            assert profile_loaded.name == "JSON_Test_Profile"
            assert np.allclose(profile_loaded.frequencies, frequencies)
            assert np.allclose(profile_loaded.k_values, k_values)
            assert np.allclose(profile_loaded.r_squared, r2_values)
            assert profile_loaded.system_metadata["speaker_volume"] == 0.80
            assert "1500.0" in profile_loaded.operational_bounds
            assert profile_loaded.operational_bounds["1500.0"]["r_max_m"] == 1.20

            # Evaluate loaded profile
            k_val, _ = profile_loaded.evaluate(1500.0)
            assert abs(k_val - 0.065) < 1e-6
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_from_constant_and_callable(self):
        """Verify constant and callable profile factories."""
        prof_const = AcousticProfile.from_constant(0.060, relative_error=0.05)
        k_val, k_err = prof_const.evaluate(2500.0)
        assert abs(k_val - 0.060) < 1e-6
        assert abs(k_err - 0.003) < 1e-6

        prof_callable = AcousticProfile.from_callable(lambda f: 0.02 + 0.00001 * f)
        k_val, _ = prof_callable.evaluate(3000.0)
        assert abs(k_val - 0.050) < 1e-6


class TestDistanceEstimator:

    def test_single_channel_distance_inversion(self):
        """Verify r = k(f0) / A(t) calculation and dynamic error propagation."""
        prof = AcousticProfile.from_constant(0.050, relative_error=0.04)
        estimator = DistanceEstimator(
            profile=prof,
            noise_gate_v=0.003,
            voltage_uncertainty_v=0.0005
        )

        # Test at r = 1.00 m (A = 0.050 V -> r = 1.00 m)
        r_est, r_err, status = estimator.estimate_distance(amplitude_v=0.050, frequency_hz=1000.0)
        assert abs(r_est - 1.00) < 1e-3
        assert 0.035 < r_err < 0.050
        assert status == "ACTIVE_VALID"

        # Test at r = 0.50 m (A = 0.100 V -> r = 0.50 m)
        r_est_half, _, status_half = estimator.estimate_distance(amplitude_v=0.100, frequency_hz=1000.0)
        assert abs(r_est_half - 0.50) < 1e-3
        assert status_half == "ACTIVE_VALID"

    def test_noise_gate_squelching(self):
        """Verify that amplitudes below noise gate return (NaN, NaN, 'SILENCE')."""
        estimator = DistanceEstimator(k_constant=0.050, noise_gate_v=0.005)

        r_est, r_err, status = estimator.estimate_distance(amplitude_v=0.002, frequency_hz=1000.0)
        assert np.isnan(r_est)
        assert np.isnan(r_err)
        assert status == "SILENCE"

    def test_operational_bounds_status_checking(self):
        """Verify that operational boundaries assign correct status flags."""
        bounds = {
            "1000.0": {"r_min_m": 0.20, "r_max_m": 0.80, "v_sat_v": 0.25, "v_min_v": 0.010}
        }
        prof = AcousticProfile(
            frequencies_hz=[1000.0],
            k_values=[0.050],
            operational_bounds=bounds
        )
        estimator = DistanceEstimator(profile=prof, noise_gate_v=0.002)

        # In-bounds test (r = 0.50 m -> inside [0.20, 0.80])
        _, _, st_valid = estimator.estimate_distance(amplitude_v=0.100, frequency_hz=1000.0)
        assert st_valid == "ACTIVE_VALID"

        # Saturation near-field test (A = 0.50 V -> r_calc = 0.10 m < 0.20 m r_min)
        _, _, st_sat = estimator.estimate_distance(amplitude_v=0.500, frequency_hz=1000.0)
        assert st_sat == "OUT_OF_BOUNDS_SATURATION"

        # Far-field noise test (A = 0.005 V -> r_calc = 10.0 m > 0.80 m r_max)
        _, _, st_noise = estimator.estimate_distance(amplitude_v=0.005, frequency_hz=1000.0)
        assert st_noise == "OUT_OF_BOUNDS_NOISE"

    def test_process_quadruple_augmentation(self):
        """Verify seamless augmentation of quadruple dictionaries with distance metrics."""
        estimator = DistanceEstimator(k_constant=0.040, noise_gate_v=0.003)

        raw_quad = {
            "frequency_hz": 1200.0,
            "amplitude_v": 0.080,  # 80 mV -> r = 0.040 / 0.080 = 0.50 m
            "phase_rad": 0.45,
            "timestamp_sec": 1.234567,
            "is_valid": True
        }

        augmented = estimator.process_quadruple(raw_quad)

        assert "distance_m" in augmented
        assert "distance_err_m" in augmented
        assert "distance_status" in augmented
        assert "k_evaluated" in augmented
        assert abs(augmented["distance_m"] - 0.50) < 1e-3
        assert abs(augmented["k_evaluated"] - 0.040) < 1e-6
        assert augmented["distance_status"] == "ACTIVE_VALID"