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
            name="SpeakerA_Calibrated"
        )

        # Exact grid evaluation
        k_1000, err_1000 = profile.evaluate(1000.0)
        assert abs(k_1000 - 0.050) < 1e-6
        assert abs(err_1000 - 0.002) < 1e-6

        # Intermediate interpolated frequency (1500 Hz between 1000 Hz and 2000 Hz)
        k_1500, err_1500 = profile.evaluate(1500.0)
        assert 0.050 < k_1500 < 0.080
        assert 0.002 < err_1500 < 0.003

    def test_profile_json_roundtrip(self):
        """Verify export to JSON and reload without precision loss."""
        frequencies = [800.0, 1500.0, 3000.0]
        k_values = [0.045, 0.065, 0.095]
        r2_values = [0.992, 0.985, 0.978]

        profile_orig = AcousticProfile(
            frequencies_hz=frequencies,
            k_values=k_values,
            r_squared=r2_values,
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

            # Evaluate loaded profile
            k_val, _ = profile_loaded.evaluate(1500.0)
            assert abs(k_val - 0.065) < 1e-6
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_from_constant_and_callable(self):
        """Verify constant and callable profile factories."""
        # 1. Constant Profile
        prof_const = AcousticProfile.from_constant(0.060, relative_error=0.05)
        k_val, k_err = prof_const.evaluate(2500.0)
        assert abs(k_val - 0.060) < 1e-6
        assert abs(k_err - 0.003) < 1e-6

        # 2. Callable Profile: k(f) = 0.02 + 0.00001 * f
        prof_callable = AcousticProfile.from_callable(lambda f: 0.02 + 0.00001 * f)
        k_val, _ = prof_callable.evaluate(3000.0)
        assert abs(k_val - 0.050) < 1e-6


class TestDistanceEstimator:

    def test_single_channel_distance_inversion(self):
        """Verify r = k(f0) / A(t) calculation and dynamic error propagation."""
        # Flat profile with k = 0.050 V*m and delta_k = 0.002 V*m (4% rel error)
        prof = AcousticProfile.from_constant(0.050, relative_error=0.04)
        estimator = DistanceEstimator(
            profile=prof,
            noise_gate_v=0.003,
            voltage_uncertainty_v=0.0005  # 0.5 mV noise
        )

        # Test at r = 1.00 m (A = k / r = 0.050 / 1.0 = 0.050 V = 50 mV)
        r_est, r_err = estimator.estimate_distance(amplitude_v=0.050, frequency_hz=1000.0)
        assert abs(r_est - 1.00) < 1e-3
        # Expected error: r * sqrt((0.04)^2 + (0.0005/0.05)^2) = 1.0 * sqrt(0.0016 + 0.0001) = ~0.041 m
        assert 0.035 < r_err < 0.050

        # Test at r = 0.50 m (A = 0.100 V = 100 mV)
        r_est_half, _ = estimator.estimate_distance(amplitude_v=0.100, frequency_hz=1000.0)
        assert abs(r_est_half - 0.50) < 1e-3

        # Test at r = 2.00 m (A = 0.025 V = 25 mV)
        r_est_double, _ = estimator.estimate_distance(amplitude_v=0.025, frequency_hz=1000.0)
        assert abs(r_est_double - 2.00) < 1e-3

    def test_noise_gate_squelching(self):
        """Verify that amplitudes below noise gate return (NaN, NaN)."""
        estimator = DistanceEstimator(k_constant=0.050, noise_gate_v=0.005)

        # Signal below noise gate (A = 0.002 V = 2 mV < 5 mV threshold)
        r_est, r_err = estimator.estimate_distance(amplitude_v=0.002, frequency_hz=1000.0)
        assert np.isnan(r_est)
        assert np.isnan(r_err)

    def test_process_quadruple_augmentation(self):
        """Verify seamless augmentation of quadruple dictionaries with distance metrics."""
        estimator = DistanceEstimator(k_constant=0.040, noise_gate_v=0.003)

        raw_quad = {
            "frequency_hz": 1200.0,
            "amplitude_v": 0.080, # 80 mV -> r = 0.040 / 0.080 = 0.50 m
            "phase_rad": 0.45,
            "timestamp_sec": 1.234567,
            "is_valid": True
        }

        augmented = estimator.process_quadruple(raw_quad)

        assert "distance_m" in augmented
        assert "distance_err_m" in augmented
        assert "k_evaluated" in augmented
        assert abs(augmented["distance_m"] - 0.50) < 1e-3
        assert abs(augmented["k_evaluated"] - 0.040) < 1e-6