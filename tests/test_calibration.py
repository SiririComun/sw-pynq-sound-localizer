"""
tests/test_calibration.py: Verification Suite for AcousticCalibrationProtocol Engine.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from pynq_localizer.kinematics import (
    AcousticCalibrationProtocol,
    AcousticProfile,
    DistanceEstimator,
)

class TestAcousticCalibrationProtocol:

    def test_synthetic_grid_regression_accuracy(self):
        """
        Verify that 1/r linear regression recovers known ground-truth k(f)
        across a multi-frequency x multi-distance grid with < 1% error.
        """
        distances_m = [0.25, 0.40, 0.60, 0.80, 1.00, 1.20]
        frequencies_hz = [500.0, 1000.0, 2000.0, 3500.0]

        # Ground-truth physical k(f) function: k(f) = 0.020 + 0.00002 * f
        true_k = np.array([0.020 + 0.00002 * f for f in frequencies_hz])  # [0.030, 0.040, 0.060, 0.090]
        true_c_room = 0.0015  # 1.5 mV room reflection floor

        # Synthesize V_RMS(r_i, f_j) = k(f_j) / r_i + c_room
        amp_matrix = np.zeros((len(distances_m), len(frequencies_hz)), dtype=np.float64)
        for i, r in enumerate(distances_m):
            for j, f in enumerate(frequencies_hz):
                amp_matrix[i, j] = (true_k[j] / r) + true_c_room

        # Run Calibration Protocol
        protocol = AcousticCalibrationProtocol(r2_threshold=0.95)
        protocol.add_dataset(distances_m, frequencies_hz, amp_matrix)
        fit_results = protocol.fit()

        assert len(fit_results) == len(frequencies_hz)

        for j, f in enumerate(frequencies_hz):
            res = fit_results[f]
            recovered_k = res["k"]
            r2 = res["r_squared"]
            error_pct = abs(recovered_k - true_k[j]) / true_k[j] * 100.0

            print(f"\n[Calib Test] f={f:.0f}Hz | True k={true_k[j]:.4f} | Recovered k={recovered_k:.4f} | Error={error_pct:.3f}% | R^2={r2:.5f}")
            assert error_pct < 1.0, f"k(f) recovery error too high for {f} Hz: {error_pct}%"
            assert r2 > 0.99, f"R^2 too low for {f} Hz: {r2}"
            assert res["passed_gate"] is True

    def test_room_intercept_absorption(self):
        """Verify that ambient room reflection offset c_room is isolated without biasing k."""
        distances_m = [0.30, 0.50, 0.70, 0.90]
        k_true = 0.050
        c_room_true = 0.003  # 3 mV room echo floor

        protocol = AcousticCalibrationProtocol(r2_threshold=0.95)
        for r in distances_m:
            v_measured = (k_true / r) + c_room_true
            protocol.add_measurement(distance_m=r, frequency_hz=1500.0, amplitude_v=v_measured)

        fits = protocol.fit()
        res = fits[1500.0]

        assert abs(res["k"] - k_true) < 1e-4
        assert abs(res["c_room"] - c_room_true) < 1e-4
        assert res["r_squared"] > 0.999

    def test_quality_gate_rejection(self):
        """Verify that non-linear or severely corrupted data points fail R^2 >= 0.95 gate."""
        protocol = AcousticCalibrationProtocol(r2_threshold=0.95)

        # Tone 1 (1000 Hz): Clean 1/r linear decay -> Should PASS
        for r in [0.30, 0.60, 0.90]:
            protocol.add_measurement(distance_m=r, frequency_hz=1000.0, amplitude_v=0.040 / r)

        # Tone 2 (2500 Hz): Saturated/corrupted constant amplitude -> Should FAIL
        for r in [0.30, 0.60, 0.90]:
            protocol.add_measurement(distance_m=r, frequency_hz=2500.0, amplitude_v=0.020)  # Flat, non-decaying

        fits = protocol.fit()
        assert fits[1000.0]["passed_gate"] is True
        assert fits[2500.0]["passed_gate"] is False  # R^2 ~ 0 -> Failed gate!

        # Exporting profile with only_passed=True should only contain 1000 Hz
        profile = protocol.export_profile(only_passed=True)
        assert len(profile.frequencies) == 1
        assert profile.frequencies[0] == 1000.0

    def test_end_to_end_export_and_runtime_inversion(self):
        """Verify export to JSON and real-time distance inference in DistanceEstimator."""
        protocol = AcousticCalibrationProtocol(r2_threshold=0.95)

        # Add 3 frequencies
        for f, k_val in [(800.0, 0.035), (1600.0, 0.055), (3200.0, 0.085)]:
            for r in [0.30, 0.50, 0.80, 1.00]:
                protocol.add_measurement(distance_m=r, frequency_hz=f, amplitude_v=k_val / r)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_json = Path(tmp.name)

        try:
            # Save profile to JSON
            protocol.save_profile_json(tmp_json, name="TestRoomProfile")

            # Load profile into DistanceEstimator
            loaded_prof = AcousticProfile.from_json(tmp_json)
            estimator = DistanceEstimator(profile=loaded_prof, noise_gate_v=0.002)

            # Test distance inference at f = 1600 Hz (k = 0.055)
            # If A = 55 mV (0.055 V) -> r should be 1.00 m
            r_1m, _ = estimator.estimate_distance(amplitude_v=0.055, frequency_hz=1600.0)
            assert abs(r_1m - 1.00) < 0.01

            # If A = 110 mV (0.110 V) -> r should be 0.50 m
            r_50cm, _ = estimator.estimate_distance(amplitude_v=0.110, frequency_hz=1600.0)
            assert abs(r_50cm - 0.50) < 0.01

            # Test distance inference at f = 800 Hz (k = 0.035)
            # If A = 70 mV (0.070 V) -> r should be 0.50 m
            r_800, _ = estimator.estimate_distance(amplitude_v=0.070, frequency_hz=800.0)
            assert abs(r_800 - 0.50) < 0.01

        finally:
            if tmp_json.exists():
                tmp_json.unlink()