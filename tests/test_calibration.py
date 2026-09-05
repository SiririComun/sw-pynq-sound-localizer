"""
tests/test_calibration.py: Verification Suite for AcousticCalibrationProtocol Engine.
Tests multi-sample statistics (N=30), dynamic boundary pruning, WLS regression, and JSON export.
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

    def test_multi_sample_statistical_aggregation(self):
        """Verify that N=30 repeat frame observations correctly compute mean, std, and SEM."""
        protocol = AcousticCalibrationProtocol()

        # Simulate 30 frame observations for 2 distance stations
        np.random.seed(42)
        samples_30cm = np.random.normal(loc=0.035, scale=0.001, size=30)
        samples_50cm = np.random.normal(loc=0.021, scale=0.001, size=30)

        protocol.add_measurement(distance_m=0.30, frequency_hz=1000.0, amplitude_v=samples_30cm)
        protocol.add_measurement(distance_m=0.50, frequency_hz=1000.0, amplitude_v=samples_50cm)
        fits = protocol.fit()

        stations = fits[1000.0]["raw_stations"]
        assert len(stations) == 2
        st0 = stations[0]
        st1 = stations[1]

        assert st0["n_samples"] == 30
        assert abs(st0["mean_v"] - 0.035) < 0.0005
        assert 0.0007 < st0["std_v"] < 0.0013
        assert st0["sem_v"] < st0["std_v"] / 4.0

        assert st1["n_samples"] == 30
        assert abs(st1["mean_v"] - 0.021) < 0.0005

    def test_wls_regression_recovery_accuracy(self):
        """
        Verify that WLS 1/r regression recovers ground-truth k(f) across
        a multi-frequency grid with < 1% error and high R^2.
        """
        distances_m = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.00]
        frequencies_hz = [500.0, 1000.0, 2000.0, 3500.0]

        true_k = np.array([0.020 + 0.00002 * f for f in frequencies_hz])  # [0.030, 0.040, 0.060, 0.090]
        true_c_room = 0.0012  # 1.2 mV room reflection baseline

        protocol = AcousticCalibrationProtocol(
            r2_threshold=0.95,
            system_metadata={"speaker_volume": 0.75, "mic_gain": "+35dB"}
        )

        np.random.seed(123)
        for i, r in enumerate(distances_m):
            for j, f in enumerate(frequencies_hz):
                # Simulate N=20 frame burst per point
                v_expected = (true_k[j] / r) + true_c_room
                samples = np.random.normal(loc=v_expected, scale=0.0005, size=20)
                protocol.add_measurement(distance_m=r, frequency_hz=f, amplitude_v=samples)

        fit_results = protocol.fit()
        assert len(fit_results) == len(frequencies_hz)

        for j, f in enumerate(frequencies_hz):
            res = fit_results[f]
            recovered_k = res["k"]
            r2 = res["r_squared"]
            error_pct = abs(recovered_k - true_k[j]) / true_k[j] * 100.0

            print(f"\n[WLS Test] f={f:.0f}Hz | True k={true_k[j]:.4f} | Recovered k={recovered_k:.4f} | Error={error_pct:.3f}% | R^2={r2:.5f}")
            assert error_pct < 1.0, f"k(f) recovery error too high for {f} Hz: {error_pct}%"
            assert r2 > 0.98, f"R^2 too low for {f} Hz: {r2}"
            assert res["passed_gate"] is True

    def test_dynamic_boundary_pruning_saturation_and_reverberation(self):
        """
        Verify that the dynamic boundary pruner detects and excludes near-field saturation
        at small r and far-field room reflection floors at large r, isolating the true 1/r window.
        """
        protocol = AcousticCalibrationProtocol(r2_threshold=0.95)
        k_true = 0.050

        # Synthesize a 12-point distance sweep:
        # [0.10, 0.15] -> Saturated clipping ceiling (flat at 0.30 V)
        # [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85] -> True 1/r decay (V = 0.050 / r)
        # [1.20, 1.50, 2.00] -> Reverberation floor (flat at 0.008 V)
        distances = [0.10, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.20, 1.50, 2.00]
        voltages = []

        for r in distances:
            if r <= 0.15:
                v = 0.300  # Saturated flat ceiling
            elif r >= 1.20:
                v = 0.008  # Flat reverberant room echo floor
            else:
                v = k_true / r  # True 1/r physics
            voltages.append(v)
            protocol.add_measurement(distance_m=r, frequency_hz=2000.0, amplitude_v=v)

        fits = protocol.fit()
        res = fits[2000.0]

        # The pruner should have excluded the saturated points and far reverberant points
        assert res["passed_gate"] is True
        assert res["n_pruned_points"] < res["n_total_points"]
        assert abs(res["k"] - k_true) / k_true * 100.0 < 1.0  # Recovers 0.050 with < 1% error
        assert res["r_squared"] > 0.99

        # Certified linear window boundaries
        assert res["r_valid_min_m"] >= 0.25  # Excluded 0.10 m and 0.15 m
        assert res["r_valid_max_m"] <= 0.85  # Excluded 1.20 m, 1.50 m, 2.00 m

    def test_end_to_end_profile_export_and_runtime_inversion(self):
        """Verify full calibration export with metadata, bounds, and DistanceEstimator runtime verification."""
        metadata = {"speaker_volume": 0.75, "mic_gain": "+35dB", "environment": "physics_lab"}
        protocol = AcousticCalibrationProtocol(r2_threshold=0.95, system_metadata=metadata)

        for f, k_val in [(1000.0, 0.040), (2000.0, 0.060)]:
            for r in [0.25, 0.35, 0.50, 0.70, 0.90]:
                v = (k_val / r) + 0.001
                protocol.add_measurement(distance_m=r, frequency_hz=f, amplitude_v=v)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_json = Path(tmp.name)

        try:
            protocol.save_profile_json(tmp_json, name="CalibratedLabProfile")

            # Load into DistanceEstimator
            loaded_prof = AcousticProfile.from_json(tmp_json)
            assert loaded_prof.system_metadata["speaker_volume"] == 0.75
            assert "1000.0" in loaded_prof.operational_bounds

            estimator = DistanceEstimator(profile=loaded_prof, noise_gate_v=0.002)

            # Test in-bounds distance inversion at f = 2000 Hz (k = 0.060)
            # If A = 120 mV (0.120 V) -> r = 0.060 / 0.120 = 0.50 m
            r_est, _, status = estimator.estimate_distance(amplitude_v=0.120, frequency_hz=2000.0)
            assert abs(r_est - 0.50) < 0.01
            assert status == "ACTIVE_VALID"

        finally:
            if tmp_json.exists():
                tmp_json.unlink()