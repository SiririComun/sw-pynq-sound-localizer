"""
tests/test_multipath_calibration.py: Synthetic Verification Suite for MultipathCalibrationProtocol.
Validates spatial centroid WLS solver against standing wave ripples, multi-sample statistics (N=30),
SWI metrics, JSON profile export, and runtime DistanceEstimator inversion.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from pynq_localizer.kinematics import (
    MultipathCalibrationProtocol,
    AcousticProfile,
    DistanceEstimator,
)

def generate_synthetic_multipath_dataset(
    true_k: float = 0.0450,
    true_c: float = 0.0080,
    distances_m: np.ndarray = None,
    wavelength_m: float = 0.13,
    ripple_ratio: float = 0.35,
    noise_sigma_v: float = 0.001,
    n_samples_per_point: int = 30
):
    """Generates synthetic multi-sample distance observations with standing wave interference."""
    if distances_m is None:
        distances_m = np.linspace(0.10, 1.00, 15)

    dataset = {}
    for r in distances_m:
        interference = ripple_ratio * np.cos(2.0 * np.pi * r / wavelength_m)
        v_clean = (true_k / r) * (1.0 + interference) + true_c
        samples = np.random.normal(loc=v_clean, scale=noise_sigma_v, size=n_samples_per_point)
        dataset[float(r)] = samples
    return distances_m, dataset

class TestMultipathCalibrationProtocol:

    def test_synthetic_ground_truth_k_recovery(self):
        """Verify that spatial centroid WLS recovers true k with < 3.5% error despite 35% ripples."""
        np.random.seed(42)
        true_k = 0.0450
        true_c = 0.0100
        distances, dataset = generate_synthetic_multipath_dataset(
            true_k=true_k, true_c=true_c, ripple_ratio=0.35, wavelength_m=0.13
        )

        protocol = MultipathCalibrationProtocol(r2_threshold=0.80)
        for r, samples in dataset.items():
            protocol.add_measurement(distance_m=r, frequency_hz=2600.0, amplitude_v=samples)

        fits = protocol.fit()
        res = fits[2600.0]
        k_recovered = res["k"]
        error_pct = abs(k_recovered - true_k) / true_k * 100.0

        print(f"\n[Multipath Test] True k={true_k:.4f} | Recovered k={k_recovered:.4f} | Error={error_pct:.2f}% | R²={res['r_squared']:.4f}")
        assert error_pct < 3.5, f"Recovery error too high: {error_pct:.2f}%"
        assert res["r_squared"] > 0.85
        assert res["passed_gate"] is True
        assert res["n_points"] == len(distances)

    def test_statistical_aggregation_n30(self):
        """Verify mean, std, and SEM calculations across N=30 burst observations."""
        protocol = MultipathCalibrationProtocol()

        np.random.seed(123)
        samples_20cm = np.random.normal(loc=0.080, scale=0.002, size=30)
        samples_40cm = np.random.normal(loc=0.040, scale=0.001, size=30)
        samples_60cm = np.random.normal(loc=0.025, scale=0.001, size=30)
        samples_80cm = np.random.normal(loc=0.018, scale=0.001, size=30)

        protocol.add_measurement(distance_m=0.20, frequency_hz=1500.0, amplitude_v=samples_20cm)
        protocol.add_measurement(distance_m=0.40, frequency_hz=1500.0, amplitude_v=samples_40cm)
        protocol.add_measurement(distance_m=0.60, frequency_hz=1500.0, amplitude_v=samples_60cm)
        protocol.add_measurement(distance_m=0.80, frequency_hz=1500.0, amplitude_v=samples_80cm)

        fits = protocol.fit()
        stations = fits[1500.0]["raw_stations"]
        assert len(stations) == 4

        st0 = stations[0]
        assert st0["n_samples"] == 30
        assert abs(st0["mean_v"] - 0.080) < 0.001
        assert 0.0015 < st0["std_v"] < 0.0025
        assert st0["sem_v"] < st0["std_v"] / 4.0
        assert st0["is_pruned_in"] is True

    def test_swi_and_ripple_metrics(self):
        """Verify standing wave index (SWI) and RMS ripple voltage extraction."""
        np.random.seed(99)
        _, dataset = generate_synthetic_multipath_dataset(
            true_k=0.050, ripple_ratio=0.40, wavelength_m=0.13
        )

        protocol = MultipathCalibrationProtocol(r2_threshold=0.75)
        for r, samples in dataset.items():
            protocol.add_measurement(distance_m=r, frequency_hz=2500.0, amplitude_v=samples)

        fits = protocol.fit()
        res = fits[2500.0]

        assert res["standing_wave_index"] > 0.05
        assert res["rms_ripple_v"] > 0.002
        assert "calibration_regime" in res

    def test_end_to_end_profile_export_and_runtime_inversion(self):
        """Verify JSON export, AcousticProfile reload, and runtime DistanceEstimator inversion."""
        protocol = MultipathCalibrationProtocol(
            r2_threshold=0.80,
            system_metadata={"speaker_volume": 0.80, "mic_gain": "+35dB"}
        )
        true_k = 0.0500

        np.random.seed(77)
        for r in np.linspace(0.15, 0.90, 10):
            ripple = 0.25 * np.cos(2.0 * np.pi * r / 0.13)
            v_base = (true_k / r) * (1.0 + ripple) + 0.005
            burst = np.random.normal(loc=v_base, scale=0.001, size=20)
            protocol.add_measurement(distance_m=r, frequency_hz=2000.0, amplitude_v=burst)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_json = Path(tmp.name)

        try:
            saved_path = protocol.save_profile_json(tmp_json, name="MultipathAgnosticProfile")
            assert saved_path.exists()

            loaded_prof = AcousticProfile.from_json(tmp_json)
            assert loaded_prof.name == "MultipathAgnosticProfile"
            assert "2000.0" in loaded_prof.operational_bounds
            assert loaded_prof.system_metadata["speaker_volume"] == 0.80

            # Test runtime estimator
            estimator = DistanceEstimator(profile=loaded_prof, noise_gate_v=0.002)
            k_eval, _ = loaded_prof.evaluate(2000.0)
            sim_amp = k_eval / 0.40  # 40 cm
            r_est, _, status = estimator.estimate_distance(amplitude_v=sim_amp, frequency_hz=2000.0)

            assert abs(r_est - 0.40) < 0.01
            assert status == "ACTIVE_VALID"
        finally:
            if tmp_json.exists():
                tmp_json.unlink()

    def test_degenerate_noise_gate_rejection(self):
        """Verify that flat random noise fails the quality gate."""
        protocol = MultipathCalibrationProtocol(r2_threshold=0.80)

        np.random.seed(55)
        # Flat noise with zero distance decay
        for r in [0.20, 0.40, 0.60, 0.80]:
            flat_noise = np.random.normal(loc=0.015, scale=0.0001, size=25)
            protocol.add_measurement(distance_m=r, frequency_hz=1000.0, amplitude_v=flat_noise)

        fits = protocol.fit()
        res = fits[1000.0]
        assert res["passed_gate"] is False
