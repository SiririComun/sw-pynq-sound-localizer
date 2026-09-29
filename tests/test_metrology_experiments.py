"""
tests/test_metrology_experiments.py: Verification Suite for Phase 2-5 Metrology Additions.
Tests:
  1. Zero-intercept energy (1/r²) WLS regression in DirectPulseCalibrationProtocol.
  2. Normalized differential energy angle-of-arrival solver (calculate_energy_bearing).
  3. Statistical metrology aggregator (compute_metrology_statistics).
  4. Dual-channel metric energy distance inversion (estimate_distance_energy).
  5. Tracker video cross-correlation alignment and synchronization (align_tracker_ground_truth).
  6. Theoretical Newtonian air-track kinematic simulators (Atwood & Elastic).
  7. Single-microphone Doppler velocity calculations in DifferentialDopplerTracker.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from pynq_localizer.kinematics import (
    KinematicAnalytics,
    DirectPulseCalibrationProtocol,
    AcousticProfile,
    DistanceEstimator,
    DifferentialDopplerTracker
)


class TestMetrologyAndExperimentalEngines:

    # =========================================================================
    # 1. Energy Inverse-Square (1/r²) Regression Solver
    # =========================================================================

    def test_energy_decay_wls_recovery(self):
        """Verify zero-intercept WLS recovers ground-truth k_E with < 1% error and high R²."""
        distances = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
        true_k_a = 0.0196          # V*m
        true_k_e = 0.000384        # counts² * m²
        f0 = 2660.0

        protocol = DirectPulseCalibrationProtocol(
            nominal_f0_hz=f0,
            r2_threshold=0.98,
            temperature_c=20.0
        )

        np.random.seed(42)
        for r in distances:
            # Generate amplitude (1/r) and energy (1/r²) with realistic noise
            a_clean = true_k_a / r
            e_clean = true_k_e / (r ** 2)
            shot_a = np.random.normal(loc=a_clean, scale=0.0005, size=20)
            shot_e = np.random.normal(loc=e_clean, scale=0.01 * e_clean, size=20)

            protocol.add_measurement(distance_m=r, amplitude_v=shot_a, frequency_hz=f0, energy_direct=shot_e, channel=1)

        fits = protocol.fit_channel(channel=1)
        res = fits[f0]

        recovered_k_e = res["k_energy"]
        r2_e = res["r_squared_energy"]
        err_pct = abs(recovered_k_e - true_k_e) / true_k_e * 100.0

        print(f"\n[Energy WLS Test] True k_E={true_k_e:.6f} | Recovered k_E={recovered_k_e:.6f} | Error={err_pct:.3f}% | R²={r2_e:.4f}")
        assert err_pct < 1.0, f"k_E recovery error too high: {err_pct}%"
        assert r2_e > 0.99, f"Energy R² too low: {r2_e}"
        assert res["passed_gate"] is True

    # =========================================================================
    # 2. Normalized Differential Energy Bearing (theta_energy)
    # =========================================================================

    def test_energy_bearing_broadside_zero(self):
        """Verify that equal energy in both channels yields exactly 0.0° broadside angle."""
        res = KinematicAnalytics.calculate_energy_bearing(
            energy_mic1=5000.0,
            energy_mic2=5000.0,
            range_m=0.30,
            mic_distance_m=0.05
        )
        assert abs(res["theta_energy_deg"] - 0.0) < 1e-4
        assert abs(res["delta_energy_norm"] - 0.0) < 1e-6
        assert abs(res["energy_ratio_db"] - 0.0) < 1e-4

    def test_energy_bearing_angular_recovery(self):
        """
        Verify that near-field energy difference recovers target angle within first-order tolerance.
        At r = 30 cm, d = 5 cm, theta = +30°:
          r1 ≈ r + (d/2)sin(30°) = 0.3125 m ==> E1 ∝ 1/r1²
          r2 ≈ r - (d/2)sin(30°) = 0.2875 m ==> E2 ∝ 1/r2²
        """
        r, d = 0.30, 0.05
        theta_true_deg = 30.0
        th_rad = np.radians(theta_true_deg)

        r1 = np.sqrt(r**2 + (d/2.0)**2 + r * d * np.sin(th_rad))
        r2 = np.sqrt(r**2 + (d/2.0)**2 - r * d * np.sin(th_rad))

        e1 = 1.0 / (r1 ** 2)
        e2 = 1.0 / (r2 ** 2)

        res = KinematicAnalytics.calculate_energy_bearing(
            energy_mic1=e1,
            energy_mic2=e2,
            range_m=r,
            mic_distance_m=d
        )

        th_est = res["theta_energy_deg"]
        err_deg = abs(th_est - theta_true_deg)
        print(f"\n[Energy Angle Test] Target={theta_true_deg}° | Estimated={th_est:.2f}° | Error={err_deg:.2f}°")
        # First-order geometric approximation holds within ~1.2°
        assert err_deg < 1.5, f"Energy angle error too high: {err_deg}°"

    # =========================================================================
    # 3. Statistical Metrology Aggregator
    # =========================================================================

    def test_compute_metrology_statistics(self):
        """Verify statistical metrics (mean, std, SEM, MAE, RMSE, bias) against analytical ground truth."""
        # 5 samples centered around 10.0 cm with known deviations
        samples = [10.0, 10.2, 9.8, 10.1, 9.9]
        ground_truth = 10.0

        stats = KinematicAnalytics.compute_metrology_statistics(samples, ground_truth=ground_truth)

        assert stats["n_samples"] == 5
        assert abs(stats["mean"] - 10.0) < 1e-6
        assert abs(stats["bias"] - 0.0) < 1e-6
        assert stats["std"] > 0.10
        assert stats["sem"] < stats["std"]
        assert abs(stats["mae"] - 0.12) < 1e-6
        assert stats["rmse"] > 0.12
        assert stats["error_pct"] < 0.1

    def test_compute_metrology_statistics_empty_handling(self):
        """Verify graceful NaN handling on empty/invalid inputs."""
        stats = KinematicAnalytics.compute_metrology_statistics([], ground_truth=5.0)
        assert stats["n_samples"] == 0
        assert np.isnan(stats["mean"])
        assert np.isnan(stats["std"])

    # =========================================================================
    # 4. Metric Distance Inversion via Energy (r = sqrt(k_E / E))
    # =========================================================================

    def test_distance_estimator_energy_inversion(self):
        """Verify physical distance r = sqrt(k_E / E) calculation and uncertainty propagation."""
        k_e = 0.000384
        prof = AcousticProfile(
            frequencies_hz=[2660.0],
            k_values=[0.0196],
            k_energy_m1=[k_e]
        )
        estimator = DistanceEstimator(profile=prof)

        # Target r = 0.50 m (50 cm) ==> E = k_E / (0.50)² = 0.000384 / 0.25 = 0.001536
        target_r = 0.500
        sim_energy = k_e / (target_r ** 2)

        r_est, delta_r, status = estimator.estimate_distance_energy(
            energy_direct=sim_energy,
            frequency_hz=2660.0,
            channel=1
        )

        assert abs(r_est - target_r) < 1e-3, f"Energy distance mismatch: {r_est} vs {target_r}"
        assert delta_r > 0.0
        assert status == "ACTIVE_VALID"

    # =========================================================================
    # 5. Tracker Video Analysis Cross-Correlation & Synchronization
    # =========================================================================

    def test_tracker_optical_alignment_and_correlation(self):
        """Verify cross-correlation recovers simulated camera time lag and interpolates velocity."""
        pytest.importorskip("pandas")

        # 1. Synthesize FPGA 100 Hz velocity curve
        dt_fpga = 0.010
        t_fpga = np.arange(0.0, 2.0, dt_fpga)
        v_fpga = 0.60 * np.sin(np.pi * t_fpga / 2.0)

        # 2. Synthesize Tracker data with a known camera latency of +0.120 s (12 frames lag)
        sim_lag_sec = 0.120
        t_camera = np.arange(0.0, 2.3, 0.033)  # 30 FPS camera
        t_rel = t_camera - sim_lag_sec
        # Glider is stationary (v=0) before launch, then follows identical acceleration profile
        v_camera = np.where(t_rel >= 0.0, 0.60 * np.sin(np.pi * np.clip(t_rel, 0.0, 2.0) / 2.0), 0.0)

        # Write temporary Tracker CSV
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write("t,v\n")
            for t, v in zip(t_camera, v_camera):
                tmp.write(f"{t:.4f},{v:.4f}\n")

        try:
            res_sync = KinematicAnalytics.align_tracker_ground_truth(
                tracker_csv_path=tmp_path,
                t_fpga=t_fpga,
                v_fpga=v_fpga
            )

            recovered_lag = res_sync["time_offset_sec"]
            r2 = res_sync["correlation_r2"]
            mae = res_sync["mae_mps"]

            print(f"\n[Tracker Sync Test] Injected Lag={sim_lag_sec:.3f}s | Recovered Lag={recovered_lag:.3f}s | R²={r2:.4f} | MAE={mae*100:.2f}cm/s")
            assert abs(recovered_lag - sim_lag_sec) < 0.025, f"Sync lag error too high: {recovered_lag}"
            assert r2 > 0.98, f"Correlation too low: {r2}"
            assert mae < 0.03, f"MAE too high: {mae} m/s"
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # =========================================================================
    # 6. Newtonian Mechanical Simulators (Atwood & Elastic)
    # =========================================================================

    def test_atwood_glider_simulation(self):
        """Verify Modified Atwood simulation exhibits constant acceleration followed by viscous coasting."""
        t_axis = np.linspace(0.0, 3.0, 300)
        m_glider = 0.200    # 200 g
        m_hanging = 0.020   # 20 g
        fall_h = 0.40       # 40 cm
        gamma = 0.05

        v_theo, a_theo = KinematicAnalytics.simulate_atwood_glider(
            t_axis=t_axis,
            m_glider_kg=m_glider,
            m_hanging_kg=m_hanging,
            fall_height_m=fall_h,
            gamma_damping=gamma
        )

        expected_a = (0.020 / (0.200 + 0.020)) * 9.80665  # ~0.8915 m/s²
        expected_t_impact = np.sqrt(2.0 * fall_h / expected_a) # ~0.947 s

        # Phase 1: Constant acceleration
        assert abs(a_theo[10] - expected_a) < 1e-3
        # Phase 2: Post-impact damping
        idx_coast = np.where(t_axis > expected_t_impact + 0.2)[0][0]
        assert a_theo[idx_coast] < 0.0  # Decelerating
        assert v_theo[idx_coast] < np.max(v_theo)

    def test_elastic_glider_simulation(self):
        """Verify Elastic recoil simulation starts at v_0 = dx * sqrt(k/M) and decays exponentially."""
        t_axis = np.linspace(0.0, 2.0, 200)
        m_glider = 0.250
        k_spring = 40.0
        dx = 0.10

        v_theo, a_theo = KinematicAnalytics.simulate_elastic_glider(
            t_axis=t_axis,
            m_glider_kg=m_glider,
            k_spring_npm=k_spring,
            delta_x_m=dx,
            gamma_damping=0.10
        )

        expected_v0 = dx * np.sqrt(k_spring / m_glider)  # 0.10 * sqrt(160) ≈ 1.265 m/s
        assert abs(v_theo[0] - expected_v0) < 1e-3
        assert v_theo[-1] < v_theo[0]  # Monotonically decaying

    # =========================================================================
    # 7. Single-Microphone Doppler Exposure
    # =========================================================================

    def test_single_mic_doppler_exposure(self):
        """Verify process_stereo_frame outputs v_mic1, v_mic2, and v_mean_single correctly."""
        tracker = DifferentialDopplerTracker(nominal_f0_hz=2660.0)
        fs = 50000.0
        n = 2048

        # Synthesize glider moving right (+0.50 m/s):
        # Mic 1 (Left): Red-shifted (moving away)
        # Mic 2 (Right): Blue-shifted (moving toward)
        c = 343.21
        v_true = 0.50
        f1 = 2660.0 * (1.0 - v_true / c)
        f2 = 2660.0 * (1.0 + v_true / c)

        t = np.arange(n) / fs
        v_a0 = 0.10 * np.cos(2.0 * np.pi * f1 * t)
        v_a1 = 0.10 * np.cos(2.0 * np.pi * f2 * t)

        res = tracker.process_stereo_frame(v_a0, v_a1, fs=fs)

        assert "v_mic1_mps" in res
        assert "v_mic2_mps" in res
        assert "v_mean_single_mps" in res
        assert abs(res["velocity_mps"] - v_true) < 0.005
        assert abs(res["v_mic1_mps"] - v_true) < 0.005
        assert abs(res["v_mic2_mps"] - v_true) < 0.005