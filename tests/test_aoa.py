"""
tests/test_aoa.py: Strict Unit Verification Suite for Angle of Arrival (AoA) Interferometry Engine.
"""

import json
from pathlib import Path
from typing import Optional
import numpy as np
import pytest

from pynq_localizer.kinematics import KinematicAnalytics, AngleOfArrivalEstimator

def generate_synthetic_aoa_stereo_frame(
    theta_deg: float,
    f0: float = 2609.73,
    mic_distance_m: float = 0.05,
    amplitude_v: float = 0.100,
    fs: float = 50000.0,
    n_samples: int = 2048,
    temperature_c: float = 20.0,
    snr_db: Optional[float] = None
):
    """
    Synthesizes a dual-channel time stream for an acoustic plane wave incident at angle theta_deg.
    Sign convention:
      - theta = 0 deg => Broadside (wavefront arrives at A0 and A1 at the same instant)
      - theta > 0 deg => Wavefront reaches Mic 2 (A1) before Mic 1 (A0) (Right)
      - theta < 0 deg => Wavefront reaches Mic 1 (A0) before Mic 2 (A1) (Left)
    """
    c = KinematicAnalytics.speed_of_sound(temperature_c)
    theta_rad = np.radians(theta_deg)

    # Time delay: tau = (d * sin(theta)) / c
    tau_sec = (mic_distance_m * np.sin(theta_rad)) / c

    t = np.arange(n_samples) / fs

    # Mic 1 (A0) is reference phase; Mic 2 (A1) leads by tau
    v_a0 = amplitude_v * np.cos(2.0 * np.pi * f0 * t)
    v_a1 = amplitude_v * np.cos(2.0 * np.pi * f0 * (t + tau_sec))

    if snr_db is not None:
        noise_sigma = amplitude_v / (10.0 ** (snr_db / 20.0)) / np.sqrt(2.0)
        v_a0 += np.random.normal(0, noise_sigma, n_samples)
        v_a1 += np.random.normal(0, noise_sigma, n_samples)

    return v_a0, v_a1

class TestAngleOfArrivalEngine:

    def test_broadside_zero_angle_recovery(self):
        """Verify that an incident plane wave at 0 deg produces exactly 0 deg with Delta phi = 0."""
        v0, v1 = generate_synthetic_aoa_stereo_frame(theta_deg=0.0)
        aoa = AngleOfArrivalEstimator(mic_distance_m=0.05, target_freq_hz=2609.73)
        res = aoa.estimate_angle(v0, v1, fs=50000.0)

        assert abs(res["theta_deg"] - 0.0) < 1e-3, f"Broadside failed: {res['theta_deg']}"
        assert abs(res["delta_phi_rad"] - 0.0) < 1e-4, f"Delta phi not zero: {res['delta_phi_rad']}"
        assert res["status"] == "ACTIVE_VALID"

    def test_angular_sweep_grid_accuracy(self):
        """
        Verify angular recovery across an incident grid from -60 deg to +60 deg
        across multiple carrier frequencies (1000 Hz, 2000 Hz, 2609.73 Hz).
        With Hann-windowed coherent projection, error must remain < 0.02 deg!
        """
        test_angles = [-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0]
        test_frequencies = [1000.0, 2000.0, 2609.73]
        d_mic = 0.05

        for f0 in test_frequencies:
            aoa = AngleOfArrivalEstimator(mic_distance_m=d_mic, target_freq_hz=f0)
            for theta_true in test_angles:
                v0, v1 = generate_synthetic_aoa_stereo_frame(
                    theta_deg=theta_true, f0=f0, mic_distance_m=d_mic
                )
                res = aoa.estimate_angle(v0, v1, fs=50000.0)
                theta_est = res["theta_deg"]
                err = abs(theta_est - theta_true)

                # High-precision threshold: < 0.02 deg
                assert err < 0.02, (
                    f"AoA grid error too high: True={theta_true}°, "
                    f"Est={theta_est:.4f}°, Err={err:.4f}° (f0={f0}Hz)"
                )
                assert res["status"] == "ACTIVE_VALID"

    def test_noise_robustness(self):
        """Verify angle recovery remains accurate under additive Gaussian noise down to 20 dB SNR."""
        theta_true = 25.0
        v0, v1 = generate_synthetic_aoa_stereo_frame(
            theta_deg=theta_true, f0=2609.73, snr_db=20.0
        )
        aoa = AngleOfArrivalEstimator(mic_distance_m=0.05, target_freq_hz=2609.73)
        res = aoa.estimate_angle(v0, v1, fs=50000.0)

        err = abs(res["theta_deg"] - theta_true)
        print(f"\n[AoA Noise Test] True={theta_true}° | Est={res['theta_deg']:.2f}° | Err={err:.2f}°")
        assert err < 0.80, f"Noisy AoA error too high: {err}°"
        assert res["status"] == "ACTIVE_VALID"

    def test_squelch_silence_gating(self):
        """Verify that amplitudes below the noise gate return status SILENCE and NaN angle."""
        aoa = AngleOfArrivalEstimator(mic_distance_m=0.05, target_freq_hz=2609.73, noise_gate_v=0.020)
        # Signal below noise gate (5 mV)
        v0, v1 = generate_synthetic_aoa_stereo_frame(theta_deg=30.0, amplitude_v=0.005)
        res = aoa.estimate_angle(v0, v1, fs=50000.0)

        assert np.isnan(res["theta_deg"])
        assert res["status"] == "SILENCE"

    def test_spatial_aliasing_boundary_flag(self):
        """Verify that microphone spacing d > lambda/2 sets OUT_OF_BOUNDS_SPATIAL_ALIASING status."""
        # For f0 = 4000 Hz, lambda/2 = 343.2 / (2 * 4000) = 0.0429 m (4.29 cm)
        # Spacing d = 0.06 m (6 cm) exceeds the spatial aliasing limit
        aoa = AngleOfArrivalEstimator(mic_distance_m=0.060, target_freq_hz=4000.0)
        v0, v1 = generate_synthetic_aoa_stereo_frame(theta_deg=10.0, f0=4000.0, mic_distance_m=0.060)
        res = aoa.estimate_angle(v0, v1, fs=50000.0)

        assert res["status"] == "OUT_OF_BOUNDS_SPATIAL_ALIASING"

    def test_profile_loading_integration(self):
        """Verify that AngleOfArrivalEstimator correctly loads f0 from the certified buzzer profile."""
        profile_path = Path("profiles/active_buzzer_2610hz.json")
        assert profile_path.exists(), "Profile file missing!"

        aoa = AngleOfArrivalEstimator(mic_distance_m=0.05, profile=profile_path)
        assert abs(aoa.target_freq_hz - 2609.73) < 0.1

        v0, v1 = generate_synthetic_aoa_stereo_frame(theta_deg=-35.0, f0=aoa.target_freq_hz)
        res = aoa.estimate_angle(v0, v1, fs=50000.0)

        assert abs(res["theta_deg"] - (-35.0)) < 0.02
        assert res["status"] == "ACTIVE_VALID"