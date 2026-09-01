"""
tests/test_quadruple.py: Strict Unit Verification Suite for Quadruple (f, A, φ, t) Engine.
"""

import numpy as np
import pytest
from pynq_localizer.kinematics import KinematicAnalytics

def generate_synthetic_polar_frame(
    f_target: float = 1000.0,
    amplitude: float = 0.5,
    phase_offset: float = np.pi / 4.0,
    fs: float = 50_000.0,
    n_fft: int = 2048,
    noise_sigma: float = 0.0
):
    """Generates a synthetic complex spectrum simulating hardware FFT & CORDIC outputs."""
    t = np.arange(n_fft) / fs
    # Synthesize tone: x(t) = A * cos(2*pi*f*t + phi)
    signal = amplitude * np.cos(2.0 * np.pi * f_target * t + phase_offset)
    if noise_sigma > 0:
        signal += np.random.normal(0, noise_sigma, n_fft)

    # Complex FFT
    X = np.fft.fft(signal)
    half_n = n_fft // 2
    X_half = X[:half_n]

    freq_axis = np.fft.fftfreq(n_fft, d=1.0 / fs)[:half_n]
    magnitude = np.abs(X_half)
    phase_rad = np.angle(X_half)

    return freq_axis, magnitude, phase_rad


class TestQuadrupleEngine:

    def test_sub_hertz_pitch_accuracy(self):
        """Verify sub-Hertz fundamental pitch tracking on arbitrary frequencies."""
        f_true = 1012.3
        freqs, mags, phases = generate_synthetic_polar_frame(
            f_target=f_true, amplitude=0.4, phase_offset=0.0
        )

        quad = KinematicAnalytics.extract_quadruple(
            freq_axis=freqs,
            magnitude=mags,
            phase_rad=phases,
            f_min=800.0,
            f_max=1200.0,
            timer_cycles=100_000_000
        )

        assert quad["is_valid"] is True
        f_est = quad["frequency_hz"]
        freq_error = abs(f_est - f_true)
        print(f"\n[Test] True f={f_true} Hz | Estimated f={f_est:.3f} Hz | Error={freq_error:.3f} Hz")
        # Sub-Hertz precision threshold (< 0.25 Hz error with N=2048 at 50 kSPS)
        assert freq_error < 0.25, f"Pitch error too high: {freq_error} Hz"

    def test_phase_extraction_accuracy(self):
        """Verify phase angle recovery across multiple quadrants."""
        test_phases = [0.0, np.pi / 6.0, np.pi / 4.0, np.pi / 2.0, -np.pi / 3.0, -3.0 * np.pi / 4.0]
        # Integer bin tone: bin 82 * (50000 / 2048) = 2001.953125 Hz
        f_tone = 82.0 * (50000.0 / 2048.0)

        for phi_true in test_phases:
            freqs, mags, phases = generate_synthetic_polar_frame(
                f_target=f_tone, amplitude=0.5, phase_offset=phi_true, noise_sigma=0.0
            )

            quad = KinematicAnalytics.extract_quadruple(
                freq_axis=freqs,
                magnitude=mags,
                phase_rad=phases,
                f_min=1800.0,
                f_max=2200.0,
                timer_cycles=0
            )

            phi_est = quad["phase_rad"]
            phase_error = np.abs(np.arctan2(np.sin(phi_est - phi_true), np.cos(phi_est - phi_true)))
            assert phase_error < 0.01, f"Phase recovery error for {phi_true} rad: {phase_error} rad"

    def test_spectral_range_gating(self):
        """Verify that out-of-band energy is completely rejected."""
        freqs, mags, phases = generate_synthetic_polar_frame(
            f_target=3500.0, amplitude=0.8, phase_offset=0.0, noise_sigma=0.0
        )

        quad = KinematicAnalytics.extract_quadruple(
            freq_axis=freqs,
            magnitude=mags,
            phase_rad=phases,
            f_min=500.0,
            f_max=1500.0,
            timer_cycles=50_000_000
        )

        assert quad["amplitude_v"] < 1e-4

    def test_phase_velocity_and_sqi(self):
        """Verify instantaneous phase derivative f_phase = (1/2pi) * dPhi/dt and SQI metric."""
        f_target = 15.0  # 15 Hz smooth Doppler phase modulation
        dt_frame = 0.010  # 10 ms per frame (100 Hz frame rate -> Nyquist satisfied for 15 Hz)
        n_frames = 50

        times = np.arange(n_frames) * dt_frame
        phases_wrapped = (2.0 * np.pi * f_target * times + np.pi) % (2.0 * np.pi) - np.pi

        unwrapped_phi, f_inst, mean_sqi = KinematicAnalytics.compute_phase_velocity(
            phase_history=phases_wrapped,
            time_history=times,
            f_expected=f_target,
            f_tol=2.0
        )

        assert len(f_inst) == n_frames - 1
        assert np.allclose(f_inst, f_target, atol=1e-2)
        assert mean_sqi > 0.99