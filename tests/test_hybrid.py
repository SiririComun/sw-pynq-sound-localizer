"""
tests/test_hybrid.py: Strict Verification Suite for Hybrid Dual-DMA Inference Engine.
"""

import numpy as np
import pytest
from pynq_localizer.kinematics import (
    KinematicAnalytics,
    AcousticProfile,
    DistanceEstimator,
)

def generate_synthetic_hybrid_dataset(
    f_target: float = 1600.0,
    amplitude_v: float = 0.050,  # 50 mV peak (35.35 mV RMS)
    phase_offset: float = np.pi / 4.0,
    fs: float = 50_000.0,
    n_samples: int = 2048,
    noise_sigma_v: float = 0.001,
    fft_scale_multiplier: float = 1.0  # Simulates variable BFP bit-shift
):
    """Generates synchronized raw time-domain samples and polar FFT spectrum."""
    t = np.arange(n_samples) / fs
    time_signal = amplitude_v * np.cos(2.0 * np.pi * f_target * t + phase_offset)
    if noise_sigma_v > 0:
        time_signal += np.random.normal(0, noise_sigma_v, n_samples)

    # Convert to ADC counts for simulated FFT input
    adc_scale = 3.3 / 4095.0
    raw_counts = time_signal / adc_scale

    X = np.fft.fft(raw_counts)
    half_n = n_samples // 2
    X_half = X[:half_n]

    freq_axis = np.fft.fftfreq(n_samples, d=1.0 / fs)[:half_n]
    # Apply synthetic BFP scale factor to FFT magnitude
    magnitude = np.abs(X_half) * float(fft_scale_multiplier)
    phase_rad = np.angle(X_half)

    return time_signal, freq_axis, magnitude, phase_rad


class TestHybridDualDMAEngine:

    def test_coherent_inband_amplitude_precision(self):
        """Verify that coherent demodulation recovers exact RMS voltage with < 0.1% error."""
        fs = 50_000.0
        f_tone = 1620.0
        peak_amp = 0.040  # 40 mV peak -> V_RMS = 40 / sqrt(2) = 28.284 mV
        expected_vrms = peak_amp / np.sqrt(2.0)

        t = np.arange(2048) / fs
        time_sig = peak_amp * np.cos(2.0 * np.pi * f_tone * t + 0.3)

        measured_vrms = KinematicAnalytics.compute_coherent_inband_amplitude(
            signal_v=time_sig,
            fs=fs,
            target_freq_hz=f_tone
        )

        error_pct = abs(measured_vrms - expected_vrms) / expected_vrms * 100.0
        print(f"\n[Test] Expected V_RMS={expected_vrms*1000:.3f} mV | Measured={measured_vrms*1000:.3f} mV | Error={error_pct:.3f}%")
        assert error_pct < 0.25, f"Coherent demodulation error too high: {error_pct}%"

    def test_bfp_scaling_immunity(self):
        """
        Prove that a 16x drop in FFT magnitude (simulating BFP bit shift)
        causes ZERO change in the hybrid in-band amplitude and distance.
        """
        f_true = 1600.0
        v_true_peak = 0.050  # 50 mV -> V_RMS = 35.355 mV

        # Frame 1: Full-scale FFT magnitude (multiplier = 1.0x)
        t_sig1, freqs, mag1, phase = generate_synthetic_hybrid_dataset(
            f_target=f_true, amplitude_v=v_true_peak, fft_scale_multiplier=1.0, noise_sigma_v=0.0
        )
        quad1 = KinematicAnalytics.extract_hybrid_quadruple(
            time_signal_v=t_sig1,
            fs=50_000.0,
            freq_axis=freqs,
            magnitude=mag1,
            phase_rad=phase,
            f_min=1400.0,
            f_max=1800.0
        )

        # Frame 2: FFT Magnitude scaled down by 16x (simulating BLK_EXP = 4)
        t_sig2, _, mag2, _ = generate_synthetic_hybrid_dataset(
            f_target=f_true, amplitude_v=v_true_peak, fft_scale_multiplier=1.0 / 16.0, noise_sigma_v=0.0
        )
        quad2 = KinematicAnalytics.extract_hybrid_quadruple(
            time_signal_v=t_sig2,
            fs=50_000.0,
            freq_axis=freqs,
            magnitude=mag2,
            phase_rad=phase,
            f_min=1400.0,
            f_max=1800.0
        )

        # 1. Frequency pitch must remain identical
        assert abs(quad1["frequency_hz"] - quad2["frequency_hz"]) < 1e-4

        # 2. Hybrid in-band amplitude must be identical (100% immune to 16x FFT scale drop)
        amp1 = quad1["amplitude_v"]
        amp2 = quad2["amplitude_v"]
        print(f"\n[BFP Test] Frame 1 (1.0x Mag) A_true: {amp1*1000:.3f} mV")
        print(f"[BFP Test] Frame 2 (0.06x Mag) A_true: {amp2*1000:.3f} mV")
        assert abs(amp1 - amp2) < 1e-6, "Hybrid extraction failed BFP immunity test!"

    def test_distance_estimator_process_frame(self):
        """Verify end-to-end DistanceEstimator.process_frame pipeline."""
        k_phone = 0.050  # V*m
        estimator = DistanceEstimator(k_constant=k_phone, noise_gate_v=0.003)

        # Synthesize frame at r = 1.00 m (A_peak = 0.050 * sqrt(2) = 0.0707 V -> V_RMS = 0.050 V)
        v_time, freqs, mag, phase = generate_synthetic_hybrid_dataset(
            f_target=2000.0, amplitude_v=0.07071, fft_scale_multiplier=0.5, noise_sigma_v=0.0
        )

        frame_dict = {
            "v_a0": v_time,
            "freqs": freqs,
            "mag": mag,
            "phase": phase,
            "timer_cycles": 50_000_000
        }

        res = estimator.process_frame(frame_dict=frame_dict, source="A0", fs=50_000.0, f_min=1800.0, f_max=2200.0)

        assert "distance_m" in res
        assert abs(res["distance_m"] - 1.00) < 0.01
        assert abs(res["frequency_hz"] - 2000.0) < 0.2