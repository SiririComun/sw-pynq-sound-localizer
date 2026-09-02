"""
tests/test_multisource.py: Verification Suite for MultiSourceTracker & Harmonic Rejection.
"""

import numpy as np
import pytest
from pynq_localizer.kinematics import MultiSourceTracker

def generate_multi_tone_polar_frame(
    tones: list,
    fs: float = 50_000.0,
    n_fft: int = 2048,
    noise_sigma_v: float = 0.0
):
    """
    Synthesizes a complex multi-source spectrum matching 12-bit FPGA ADC integer counts.
    :param tones: List of dicts with 'f', 'amp' (in Volts), 'phase', and optional 'h2_ratio', 'h3_ratio'.
    """
    t = np.arange(n_fft) / fs
    adc_scale = 3.3 / 4095.0  # Volts per ADC count
    signal_v = np.zeros(n_fft, dtype=np.float64)

    for tone in tones:
        f = tone["f"]
        a = tone["amp"]
        phi = tone.get("phase", 0.0)
        # Fundamental tone
        signal_v += a * np.cos(2.0 * np.pi * f * t + phi)
        # 2nd harmonic non-linearity
        if "h2_ratio" in tone:
            signal_v += (a * tone["h2_ratio"]) * np.cos(2.0 * np.pi * (2.0 * f) * t + 2.0 * phi)
        # 3rd harmonic non-linearity
        if "h3_ratio" in tone:
            signal_v += (a * tone["h3_ratio"]) * np.cos(2.0 * np.pi * (3.0 * f) * t + 3.0 * phi)

    if noise_sigma_v > 0:
        signal_v += np.random.normal(0, noise_sigma_v, n_fft)

    # Convert physical Volts to raw 12-bit ADC integer counts (matching hardware FPGA input)
    raw_counts = signal_v / adc_scale

    X = np.fft.fft(raw_counts)
    half_n = n_fft // 2
    X_half = X[:half_n]

    freq_axis = np.fft.fftfreq(n_fft, d=1.0 / fs)[:half_n]
    magnitude = np.abs(X_half)
    phase_rad = np.angle(X_half)

    return freq_axis, magnitude, phase_rad


class TestMultiSourceTracker:

    def test_concurrent_3_source_tracking(self):
        """Verify independent quadruple extraction for 3 concurrent physical sources."""
        tones = [
            {"f": 1000.0, "amp": 0.50, "phase": 0.0},
            {"f": 2500.0, "amp": 0.35, "phase": np.pi / 3.0},
            {"f": 4200.0, "amp": 0.20, "phase": -np.pi / 4.0},
        ]
        freqs, mags, phases = generate_multi_tone_polar_frame(tones)

        bands = {
            "Source_1": (850.0, 1150.0),
            "Source_2": (2350.0, 2650.0),
            "Source_3": (4050.0, 4350.0),
        }
        tracker = MultiSourceTracker(source_bands=bands, noise_gate_v=0.010)
        res = tracker.process_spectral_frame(freqs, mags, phases, timer_cycles=100_000_000)

        s1 = res["sources"]["Source_1"]
        s2 = res["sources"]["Source_2"]
        s3 = res["sources"]["Source_3"]

        # 1. Pitch Accuracy (< 2.0 Hz across multi-tone spectrum)
        assert abs(s1["frequency_hz"] - 1000.0) < 0.2
        assert abs(s2["frequency_hz"] - 2500.0) < 0.5
        assert abs(s3["frequency_hz"] - 4200.0) < 2.0

        # 2. Phase Angle Accuracy on Unit Circle
        err1 = np.abs(np.arctan2(np.sin(s1["phase_rad"] - 0.0), np.cos(s1["phase_rad"] - 0.0)))
        err2 = np.abs(np.arctan2(np.sin(s2["phase_rad"] - (np.pi / 3.0)), np.cos(s2["phase_rad"] - (np.pi / 3.0))))
        err3 = np.abs(np.arctan2(np.sin(s3["phase_rad"] - (-np.pi / 4.0)), np.cos(s3["phase_rad"] - (-np.pi / 4.0))))

        assert err1 < 0.08, f"S1 Phase error: {err1}"
        assert err2 < 0.08, f"S2 Phase error: {err2}"
        assert err3 < 0.25, f"S3 Phase error: {err3}" # Multi-tone sidelobe pulling bound

        # 3. All sources active (> 10 mV noise gate)
        assert s1["is_active"] and s2["is_active"] and s3["is_active"]

    def test_harmonic_leakage_cancellation(self):
        """
        Verify that 2f0 harmonic leakage from Source 1 (1000 Hz) into Source 2 (2000 Hz)
        is suppressed when Source 2 is physically silent.
        """
        # Source 1 is ON (1000 Hz @ 0.6V) with 5% 2nd harmonic distortion (2000 Hz)
        # Source 2 (2000 Hz) is OFF (0.0V)
        tones = [
            {"f": 1000.0, "amp": 0.60, "phase": 0.0, "h2_ratio": 0.05},
        ]
        freqs, mags, phases = generate_multi_tone_polar_frame(tones)

        bands = {
            "Speaker_1000": (850.0, 1150.0),
            "Speaker_2000": (1850.0, 2150.0),
        }

        # Tracker with harmonic rejection enabled
        tracker = MultiSourceTracker(
            source_bands=bands,
            harmonic_rejection=True,
            h2_coeff=0.03,
            noise_gate_v=0.020
        )
        res = tracker.process_spectral_frame(freqs, mags, phases, timer_cycles=50_000_000)

        s1 = res["sources"]["Speaker_1000"]
        s2 = res["sources"]["Speaker_2000"]

        # Speaker 1000 is detected and active
        assert s1["is_active"] is True
        assert abs(s1["frequency_hz"] - 1000.0) < 0.2

        # Speaker 2000 leakage should be cancelled and below noise gate
        print(f"\n[Test] Speaker 2000 Cleaned Amp: {s2['amplitude_v']*1000:.2f} mV (Active={s2['is_active']}, SIR={s2['sir_db']:.1f} dB)")
        assert s2["is_active"] is False

    def test_trajectory_accumulation(self):
        """Verify rolling time-series tracking over multiple frames."""
        bands = {"Tone_A": (900.0, 1100.0)}
        tracker = MultiSourceTracker(source_bands=bands, noise_gate_v=0.010)

        for i in range(10):
            freqs, mags, phases = generate_multi_tone_polar_frame(
                [{"f": 1000.0 + i * 2.0, "amp": 0.30}], noise_sigma_v=0.0
            )
            tracker.process_spectral_frame(freqs, mags, phases, timer_cycles=i * 4_000_000)

        traj = tracker.get_source_trajectory("Tone_A")
        assert len(traj["t"]) == 10
        assert len(traj["f"]) == 10
        # Check ascending pitch trajectory
        assert traj["f"][-1] > traj["f"][0]