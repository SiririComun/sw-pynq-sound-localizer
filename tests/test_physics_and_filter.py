"""
tests/test_physics_and_filter.py: Strict Verification Suite for Spectral Filtering,
Hermitian Symmetric IFFT Reconstruction, and Physical Acoustic Distance Laws.
"""

import unittest
import numpy as np

from pynq_localizer.kinematics import KinematicAnalytics
from pynq_localizer.spectral_mask import SpectralMaskDriver


class MockMMIO:
    """Mock MMIO memory controller for testing hardware register interactions."""

    def __init__(self):
        self.regs = {
            0x00: 0,           # REG_CTRL
            0x04: 0,           # REG_BIN_START
            0x08: 10,          # REG_BIN_STOP
            0x0C: 1024,        # REG_FFT_LEN
            0x10: 0,           # REG_STATUS
        }

    def read(self, offset: int) -> int:
        return self.regs.get(offset, 0)

    def write(self, offset: int, val: int):
        self.regs[offset] = int(val)


class TestPhysicsAndFilterVerification(unittest.TestCase):

    def setUp(self):
        self.fs = 50_000.0   # 50 kSPS
        self.fft_len = 1024  # Delta f = 48.828 Hz
        self.mock_mmio = MockMMIO()
        self.mask_driver = SpectralMaskDriver(self.mock_mmio, fs=self.fs, fft_len=self.fft_len)

    # =========================================================================
    # Gate 1: Angular Frequency to Discrete Bin Translation
    # =========================================================================

    def test_angular_frequency_bin_mapping(self):
        """Validates exact mapping from continuous (omega +- Delta omega) to hardware bins."""
        f0 = 1000.0
        delta_f = 100.0
        omega_0 = f0 * 2.0 * np.pi
        delta_omega = delta_f * 2.0 * np.pi

        bins = KinematicAnalytics.calculate_filter_bins(
            omega_0=omega_0,
            delta_omega=delta_omega,
            fs=self.fs,
            fft_len=self.fft_len
        )

        df = self.fs / self.fft_len
        expected_k_start = int(round(900.0 / df))
        expected_k_stop = int(round(1100.0 / df))

        self.assertEqual(bins["k_start"], expected_k_start)
        self.assertEqual(bins["k_stop"], expected_k_stop)
        self.assertAlmostEqual(bins["f_center_hz"], 1000.0, places=2)
        self.assertAlmostEqual(bins["delta_f_hz"], 100.0, places=2)
        self.assertTrue(bins["k_start"] < bins["k_stop"])
        self.assertTrue(bins["k_stop"] <= self.fft_len // 2)

    # =========================================================================
    # Gate 2: SpectralMaskDriver Register Mapping
    # =========================================================================

    def test_spectral_mask_driver_modes(self):
        """Validates that driver methods write exact register bit patterns."""
        # 1. Bandpass Mode: f = 1200 Hz +- 150 Hz
        self.mask_driver.set_bandpass(center_hz=1200.0, delta_hz=150.0)
        self.assertTrue(self.mask_driver.is_enabled)
        self.assertEqual(self.mask_driver.mode, "bandpass")
        self.assertEqual(self.mock_mmio.read(0x00) & 0x07, 0b101)

        # 2. Lowpass Mode: fc = 500 Hz
        self.mask_driver.set_lowpass(cutoff_hz=500.0)
        self.assertEqual(self.mask_driver.mode, "lowpass")
        self.assertEqual(self.mock_mmio.read(0x04), 0)

        # 3. Highpass Mode: fc = 2000 Hz
        self.mask_driver.set_highpass(cutoff_hz=2000.0)
        self.assertEqual(self.mask_driver.mode, "highpass")
        self.assertEqual(self.mock_mmio.read(0x08), self.fft_len // 2)

        # 4. Bypass Mode
        self.mask_driver.bypass()
        self.assertFalse(self.mask_driver.is_enabled)
        self.assertEqual(self.mask_driver.mode, "bypass")

    # =========================================================================
    # Gate 3A: Coherent Exact-Bin Mathematical Fidelity (Zero-Leakage Benchmark)
    # =========================================================================

    def test_coherent_spectral_mask_exact_fidelity(self):
        """
        Validates exact 1:1 mathematical reconstruction when frequencies align
        with discrete FFT bins (f = k * Delta f), eliminating spectral leakage.
        Expected amplitude error: < 0.01%.
        """
        n = self.fft_len
        df = self.fs / n
        t = np.arange(n) / self.fs

        # Bin-centered tones: Bin 20 (976.5625 Hz) and Bin 72 (3515.625 Hz)
        f1 = 20 * df
        f2 = 72 * df
        a1, a2 = 1.00, 1.00

        s_raw = a1 * np.sin(2.0 * np.pi * f1 * t) + a2 * np.sin(2.0 * np.pi * f2 * t)

        # 1. Forward FFT -> Mask -> IFFT
        spec_complex = np.fft.fft(s_raw)
        k_axis = np.arange(n)
        k_eff = np.minimum(k_axis, n - k_axis)

        # Bandpass bins: [18, 23] around Bin 20
        pass_mask = (k_eff >= 18) & (k_eff <= 23)
        spec_masked = np.where(pass_mask, spec_complex, 0.0 + 0.0j)
        s_reconstructed = np.fft.ifft(spec_masked)

        # 2. Verify zero imaginary leakage
        self.assertLess(np.max(np.abs(np.imag(s_reconstructed))), 1e-12)

        # 3. Verify exact 1:1 amplitude fidelity (Error < 0.01%)
        s_filt = np.real(s_reconstructed)
        target_only = a1 * np.sin(2.0 * np.pi * f1 * t)
        rms_target = np.sqrt(np.mean(target_only ** 2))
        rms_filtered = np.sqrt(np.mean(s_filt ** 2))

        amp_error_pct = abs(rms_filtered - rms_target) / rms_target * 100.0
        self.assertLess(amp_error_pct, 0.01, f"Exact math fidelity error: {amp_error_pct:.4f}%")

    # =========================================================================
    # Gate 3B: Non-Coherent Real-World Signal Isolation (Arbitrary Frequencies)
    # =========================================================================

    def test_non_coherent_real_world_signal_isolation(self):
        """
        Validates real-world signals with arbitrary non-integer frequencies (1000 Hz & 3500 Hz):
          • Verifies stopband rejection of interferer > 40 dB
          • Verifies passband amplitude preservation within realistic leakage bounds (< 4.0%)
        """
        n = self.fft_len
        df = self.fs / n
        t = np.arange(n) / self.fs

        f1, a1 = 1000.0, 1.00  # Target (Bin 20.48)
        f2, a2 = 3500.0, 1.00  # Interferer (Bin 71.68)

        s_raw = a1 * np.sin(2.0 * np.pi * f1 * t) + a2 * np.sin(2.0 * np.pi * f2 * t)

        spec_complex = np.fft.fft(s_raw)
        k_axis = np.arange(n)
        k_eff = np.minimum(k_axis, n - k_axis)

        k_start = int(round(900.0 / df))
        k_stop = int(round(1100.0 / df))

        pass_mask = (k_eff >= k_start) & (k_eff <= k_stop)
        spec_masked = np.where(pass_mask, spec_complex, 0.0 + 0.0j)
        s_filt = np.real(np.fft.ifft(spec_masked))

        # 1. Amplitude preservation under spectral leakage (< 4.0% error)
        target_only = a1 * np.sin(2.0 * np.pi * f1 * t)
        rms_target = np.sqrt(np.mean(target_only ** 2))
        rms_filtered = np.sqrt(np.mean(s_filt ** 2))

        amp_error_pct = abs(rms_filtered - rms_target) / rms_target * 100.0
        self.assertLess(amp_error_pct, 4.0, f"Real-world amplitude error exceeded 4%: {amp_error_pct:.2f}%")

        # 2. Stopband isolation of 3500 Hz interferer (> 40 dB)
        metrics = KinematicAnalytics.calculate_filter_isolation_metrics(
            raw_signal=s_raw,
            filtered_signal=s_filt,
            fs=self.fs,
            target_band_hz=(950.0, 1050.0),
            interferer_band_hz=(3400.0, 3600.0)
        )
        self.assertGreater(metrics["stopband_rejection_db"], 40.0)

    # =========================================================================
    # Gate 4: Physical Inverse-Distance Law (1/r and 1/r^2)
    # =========================================================================

    def test_inverse_distance_power_law_fit(self):
        """
        Validates that fit_inverse_distance_law accurately recovers:
          • Pressure Exponent: n = 1.00 +- 0.05
          • Intensity Exponent: 2n = 2.00 +- 0.10
          • Goodness of Fit: R^2 > 0.99
        """
        distances = np.array([0.25, 0.50, 0.75, 1.00, 1.50, 2.00])
        ideal_amplitude = 1.5

        voltages_rms = ideal_amplitude / distances

        np.random.seed(42)
        voltages_noisy = voltages_rms * (1.0 + np.random.normal(0, 0.01, size=len(distances)))

        fit_result = KinematicAnalytics.fit_inverse_distance_law(distances, voltages_noisy)

        self.assertAlmostEqual(fit_result["measured_exponent_n"], 1.00, delta=0.03)
        self.assertAlmostEqual(fit_result["intensity_exponent_2n"], 2.00, delta=0.06)
        self.assertGreater(fit_result["r_squared"], 0.99)
        self.assertLess(fit_result["error_pct_from_ideal_1_over_r"], 3.0)


if __name__ == "__main__":
    unittest.main()