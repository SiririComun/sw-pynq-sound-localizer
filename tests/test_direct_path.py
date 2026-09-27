"""
tests/test_direct_path.py: Strict Unit Verification Suite for Quasi-Anechoic Direct-Pulse
Metrology, Hardware Energy Accumulator, and Zero-Intercept 1/r Regressions.
"""

import math
import numpy as np
import pytest
from pynq_localizer.hw_trigger import HardwareTrigger
from pynq_localizer.kinematics import KinematicAnalytics, DirectPulseCalibrationProtocol


class MockTriggerMMIO:
    """Mock MMIO simulating axis_trigger_unit registers for unit testing without PYNQ board."""
    def __init__(self):
        self.regs = {
            0x00: 0x00000003,  # REG_CONTROL
            0x04: 0x00000000,  # REG_STATUS
            0x08: 0x00000800,  # REG_THRESHOLD
            0x0C: 5000000,     # REG_TIMEOUT
            0x10: 0x00000010,  # REG_HYSTERESIS
            0x14: 0x00000001,  # REG_DECIMATION (M=10)
            0x18: 0x0000010A,  # REG_FFT_CONFIG
            0x1C: 0x00000800,  # REG_PACKET_SIZE
            0x20: 500000,      # REG_PULSE_WIDTH
            0x24: 0,           # REG_MIC1_TOA
            0x28: 0,           # REG_MIC2_TOA
            0x2C: 0x75300200,  # REG_TOA_CONFIG
            0x30: 0x80008000,  # REG_MIC_DC_REF (1.65V DC)
            0x34: 0,           # REG_MIC1_DIRECT_ENERGY
            0x38: 0,           # REG_MIC2_DIRECT_ENERGY
            0x3C: 0x00000240,  # REG_GATE_CONFIG (576 samples default)
        }

    def read(self, offset: int) -> int:
        return self.regs.get(offset, 0)

    def write(self, offset: int, value: int):
        self.regs[offset] = int(value) & 0xFFFFFFFF


class TestDirectPathMetrology:

    def test_direct_energy_voltage_conversion(self):
        """
        Verify that 32-bit hardware energy counts convert to physical RMS Volts
        with < 0.1% error across multiple amplitude levels and gate lengths.
        """
        mmio = MockTriggerMMIO()
        trig = HardwareTrigger(mmio)

        adc_scale = 3.3 / 4095.0  # Volts per count (~0.80586 mV/count)
        test_amplitudes_peak_v = [0.020, 0.050, 0.100, 0.250]  # 20 mV to 250 mV
        test_gate_lengths = [192, 384, 576]                    # 1, 2, 3 cycles

        for v_peak in test_amplitudes_peak_v:
            v_expected_rms = v_peak / math.sqrt(2.0)
            a_counts = v_peak / adc_scale

            for n_samples in test_gate_lengths:
                trig.set_direct_gate_samples(n_samples)
                assert trig.get_direct_gate_samples() == n_samples

                # Theoretical discrete squared sum over n_samples: n_samples * (A^2 / 2)
                mean_sq_dev = (a_counts ** 2) / 2.0
                simulated_energy = int(round(mean_sq_dev * n_samples))

                # Inject into hardware registers
                mmio.regs[0x34] = simulated_energy
                mmio.regs[0x38] = simulated_energy
                mmio.regs[0x04] |= (1 << 7) | (1 << 8)  # Set gate done bits

                assert trig.is_mic1_gate_done is True
                assert trig.is_mic2_gate_done is True
                assert trig.is_direct_gate_done is True

                v1_rms, v2_rms = trig.get_direct_rms_voltage()

                err1_pct = abs(v1_rms - v_expected_rms) / v_expected_rms * 100.0
                err2_pct = abs(v2_rms - v_expected_rms) / v_expected_rms * 100.0

                print(
                    f"\n[Test] V_peak={v_peak*1000:.0f}mV, N={n_samples} | "
                    f"Expected RMS={v_expected_rms*1000:.3f}mV | "
                    f"Converted RMS={v1_rms*1000:.3f}mV | "
                    f"Error={err1_pct:.4f}%"
                )

                assert err1_pct < 0.10, f"Mic 1 voltage conversion error too high: {err1_pct}%"
                assert err2_pct < 0.10, f"Mic 2 voltage conversion error too high: {err2_pct}%"
    
    def test_integer_cycle_orthogonality(self):
        """
        Verify that integrating over exact integer cycles (K = 1, 2, 3, 4) achieves
        pure Fourier orthogonality with zero spectral leakage and exact phase recovery
        across arbitrary carrier phase offsets, while fractional cuts suffer severe bias.
        """
        fs = 500_000.0
        f0 = 2609.73
        v_peak = 0.080  # 80 mV peak -> V_RMS = 56.5685 mV
        v_expected_rms = v_peak / np.sqrt(2.0)

        test_phases = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0, -np.pi / 3.0]
        test_cycles = [1, 2, 3, 4]

        # 1. Verify exact integer cycle orthogonality across all phases
        for k_cycles in test_cycles:
            for phi_true in test_phases:
                samples_per_cycle = int(round(fs / f0))
                n_samples = k_cycles * samples_per_cycle

                t = np.arange(n_samples) / fs
                tone = v_peak * np.cos(2.0 * np.pi * f0 * t + phi_true)

                res = KinematicAnalytics.extract_gated_direct_fourier(
                    signal_v=tone,
                    fs=fs,
                    f0=f0,
                    n_cycles=k_cycles,
                    start_idx=0,
                    remove_dc=True
                )

                assert res["is_valid"] is True
                assert res["n_samples_gated"] == n_samples

                err_amp_pct = abs(res["amplitude_v"] - v_expected_rms) / v_expected_rms * 100.0
                err_phi = abs(np.arctan2(np.sin(res["phase_rad"] - phi_true), np.cos(res["phase_rad"] - phi_true)))

                # Exact integer cycles achieve < 0.30% error (bounded by 191.59 -> 192 integer rounding)
                assert err_amp_pct < 0.30, (
                    f"Orthogonality failed: K={k_cycles}, phi={phi_true:.2f} rad | "
                    f"Amp Err={err_amp_pct:.3f}%"
                )
                assert err_phi < 0.02, (
                    f"Phase recovery failed: K={k_cycles}, phi={phi_true:.2f} rad | "
                    f"Phi Err={err_phi:.4f} rad"
                )

        # 2. Contrast with non-integer fractional truncation (e.g. 2.4 cycles)
        # Demonstrates the spectral leakage bias when orthogonality is violated
        fractional_samples = int(2.4 * (fs / f0))
        t_frac = np.arange(fractional_samples) / fs
        # Worst-case phase for rectangular leakage
        tone_leaking = v_peak * np.cos(2.0 * np.pi * f0 * t_frac + np.pi / 4.0)

        # Unwindowed projection over non-integer window
        t_local = np.arange(fractional_samples) / fs
        phasor = np.exp(-2.0j * np.pi * f0 * t_local)
        x_leaking = (2.0 / fractional_samples) * np.dot(tone_leaking - np.mean(tone_leaking), phasor)
        v_rms_leaking = np.abs(x_leaking) / np.sqrt(2.0)

        err_leak_pct = abs(v_rms_leaking - v_expected_rms) / v_expected_rms * 100.0

        print(f"\n[Orthogonality Test] K=3 integer error    : {err_amp_pct:.4f}%")
        print(f"[Orthogonality Test] K=2.4 fractional error: {err_leak_pct:.2f}%")

        # Fractional cut must exhibit substantial spectral leakage (> 2.0%) compared to integer cut
        assert err_leak_pct > 2.0, "Expected fractional cut to exhibit significant leakage bias!"
    
    def test_multipath_echo_rejection(self):
        """
        Verify that quasi-anechoic time-gating achieves 100% rejection of strong
        delayed multipath reflections (echoes), while ungated windows suffer severe bias.
        """
        fs = 500_000.0
        f0 = 2609.73
        c_sound = 343.21  # m/s

        # 1. Timeline configuration (Total window = 10.0 ms = 5000 samples)
        n_total = 5000
        t_axis = np.arange(n_total) / fs

        # Direct wave parameters (Standoff r_direct = 40.0 cm -> t = 1.165 ms)
        t_direct = 0.0020  # Direct arrival at 2.0 ms
        v_direct_peak = 0.100  # 100 mV peak -> V_RMS = 70.71 mV
        v_expected_rms = v_direct_peak / np.sqrt(2.0)

        # Delayed echo parameters (Table bounce arriving 1.8 ms later with 75% amplitude)
        t_echo = 0.0038  # Echo arrival at 3.8 ms
        v_echo_peak = 0.075  # 75 mV peak reflection

        # Direct-gate duration: K = 3 cycles (576 samples = 1.152 ms)
        n_gate = int(3 * round(fs / f0))
        t_gate_sec = n_gate / fs

        # 2. Synthesize composite acoustic signal: Direct Wave + Delayed Multipath Echo
        signal_composite = np.zeros(n_total, dtype=np.float64)

        # Direct pulse burst (lasts 1.152 ms, from 2.000 ms to 3.152 ms)
        mask_direct = (t_axis >= t_direct) & (t_axis < (t_direct + t_gate_sec))
        signal_composite[mask_direct] += v_direct_peak * np.cos(
            2.0 * np.pi * f0 * (t_axis[mask_direct] - t_direct) + (np.pi / 6.0)
        )

        # Multipath reflection burst (starts at 3.800 ms, outside the direct gate)
        mask_echo = (t_axis >= t_echo) & (t_axis < (t_echo + 0.0030))
        signal_composite[mask_echo] += v_echo_peak * np.cos(
            2.0 * np.pi * f0 * (t_axis[mask_echo] - t_echo) - (np.pi / 4.0)
        )

        # Add realistic background noise (sigma = 1.0 mV)
        np.random.seed(42)
        signal_composite += np.random.normal(0, 0.001, n_total)

        # 3. Time-Gated Extraction starting at exact direct wavefront index
        idx_direct = int(round(t_direct * fs))
        res_gated = KinematicAnalytics.extract_gated_direct_fourier(
            signal_v=signal_composite,
            fs=fs,
            f0=f0,
            n_cycles=3,
            start_idx=idx_direct,
            remove_dc=True
        )

        v_gated_rms = res_gated["amplitude_v"]
        err_gated_pct = abs(v_gated_rms - v_expected_rms) / v_expected_rms * 100.0

        # 4. Classical Ungated Window Extraction (spans direct pulse + reflection)
        # Slices across 6.0 ms, ingesting both the direct wave and the echo
        ungated_slice = signal_composite[idx_direct : idx_direct + int(0.0050 * fs)]
        v_ungated_rms = np.sqrt(np.mean((ungated_slice - np.mean(ungated_slice)) ** 2))
        err_ungated_pct = abs(v_ungated_rms - v_expected_rms) / v_expected_rms * 100.0

        # 5. SMR (Signal-to-Multipath Ratio) Calculation
        e_direct = res_gated["energy_v2"]
        idx_tail = idx_direct + n_gate
        e_tail = float(np.sum((signal_composite[idx_tail:] - np.mean(signal_composite[idx_tail:])) ** 2))
        smr_db = 10.0 * np.log10(max(e_direct, 1e-9) / max(e_tail, 1e-9))

        print(f"\n[Multipath Rejection Test] Expected Direct RMS : {v_expected_rms*1000:.3f} mV")
        print(f"[Multipath Rejection Test] Gated Direct RMS    : {v_gated_rms*1000:.3f} mV (Error = {err_gated_pct:.3f}%)")
        print(f"[Multipath Rejection Test] Ungated RMS (w/ Echo): {v_ungated_rms*1000:.3f} mV (Error = {err_ungated_pct:.1f}%)")
        print(f"[Multipath Rejection Test] Measured SMR        : {smr_db:.2f} dB")

        # Assertions:
        # Time-gated extraction must reject the echo and achieve < 1.5% error under noise
        assert err_gated_pct < 1.50, f"Gated direct extraction failed: {err_gated_pct}%"
        # Ungated extraction must show severe distortion (> 15% error) due to echo ingestion
        assert err_ungated_pct > 15.0, "Expected ungated window to be corrupted by multipath echo!"
        assert res_gated["is_valid"] is True
    
    