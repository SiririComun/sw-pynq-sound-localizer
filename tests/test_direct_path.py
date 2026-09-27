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