#!/usr/bin/env python3
"""
tools/verify_hw_energy_gate.py: Standalone Smoke Verification Script for
Hardware Direct-Path Energy Accumulation & Quasi-Anechoic Time-Gating.

Supports both:
  1. Live Execution on PYNQ-Z2 (real MMIO registers + AR2 buzzer pulse).
  2. Mocked Execution on Host PC (validates math, bitmasks, and scaling).
"""

import sys
import math
from unittest.mock import MagicMock
import numpy as np

# Try importing real MMIO from PYNQ; fallback to mock for PC testing
try:
    from pynq import MMIO
    _IS_PYNQ = True
except (ImportError, ModuleNotFoundError):
    MMIO = None
    _IS_PYNQ = False

from pynq_localizer.hw_trigger import HardwareTrigger


class MockMMIO:
    """Simulates 64KB AXI4-Lite register bank for axis_trigger_unit IP."""
    def __init__(self):
        self.regs = {
            0x00: 0x00000003,  # REG_CONTROL (Armed + Auto)
            0x04: 0x00000000,  # REG_STATUS
            0x08: 0x00000800,  # REG_THRESHOLD
            0x0C: 5000000,     # REG_TIMEOUT
            0x10: 0x00000010,  # REG_HYSTERESIS
            0x14: 0x00000001,  # REG_DECIMATION (M=10)
            0x18: 0x0000010A,  # REG_FFT_CONFIG
            0x1C: 0x00000800,  # REG_PACKET_SIZE
            0x20: 500000,      # REG_PULSE_WIDTH (5.0 ms)
            0x24: 0,           # REG_MIC1_TOA
            0x28: 0,           # REG_MIC2_TOA
            0x2C: 0x75300200,  # REG_TOA_CONFIG
            0x30: 0x80008000,  # REG_MIC_DC_REF (1.65V DC baseline)
            0x34: 0,           # REG_MIC1_DIRECT_ENERGY
            0x38: 0,           # REG_MIC2_DIRECT_ENERGY
            0x3C: 0x00000240,  # REG_GATE_CONFIG (576 samples)
        }

    def read(self, offset: int) -> int:
        return self.regs.get(offset, 0)

    def write(self, offset: int, value: int):
        self.regs[offset] = int(value) & 0xFFFFFFFF


def verify_mock_pipeline():
    """Runs strict unit and mathematical sanity checks on PC."""
    print("=" * 78)
    print("🧪 RUNNING MOCK VERIFICATION (Host PC Mode)")
    print("=" * 78)

    mock_mmio = MockMMIO()
    trig = HardwareTrigger(mock_mmio)

    # 1. Verify default N_gate configuration
    n_gate_default = trig.get_direct_gate_samples()
    print(f"• Default Direct Gate Samples : {n_gate_default} (Expected: 576)")
    assert n_gate_default == 576, f"Default gate count mismatch: {n_gate_default}"

    # 2. Test dynamic gate sample reprogramming
    trig.set_direct_gate_samples(192)  # 1 acoustic cycle @ 2610 Hz
    assert trig.get_direct_gate_samples() == 192, "Reprogramming N_gate failed!"
    trig.set_direct_gate_samples(576)  # Restore 3 cycles
    assert trig.get_direct_gate_samples() == 576

    # 3. Simulate hardware acoustic burst and energy calculation
    # Pure tone: amplitude 50 mV peak (35.355 mV RMS)
    # ADC scale = 3.3V / 4095 = ~0.80586 mV/count
    # 50 mV peak corresponds to ~62.04 ADC counts peak
    # For a sine wave, average squared deviation is (A_counts)^2 / 2
    adc_scale = 3.3 / 4095.0
    v_peak = 0.050
    a_counts = v_peak / adc_scale
    mean_sq_dev = (a_counts ** 2) / 2.0  # ~1924.8 counts^2

    n_samples = 576
    simulated_energy = int(mean_sq_dev * n_samples)  # ~1,108,685 counts^2

    mock_mmio.regs[0x34] = simulated_energy
    mock_mmio.regs[0x38] = simulated_energy
    # Assert Mic1GateDone (bit 7) and Mic2GateDone (bit 8)
    mock_mmio.regs[0x04] |= (1 << 7) | (1 << 8)

    assert trig.is_mic1_gate_done is True
    assert trig.is_mic2_gate_done is True
    assert trig.is_direct_gate_done is True

    e1, e2 = trig.get_direct_energy_counts()
    v1_rms, v2_rms = trig.get_direct_rms_voltage()

    expected_v_rms = v_peak / math.sqrt(2.0)
    err1_pct = abs(v1_rms - expected_v_rms) / expected_v_rms * 100.0

    print(f"• Injected Energy Counts     : {simulated_energy} counts² over {n_samples} samples")
    print(f"• Expected Physical V_RMS    : {expected_v_rms * 1000.0:.3f} mV")
    print(f"• Converted Mic 1 V_RMS      : {v1_rms * 1000.0:.3f} mV (Error = {err1_pct:.3f}%)")
    print(f"• Converted Mic 2 V_RMS      : {v2_rms * 1000.0:.3f} mV")

    assert err1_pct < 0.1, f"Voltage conversion error too high: {err1_pct}%"

    # 4. Monotonic inverse-square energy decay test (20 cm -> 60 cm)
    distances = np.array([0.20, 0.40, 0.60])
    energies = []
    k_const = 0.050  # V*m

    for r in distances:
        v_r = (k_const / r)
        counts_r = v_r / adc_scale
        # RMS energy counts over N samples: N * (counts_r)^2
        e_r = int(n_samples * (counts_r ** 2))
        energies.append(e_r)

    print("-" * 78)
    print("• Distance Decay Verification (E ∝ 1/r²):")
    for r, e in zip(distances, energies):
        print(f"   At r = {r*100:.0f} cm ──► Direct Energy = {e:10d} counts²")

    assert energies[0] > energies[1] > energies[2], "Energy is not monotonically decreasing with distance!"
    ratio_expected = (0.40 / 0.20) ** 2  # 4x reduction for 2x distance
    ratio_measured = energies[0] / energies[1]
    assert abs(ratio_measured - ratio_expected) < 0.05

    print("=" * 78)
    print("✅ Step 3.3 MOCK SMOKE VERIFICATION PASSED SUCCESSFULLY")
    print("=" * 78)


def verify_live_hardware():
    """Runs live pulse capture and direct-energy readout on PYNQ-Z2."""
    print("=" * 78)
    print("🔌 RUNNING LIVE HARDWARE VERIFICATION (PYNQ-Z2)")
    print("=" * 78)

    from pynq_localizer import MicrophoneArrayOverlay
    ol = MicrophoneArrayOverlay()
    trig = ol.trigger

    print(f"• Driver: {trig}")
    trig.set_direct_gate_samples(576)
    print(f"• Configured N_gate: {trig.get_direct_gate_samples()} samples")

    # Fire pulse and check ToA + direct-energy registers
    print("• Strobing hardware pulse on AR2 (U13)...")
    trig.fire_pulse()

    import time
    time.sleep(0.060)  # Wait for 50 ms ToA window and gate completion

    c1, c2 = trig.get_toa_cycles()
    e1, e2 = trig.get_direct_energy_counts()
    v1, v2 = trig.get_direct_rms_voltage()

    print(f"• Arrival Cycles (10 ns ticks) : Mic 1 = {c1}, Mic 2 = {c2}")
    print(f"• Direct Energy (counts²)      : Mic 1 = {e1}, Mic 2 = {e2}")
    print(f"• Direct Voltage V_RMS         : Mic 1 = {v1*1000.0:.2f} mV, Mic 2 = {v2*1000.0:.2f} mV")
    print(f"• Gate Status                  : Mic 1 Done = {trig.is_mic1_gate_done}, Mic 2 Done = {trig.is_mic2_gate_done}")

    ol.close()
    print("=" * 78)
    print("✅ LIVE HARDWARE SMOKE TEST COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    if _IS_PYNQ:
        verify_live_hardware()
    else:
        verify_mock_pipeline()