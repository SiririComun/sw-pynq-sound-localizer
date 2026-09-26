"""
pynq_localizer.hw_trigger: High-Level Driver for FPGA 'axis_trigger_unit' IP.
Acts as the central Timing & Configuration Controller for Triggering, Decimation,
FFT configuration, Hardware Active Buzzer Pulse Generation, and Hardware-Accelerated
100 MHz ToA / TDOA Cycle-Accurate Counters.
"""

import time
from typing import Union, Optional, Tuple
try:
    from pynq import MMIO
except (ImportError, ModuleNotFoundError):
    MMIO = None

class HardwareTrigger:
    """
    High-level Python driver for the FPGA-based 'axis_trigger_unit' IP.

    Interfaces via AXI4-Lite registers to configure hardware-level edge detection,
    trigger channel source selection (CH1/A0 vs CH2/A1), FFT channel routing (A0 vs A1),
    voltage thresholds, decimation factors (M=1, 10, 20, 50), FFT transform length (N=512, 1024, 2048),
    packetizer boundaries, hardware pulse generation, and hardware-accelerated 100 MHz ToA / TDOA counters.
    """

    # Register Byte Offsets matching axis_trigger_unit.vhd (6-bit address decoder)
    REG_CONTROL     = 0x00  # [0]=Arm, [1]=Auto, [2]=Fall, [3]=Single, [4]=Force, [5]=TrigSrc, [6]=FFTSrc, [7]=FIRE_PULSE
    REG_STATUS      = 0x04  # [0]=Armed, [1]=Triggered, [2]=Streaming, [3]=PulseActive, [4]=Mic1Locked, [5]=Mic2Locked, [6]=ToaDone
    REG_THRESHOLD   = 0x08  # [15:0] 12-bit left-aligned comparator threshold
    REG_TIMEOUT     = 0x0C  # [31:0] Auto-trigger timeout in clock cycles
    REG_HYSTERESIS  = 0x10  # [15:0] Noise rejection band
    REG_DECIMATION  = 0x14  # [1:0]  00=M=1, 01=M=10, 10=M=20, 11=M=50
    REG_FFT_CONFIG  = 0x18  # [15:0] (FWD_INV << 8) | NFFT (PG109 Format)
    REG_PACKET_SIZE = 0x1C  # [15:0] Samples per DMA frame
    REG_PULSE_WIDTH = 0x20  # [31:0] Hardware pulse duration in 100 MHz clock cycles (default 500,000 = 5.0 ms)
    
    # Hardware-Accelerated ToA / TDOA Registers
    REG_MIC1_TOA    = 0x24  # [31:0] Mic 1 arrival timestamp in 100 MHz clock cycles (10.0 ns ticks)
    REG_MIC2_TOA    = 0x28  # [31:0] Mic 2 arrival timestamp in 100 MHz clock cycles (10.0 ns ticks)
    REG_TOA_CONFIG  = 0x2C  # [31:16]=Blanking cycles (10 ns ticks), [15:0]=Threshold delta counts
    REG_MIC_DC_REF  = 0x30  # [31:16]=Mic 2 DC baseline, [15:0]=Mic 1 DC baseline

    # Bit masks for REG_CONTROL (0x00)
    BIT_ARM          = 1 << 0  # Bit 0: Arm trigger unit
    BIT_AUTO_MODE    = 1 << 1  # Bit 1: 1 = Auto Mode, 0 = Normal Mode
    BIT_EDGE_FALLING = 1 << 2  # Bit 2: 0 = Rising Edge, 1 = Falling Edge
    BIT_SINGLE_SHOT  = 1 << 3  # Bit 3: 1 = Single Shot, 0 = Continuous
    BIT_FORCE_TRIG   = 1 << 4  # Bit 4: Software force trigger pulse
    BIT_TRIG_SRC_CH2 = 1 << 5  # Bit 5: 0 = Trigger on CH1 (A0), 1 = Trigger on CH2 (A1)
    BIT_FFT_SRC_CH2  = 1 << 6  # Bit 6: 0 = Route CH1 (A0) to FFT, 1 = Route CH2 (A1) to FFT
    BIT_FIRE_PULSE   = 1 << 7  # Bit 7: Strobe hardware buzzer pulse & sync acquisition

    # Bit masks for REG_STATUS (0x04)
    STATUS_ARMED           = 1 << 0
    STATUS_TRIGGERED       = 1 << 1
    STATUS_STREAMING       = 1 << 2
    STATUS_PULSE_ACTIVE    = 1 << 3  # Bit 3: 1 while buzzer pulse pin is actively firing HIGH
    STATUS_MIC1_TOA_LOCKED = 1 << 4  # Bit 4: 1 when Mic 1 (A0) wavefront arrival is latched
    STATUS_MIC2_TOA_LOCKED = 1 << 5  # Bit 5: 1 when Mic 2 (A1) wavefront arrival is latched
    STATUS_TOA_DONE        = 1 << 6  # Bit 6: 1 when both channels locked or 50 ms timeout expired

    DECIMATION_MAP = {
        1: 0,   # "00" -> M = 1 (Bypass: 500 kSPS Lab Scope)
        10: 1,  # "01" -> M = 10 (50 kSPS Full Audio)
        20: 2,  # "10" -> M = 20 (25 kSPS Speech / Vocal)
        50: 3   # "11" -> M = 50 (10 kSPS Deep Bass Zoom)
    }
    REVERSE_DECIMATION_MAP = {0: 1, 1: 10, 2: 20, 3: 50}

    def __init__(self, overlay_or_mmio: Union[object, MMIO], clock_freq_hz: int = 100_000_000):
        self.clock_freq_hz = clock_freq_hz
        self.max_voltage = 3.3

        if hasattr(overlay_or_mmio, "read") and hasattr(overlay_or_mmio, "write"):
            self.mmio = overlay_or_mmio
        elif hasattr(overlay_or_mmio, "axis_trigger_unit_0"):
            self.mmio = overlay_or_mmio.axis_trigger_unit_0.mmio
        elif hasattr(overlay_or_mmio, "ip_dict"):
            trigger_ips = [k for k in overlay_or_mmio.ip_dict.keys() if "trigger" in k.lower()]
            if trigger_ips:
                self.mmio = getattr(overlay_or_mmio, trigger_ips[0]).mmio
            else:
                self.mmio = MMIO(0x43C10000, 65536) if MMIO is not None else None
        else:
            self.mmio = MMIO(0x43C10000, 65536) if MMIO is not None else None

        if self.mmio is not None:
            self.configure(
                mode="Auto",
                edge="Rising",
                source="CH1",
                threshold_volts=1.65,
                timeout_ms=50.0,
                hysteresis_volts=0.02
            )
            self.set_fft_channel("CH1")
            self.set_pulse_width_ms(5.0)
            # Default ToA: 50 mV threshold above DC, 0.30 ms blanking (30,000 cycles)
            self.set_toa_config(threshold_mv=50.0, blanking_ms=0.30)
            self.set_mic_dc_references(dc_mic1_v=1.65, dc_mic2_v=1.65)

    def configure(
        self,
        mode: str = "Auto",
        edge: str = "Rising",
        source: str = "CH1",
        threshold_volts: float = 1.65,
        timeout_ms: float = 50.0,
        hysteresis_volts: float = 0.02
    ):
        """Configure all hardware trigger settings simultaneously."""
        self.set_threshold(threshold_volts)
        self.set_timeout_ms(timeout_ms)
        self.set_hysteresis(hysteresis_volts)
        self.set_source(source)
        self.set_edge(edge)
        self.set_mode(mode)

    def _read_control_sanitized(self) -> int:
        """Reads REG_CONTROL with transient strobe bits masked out."""
        return self.mmio.read(self.REG_CONTROL) & ~self.BIT_FIRE_PULSE & ~self.BIT_FORCE_TRIG

    def set_source(self, source: str):
        """Set trigger source channel: 'CH1' (A0) or 'CH2' (A1)."""
        src_clean = source.strip().upper()
        ctrl = self._read_control_sanitized()
        if "CH2" in src_clean or "A1" in src_clean:
            ctrl |= self.BIT_TRIG_SRC_CH2
        else:
            ctrl &= ~self.BIT_TRIG_SRC_CH2
        self.mmio.write(self.REG_CONTROL, ctrl)

    def get_source(self) -> str:
        """Read active trigger source channel from hardware."""
        ctrl = self.mmio.read(self.REG_CONTROL)
        return "CH2 (A1)" if (ctrl & self.BIT_TRIG_SRC_CH2) else "CH1 (A0)"

    def set_fft_channel(self, source: str = "CH1"):
        """Configures the hardware stream demux routing to the FFT core."""
        src_clean = source.strip().upper()
        ctrl = self._read_control_sanitized()
        if "CH2" in src_clean or "A1" in src_clean:
            ctrl |= self.BIT_FFT_SRC_CH2
        else:
            ctrl &= ~self.BIT_FFT_SRC_CH2
        self.mmio.write(self.REG_CONTROL, ctrl)

    def get_fft_channel(self) -> str:
        """Read the active channel routed to the hardware FFT core."""
        ctrl = self.mmio.read(self.REG_CONTROL)
        return "CH2 (A1)" if (ctrl & self.BIT_FFT_SRC_CH2) else "CH1 (A0)"

    def set_mode(self, mode: str):
        """Set trigger operating mode: 'Auto', 'Normal', or 'Single'."""
        mode_clean = mode.strip().capitalize()
        ctrl = self._read_control_sanitized()
        if mode_clean == "Auto":
            ctrl |= (self.BIT_ARM | self.BIT_AUTO_MODE)
            ctrl &= ~self.BIT_SINGLE_SHOT
        elif mode_clean == "Normal":
            ctrl |= self.BIT_ARM
            ctrl &= ~(self.BIT_AUTO_MODE | self.BIT_SINGLE_SHOT)
        elif mode_clean == "Single":
            ctrl |= (self.BIT_ARM | self.BIT_SINGLE_SHOT)
            ctrl &= ~self.BIT_AUTO_MODE
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: 'Auto', 'Normal', 'Single'.")
        self.mmio.write(self.REG_CONTROL, ctrl)

    def set_edge(self, edge: str):
        """Set trigger slope direction: 'Rising' or 'Falling'."""
        edge_clean = edge.strip().capitalize()
        ctrl = self._read_control_sanitized()
        if edge_clean == "Rising":
            ctrl &= ~self.BIT_EDGE_FALLING
        elif edge_clean == "Falling":
            ctrl |= self.BIT_EDGE_FALLING
        else:
            raise ValueError(f"Invalid edge '{edge}'. Choose from: 'Rising' or 'Falling'.")
        self.mmio.write(self.REG_CONTROL, ctrl)

    def set_threshold(self, volts: float):
        """Set analog trigger threshold in Volts (0.0V to 3.3V)."""
        clamped_volts = max(0.0, min(self.max_voltage, float(volts)))
        raw_12bit = int((clamped_volts / self.max_voltage) * 4095.0)
        raw_code = (raw_12bit & 0xFFF) << 4
        self.mmio.write(self.REG_THRESHOLD, raw_code)

    def get_threshold(self) -> float:
        """Read active threshold voltage from hardware register."""
        raw_code = self.mmio.read(self.REG_THRESHOLD)
        raw_12bit = (raw_code >> 4) & 0xFFF
        return (raw_12bit / 4095.0) * self.max_voltage

    def set_timeout_ms(self, timeout_ms: float):
        """Set timeout in milliseconds for Auto-Trigger mode."""
        cycles = int((float(timeout_ms) / 1000.0) * self.clock_freq_hz)
        self.mmio.write(self.REG_TIMEOUT, max(100, cycles))

    def get_timeout_ms(self) -> float:
        """Read active auto-timeout in milliseconds."""
        cycles = self.mmio.read(self.REG_TIMEOUT)
        return (cycles / self.clock_freq_hz) * 1000.0

    def set_hysteresis(self, volts: float):
        """Set noise rejection band in Volts."""
        clamped_volts = max(0.0, min(0.5, float(volts)))
        raw_12bit = int((clamped_volts / self.max_voltage) * 4095.0)
        raw_code = (raw_12bit & 0xFFF) << 4
        self.mmio.write(self.REG_HYSTERESIS, raw_code)

    def arm(self):
        """Arm the trigger unit."""
        ctrl = self._read_control_sanitized()
        self.mmio.write(self.REG_CONTROL, ctrl | self.BIT_ARM)

    def disarm(self):
        """Disarm the trigger unit."""
        ctrl = self._read_control_sanitized()
        self.mmio.write(self.REG_CONTROL, ctrl & ~self.BIT_ARM)

    def force_trigger(self):
        """Manually trigger acquisition via software pulse."""
        ctrl = self._read_control_sanitized()
        self.mmio.write(self.REG_CONTROL, ctrl | self.BIT_FORCE_TRIG)
        self.mmio.write(self.REG_CONTROL, ctrl)  # Self-clear

    def set_decimation(self, factor: int):
        """Configures the hardware AXI-Stream decimator ratio M."""
        if factor not in self.DECIMATION_MAP:
            raise ValueError(f"Invalid decimation factor {factor}. Choose from: 1, 10, 20, 50.")
        code = self.DECIMATION_MAP[factor]
        self.mmio.write(self.REG_DECIMATION, code)

    def get_decimation(self) -> int:
        """Reads active hardware decimation factor M."""
        code = self.mmio.read(self.REG_DECIMATION) & 0x3
        return self.REVERSE_DECIMATION_MAP.get(code, 10)

    def set_fft_config(self, n_points: int = 2048, forward: bool = True):
        """Dynamically configures the Xilinx LogiCORE FFT core."""
        valid_sizes = {512: 9, 1024: 10, 2048: 11}
        if n_points not in valid_sizes:
            raise ValueError(f"Invalid FFT size {n_points}. Supported sizes: 512, 1024, 2048.")
        nfft = valid_sizes[n_points]
        fwd_bit = 1 if forward else 0
        config_word = (fwd_bit << 8) | nfft
        self.mmio.write(self.REG_FFT_CONFIG, config_word)

    def get_fft_length(self) -> int:
        """Reads active FFT transform length N from hardware register."""
        raw = self.mmio.read(self.REG_FFT_CONFIG)
        nfft = raw & 0x1F
        return 2 ** nfft

    def set_packet_size(self, size_samples: int):
        """Configures the hardware TLAST packetizer sample count limit."""
        clamped = max(64, min(65535, int(size_samples)))
        self.mmio.write(self.REG_PACKET_SIZE, clamped)

    def get_packet_size(self) -> int:
        """Reads active hardware TLAST packet size."""
        return self.mmio.read(self.REG_PACKET_SIZE) & 0xFFFF

    # =========================================================================
    # Hardware Active Buzzer Pulse Generator Methods (Offset 0x20 & Bit 7)
    # =========================================================================

    def set_pulse_width_cycles(self, cycles: int):
        """Sets hardware pulse duration in 100 MHz clock cycles."""
        clamped = max(10, min(100_000_000, int(cycles)))
        self.mmio.write(self.REG_PULSE_WIDTH, clamped)

    def get_pulse_width_cycles(self) -> int:
        """Reads active hardware pulse duration in 100 MHz clock cycles."""
        return self.mmio.read(self.REG_PULSE_WIDTH)

    def set_pulse_width_ms(self, duration_ms: float):
        """Sets hardware pulse duration in milliseconds."""
        cycles = int((float(duration_ms) / 1000.0) * self.clock_freq_hz)
        self.set_pulse_width_cycles(cycles)

    def get_pulse_width_ms(self) -> float:
        """Reads active hardware pulse duration in milliseconds."""
        cycles = self.get_pulse_width_cycles()
        return (cycles / self.clock_freq_hz) * 1000.0

    def fire_pulse(self):
        """
        Strobes the hardware active buzzer pulse on Arduino pin AR2 (Pin U13)
        and synchronously triggers the ToA engine on cycle 0.
        Self-clears Bit 7 immediately to guarantee no sticky re-triggering.
        """
        ctrl = self._read_control_sanitized()
        # Assert Bit 7 strobe
        self.mmio.write(self.REG_CONTROL, ctrl | self.BIT_FIRE_PULSE)
        # Immediately de-assert Bit 7
        self.mmio.write(self.REG_CONTROL, ctrl)

    # =========================================================================
    # Hardware-Accelerated 100 MHz ToA & TDOA Engine (0x24, 0x28, 0x2C, 0x30)
    # =========================================================================

    def set_toa_config(self, threshold_mv: float = 50.0, blanking_ms: float = 0.30):
        """
        Configures the PL ToA comparator threshold and initial blanking window.
        :param threshold_mv: Minimum acoustic wave excursion in mV above DC baseline.
        :param blanking_ms: Blanking period in ms to ignore transistor electrical noise.
        """
        clamped_mv = max(1.0, min(1000.0, float(threshold_mv)))
        # Convert mV to 16-bit left-aligned ADC code counts (3.3V = 65535 counts)
        thresh_code = int((clamped_mv / 1000.0 / self.max_voltage) * 65535.0)
        thresh_code = max(1, min(65535, thresh_code))

        blanking_cycles = int((float(blanking_ms) / 1000.0) * self.clock_freq_hz)
        blanking_cycles = max(10, min(65535, blanking_cycles))

        word_val = ((blanking_cycles & 0xFFFF) << 16) | (thresh_code & 0xFFFF)
        self.mmio.write(self.REG_TOA_CONFIG, word_val)

    def get_toa_config(self) -> Tuple[float, float]:
        """Reads (threshold_mv, blanking_ms) from REG_TOA_CONFIG (0x2C)."""
        raw = self.mmio.read(self.REG_TOA_CONFIG)
        thresh_code = raw & 0xFFFF
        blanking_cycles = (raw >> 16) & 0xFFFF

        thresh_mv = (thresh_code / 65535.0) * self.max_voltage * 1000.0
        blanking_ms = (blanking_cycles / self.clock_freq_hz) * 1000.0
        return thresh_mv, blanking_ms

    def set_mic_dc_references(self, dc_mic1_v: float = 1.65, dc_mic2_v: float = 1.65):
        """
        Sets nominal DC operating baseline voltages for Mic 1 and Mic 2.
        :param dc_mic1_v: DC bias voltage of Mic 1 (default 1.65V).
        :param dc_mic2_v: DC bias voltage of Mic 2 (default 1.65V).
        """
        code1 = int((float(dc_mic1_v) / self.max_voltage) * 65535.0) & 0xFFFF
        code2 = int((float(dc_mic2_v) / self.max_voltage) * 65535.0) & 0xFFFF
        word_val = (code2 << 16) | code1
        self.mmio.write(self.REG_MIC_DC_REF, word_val)

    def get_mic_dc_references(self) -> Tuple[float, float]:
        """Reads (dc_mic1_v, dc_mic2_v) from REG_MIC_DC_REF (0x30)."""
        raw = self.mmio.read(self.REG_MIC_DC_REF)
        code1 = raw & 0xFFFF
        code2 = (raw >> 16) & 0xFFFF
        return (code1 / 65535.0) * self.max_voltage, (code2 / 65535.0) * self.max_voltage

    def get_mic1_toa_cycles(self) -> int:
        """Reads Mic 1 hardware arrival timestamp in 100 MHz clock cycles (10.0 ns ticks)."""
        return self.mmio.read(self.REG_MIC1_TOA)

    def get_mic2_toa_cycles(self) -> int:
        """Reads Mic 2 hardware arrival timestamp in 100 MHz clock cycles (10.0 ns ticks)."""
        return self.mmio.read(self.REG_MIC2_TOA)

    def get_toa_cycles(self) -> Tuple[int, int]:
        """Returns (cycles_mic1, cycles_mic2) from hardware registers."""
        return self.get_mic1_toa_cycles(), self.get_mic2_toa_cycles()

    def get_tdoa_cycles(self) -> int:
        """Returns signed cycle difference (Mic 2 - Mic 1) in 10.0 ns clock cycles."""
        c1, c2 = self.get_toa_cycles()
        return c2 - c1

    @property
    def is_mic1_toa_locked(self) -> bool:
        """True when Mic 1 acoustic arrival has been detected by PL."""
        return bool(self.mmio.read(self.REG_STATUS) & self.STATUS_MIC1_TOA_LOCKED)

    @property
    def is_mic2_toa_locked(self) -> bool:
        """True when Mic 2 acoustic arrival has been detected by PL."""
        return bool(self.mmio.read(self.REG_STATUS) & self.STATUS_MIC2_TOA_LOCKED)

    @property
    def is_toa_done(self) -> bool:
        """True when both microphones locked or 50 ms timeout expired."""
        return bool(self.mmio.read(self.REG_STATUS) & self.STATUS_TOA_DONE)

    def trigger_hardware_toa(self, timeout_ms: float = 50.0) -> Tuple[int, int]:
        """
        Fires the hardware pulse and polls REG_STATUS until TOA_DONE is asserted.
        Executes entirely in ~1 to 5 ms with ZERO DMA interaction.
        :param timeout_ms: Maximum wait time in milliseconds.
        :return: (cycles_mic1, cycles_mic2) raw counts at 100 MHz.
        """
        self.fire_pulse()
        t0 = time.time()
        while not self.is_toa_done:
            if (time.time() - t0) * 1000.0 > timeout_ms:
                break
            time.sleep(0.0001)

        return self.get_toa_cycles()

    @property
    def is_pulse_active(self) -> bool:
        """True while the hardware pulse output pin is actively firing HIGH."""
        return bool(self.mmio.read(self.REG_STATUS) & self.STATUS_PULSE_ACTIVE)

    @property
    def is_armed(self) -> bool:
        return bool(self.mmio.read(self.REG_STATUS) & self.STATUS_ARMED)

    @property
    def is_triggered(self) -> bool:
        return bool(self.mmio.read(self.REG_STATUS) & self.STATUS_TRIGGERED)

    def __repr__(self) -> str:
        ctrl = self.mmio.read(self.REG_CONTROL) if self.mmio else 0
        src = "CH2 (A1)" if (ctrl & self.BIT_TRIG_SRC_CH2) else "CH1 (A0)"
        mode = "Auto" if (ctrl & self.BIT_AUTO_MODE) else ("Single" if (ctrl & self.BIT_SINGLE_SHOT) else "Normal")
        armed = "ARMED" if (ctrl & self.BIT_ARM) else "DISARMED"
        m = self.get_decimation() if self.mmio else 10
        pulse_ms = self.get_pulse_width_ms() if self.mmio else 5.0
        c1, c2 = self.get_toa_cycles() if self.mmio else (0, 0)
        return (f"<HardwareTrigger: {armed}, Mode={mode}, TrigSrc={src}, M={m}x, Pulse={pulse_ms:.1f}ms, "
                f"PL_ToA_Cycles=(M1:{c1}, M2:{c2})>")