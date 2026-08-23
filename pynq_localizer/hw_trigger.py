"""
pynq_localizer.hw_trigger: High-Level Driver for FPGA 'axis_trigger_unit' IP.
Acts as the central Timing & Configuration Controller for Triggering, Decimation, and Packet Limits.
"""

from typing import Union
from pynq import MMIO


class HardwareTrigger:
    """
    High-level Python driver for the FPGA-based 'axis_trigger_unit' IP.

    Interfaces via AXI4-Lite registers to configure hardware-level edge detection,
    trigger channel source selection (CH1/A0 vs CH2/A1), FFT channel routing (A0 vs A1),
    voltage thresholds, decimation factors (M=1, 10, 20, 50), FFT transform length (N=512, 1024, 2048),
    and packetizer boundaries.
    """

    # Register Byte Offsets matching axis_trigger_unit.vhd
    REG_CONTROL     = 0x00
    REG_STATUS      = 0x04
    REG_THRESHOLD   = 0x08
    REG_TIMEOUT     = 0x0C
    REG_HYSTERESIS  = 0x10
    REG_DECIMATION  = 0x14  # [1:0] 00=M=1, 01=M=10, 10=M=20, 11=M=50
    REG_FFT_CONFIG  = 0x18  # [15:0] (FWD_INV << 8) | NFFT (PG109 Format)
    REG_PACKET_SIZE = 0x1C  # [15:0] Samples per DMA frame

    # Bit masks for REG_CONTROL (0x00)
    BIT_ARM          = 1 << 0  # Bit 0: Arm trigger unit
    BIT_AUTO_MODE    = 1 << 1  # Bit 1: 1 = Auto Mode, 0 = Normal Mode
    BIT_EDGE_FALLING = 1 << 2  # Bit 2: 0 = Rising Edge, 1 = Falling Edge
    BIT_SINGLE_SHOT  = 1 << 3  # Bit 3: 1 = Single Shot, 0 = Continuous
    BIT_FORCE_TRIG   = 1 << 4  # Bit 4: Software force trigger pulse
    BIT_TRIG_SRC_CH2 = 1 << 5  # Bit 5: 0 = Trigger on CH1 (A0), 1 = Trigger on CH2 (A1)
    BIT_FFT_SRC_CH2  = 1 << 6  # Bit 6: 0 = Route CH1 (A0) to FFT, 1 = Route CH2 (A1) to FFT

    # Bit masks for REG_STATUS (0x04)
    STATUS_ARMED     = 1 << 0
    STATUS_TRIGGERED = 1 << 1
    STATUS_STREAMING = 1 << 2

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

        if isinstance(overlay_or_mmio, MMIO):
            self.mmio = overlay_or_mmio
        elif hasattr(overlay_or_mmio, "axis_trigger_unit_0"):
            self.mmio = overlay_or_mmio.axis_trigger_unit_0.mmio
        elif hasattr(overlay_or_mmio, "ip_dict"):
            trigger_ips = [k for k in overlay_or_mmio.ip_dict.keys() if "trigger" in k.lower()]
            if trigger_ips:
                self.mmio = getattr(overlay_or_mmio, trigger_ips[0]).mmio
            else:
                self.mmio = MMIO(0x43C10000, 65536)
        else:
            self.mmio = MMIO(0x43C10000, 65536)

        # Default initialization: 1.65V Threshold, Auto Mode, Rising Edge, CH1 (A0), M=10, N=2048
        self.configure(
            mode="Auto",
            edge="Rising",
            source="CH1",
            threshold_volts=1.65,
            timeout_ms=50.0,
            hysteresis_volts=0.02
        )
        self.set_fft_channel("CH1")

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

    def set_source(self, source: str):
        """Set trigger source channel: 'CH1' (A0) or 'CH2' (A1)."""
        src_clean = source.strip().upper()
        ctrl = self.mmio.read(self.REG_CONTROL)
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
        ctrl = self.mmio.read(self.REG_CONTROL)
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
        ctrl = self.mmio.read(self.REG_CONTROL)
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
        ctrl = self.mmio.read(self.REG_CONTROL)
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
        ctrl = self.mmio.read(self.REG_CONTROL)
        self.mmio.write(self.REG_CONTROL, ctrl | self.BIT_ARM)

    def disarm(self):
        """Disarm the trigger unit."""
        ctrl = self.mmio.read(self.REG_CONTROL)
        self.mmio.write(self.REG_CONTROL, ctrl & ~self.BIT_ARM)

    def force_trigger(self):
        """Manually trigger acquisition via software pulse."""
        ctrl = self.mmio.read(self.REG_CONTROL)
        self.mmio.write(self.REG_CONTROL, ctrl | self.BIT_FORCE_TRIG)

    def set_decimation(self, factor: int):
        """
        Configures the hardware AXI-Stream decimator ratio M.
        :param factor: 1 (500 kSPS), 10 (50 kSPS), 20 (25 kSPS), or 50 (10 kSPS).
        """
        if factor not in self.DECIMATION_MAP:
            raise ValueError(f"Invalid decimation factor {factor}. Choose from: 1, 10, 20, 50.")
        code = self.DECIMATION_MAP[factor]
        self.mmio.write(self.REG_DECIMATION, code)

    def get_decimation(self) -> int:
        """Reads active hardware decimation factor M."""
        code = self.mmio.read(self.REG_DECIMATION) & 0x3
        return self.REVERSE_DECIMATION_MAP.get(code, 10)

    def set_fft_config(self, n_points: int = 2048, forward: bool = True):
        """
        Dynamically configures the Xilinx LogiCORE FFT core (PG109 format).
        :param n_points: Transform length (512, 1024, 2048).
        :param forward: True for Forward FFT, False for Inverse FFT.
        """
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

    @property
    def is_armed(self) -> bool:
        return bool(self.mmio.read(self.REG_STATUS) & self.STATUS_ARMED)

    @property
    def is_triggered(self) -> bool:
        return bool(self.mmio.read(self.REG_STATUS) & self.STATUS_TRIGGERED)

    def __repr__(self) -> str:
        ctrl = self.mmio.read(self.REG_CONTROL)
        src = "CH2 (A1)" if (ctrl & self.BIT_TRIG_SRC_CH2) else "CH1 (A0)"
        fft_src = "CH2 (A1)" if (ctrl & self.BIT_FFT_SRC_CH2) else "CH1 (A0)"
        edge = "Falling" if (ctrl & self.BIT_EDGE_FALLING) else "Rising"
        mode = "Auto" if (ctrl & self.BIT_AUTO_MODE) else ("Single" if (ctrl & self.BIT_SINGLE_SHOT) else "Normal")
        armed = "ARMED" if (ctrl & self.BIT_ARM) else "DISARMED"
        m = self.get_decimation()
        n = self.get_fft_length()
        pkt = self.get_packet_size()
        return f"<HardwareTrigger: {armed}, TrigSrc={src}, FFTSrc={fft_src}, Mode={mode}, Edge={edge}, M={m}x, N={n}, Pkt={pkt}>"