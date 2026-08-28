"""
pynq_localizer.spectral_mask: High-Level Driver for FPGA 'axis_spectral_mask' IP.
Provides runtime control over real-time frequency-domain filtering (Lowpass, Highpass,
Bandpass, Notch) with Hermitian symmetry preservation for IFFT time reconstruction.
"""

from typing import Union, Optional, Dict
try:
    from pynq import MMIO
except ImportError:
    MMIO = object


class SpectralMaskDriver:
    """
    High-level Python driver for the FPGA 'axis_spectral_mask' IP at base address 0x43C20000.

    Register Map (Matching axis_spectral_mask.vhd):
      • 0x00: REG_CTRL       - [0] Enable (0=Bypass, 1=Enable), [2:1] Mode
                               00=Lowpass, 01=Highpass, 10=Bandpass, 11=Notch
      • 0x04: REG_BIN_START  - [15:0] Lower cutoff frequency bin (k_start)
      • 0x08: REG_BIN_STOP   - [15:0] Upper cutoff frequency bin (k_stop)
      • 0x0C: REG_FFT_LEN    - [15:0] FFT transform length N (default: 1024 or 2048)
      • 0x10: REG_STATUS     - [0] Frame Active, [31:16] Current streaming bin index
    """

    # Register Byte Offsets
    REG_CTRL      = 0x00
    REG_BIN_START = 0x04
    REG_BIN_STOP  = 0x08
    REG_FFT_LEN   = 0x0C
    REG_STATUS    = 0x10

    # Control Bit Masks
    BIT_ENABLE    = 1 << 0

    # Mode Bit Patterns
    MODE_LOWPASS  = 0b00 << 1  # 0x00: Lowpass (Bass)
    MODE_HIGHPASS = 0b01 << 1  # 0x02: Highpass (Treble)
    MODE_BANDPASS = 0b10 << 1  # 0x04: Bandpass
    MODE_NOTCH    = 0b11 << 1  # 0x06: Notch (Bandstop)

    MODE_MAP = {
        "lowpass":  MODE_LOWPASS,
        "highpass": MODE_HIGHPASS,
        "bandpass": MODE_BANDPASS,
        "notch":    MODE_NOTCH,
    }
    REVERSE_MODE_MAP = {
        0b00: "lowpass",
        0b01: "highpass",
        0b10: "bandpass",
        0b11: "notch",
    }

    def __init__(
        self,
        overlay_or_mmio: Union[object, MMIO],
        fs: float = 50_000.0,
        fft_len: int = 1024
    ):
        """
        Initializes the Spectral Mask driver.
        :param overlay_or_mmio: PYNQ Overlay instance, IP block, or MMIO instance.
        :param fs: Active sampling rate in Hz (e.g. 50,000 for Audio profile).
        :param fft_len: Transform length N (512, 1024, or 2048).
        """
        self.fs = float(fs)
        self.fft_len = int(fft_len)

        if isinstance(overlay_or_mmio, MMIO):
            self.mmio = overlay_or_mmio
        elif hasattr(overlay_or_mmio, "axis_spectral_mask_0"):
            self.mmio = overlay_or_mmio.axis_spectral_mask_0.mmio
        elif hasattr(overlay_or_mmio, "ip_dict"):
            mask_ips = [k for k in overlay_or_mmio.ip_dict.keys() if "spectral_mask" in k.lower() or "mask" in k.lower()]
            if mask_ips:
                self.mmio = getattr(overlay_or_mmio, mask_ips[0]).mmio
            else:
                self.mmio = MMIO(0x43C20000, 65536)
        else:
            self.mmio = MMIO(0x43C20000, 65536)

        # Synchronize hardware FFT length and default to Bypass
        self.set_fft_length(self.fft_len)
        self.bypass()

    # =========================================================================
    # 1. Frequency Bin Conversions
    # =========================================================================

    def freq_to_bin(self, freq_hz: float) -> int:
        """Converts frequency in Hz to closest hardware discrete bin index k."""
        df = self.fs / float(self.fft_len)
        k = int(round(float(freq_hz) / df))
        nyquist_bin = self.fft_len // 2
        return max(0, min(nyquist_bin, k))

    def bin_to_freq(self, bin_k: int) -> float:
        """Converts discrete hardware bin index k to frequency in Hz."""
        df = self.fs / float(self.fft_len)
        return float(bin_k) * df

    # =========================================================================
    # 2. Filter Configuration & Custom Range Mapping
    # =========================================================================

    def set_bandpass(self, center_hz: float, delta_hz: float):
        """
        Configures a bandpass filter over (center_hz - delta_hz, center_hz + delta_hz).
        :param center_hz: Center / carrier frequency in Hz (f0).
        :param delta_hz: Half-bandwidth in Hz (Delta f).
        """
        f_start = max(0.0, center_hz - delta_hz)
        f_stop = min(self.fs / 2.0, center_hz + delta_hz)

        k_start = self.freq_to_bin(f_start)
        k_stop = self.freq_to_bin(f_stop)

        if k_start == k_stop:
            k_stop = min(self.fft_len // 2, k_start + 1)

        self._set_raw_filter(mode="bandpass", k_start=k_start, k_stop=k_stop)

    def set_lowpass(self, cutoff_hz: float):
        """Passes all frequencies below cutoff_hz (k_eff <= k_stop)."""
        k_stop = self.freq_to_bin(cutoff_hz)
        self._set_raw_filter(mode="lowpass", k_start=0, k_stop=k_stop)

    def set_highpass(self, cutoff_hz: float):
        """Passes all frequencies above cutoff_hz (k_eff >= k_start)."""
        k_start = self.freq_to_bin(cutoff_hz)
        self._set_raw_filter(mode="highpass", k_start=k_start, k_stop=self.fft_len // 2)

    def set_notch(self, center_hz: float, delta_hz: float):
        """Zeroes out the band (center_hz - delta_hz, center_hz + delta_hz)."""
        f_start = max(0.0, center_hz - delta_hz)
        f_stop = min(self.fs / 2.0, center_hz + delta_hz)

        k_start = self.freq_to_bin(f_start)
        k_stop = self.freq_to_bin(f_stop)

        self._set_raw_filter(mode="notch", k_start=k_start, k_stop=k_stop)

    def _set_raw_filter(self, mode: str, k_start: int, k_stop: int):
        """Writes hardware registers for cutoff bins and mode."""
        mode_clean = mode.lower().strip()
        if mode_clean not in self.MODE_MAP:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {list(self.MODE_MAP.keys())}")

        mode_bits = self.MODE_MAP[mode_clean]

        # Clamp bins within [0, N/2]
        k_max = self.fft_len // 2
        k_start_clamped = max(0, min(k_max, int(k_start)))
        k_stop_clamped = max(0, min(k_max, int(k_stop)))

        # Update registers
        self.mmio.write(self.REG_BIN_START, k_start_clamped)
        self.mmio.write(self.REG_BIN_STOP, k_stop_clamped)

        # Enable filter with selected mode
        ctrl_val = mode_bits | self.BIT_ENABLE
        self.mmio.write(self.REG_CTRL, ctrl_val)

    def bypass(self):
        """Bypasses hardware spectral masking (all frequency bins passed)."""
        self.mmio.write(self.REG_CTRL, 0x00)

    def set_fft_length(self, n_points: int):
        """Sets the FFT transform length N in hardware (e.g. 512, 1024, 2048)."""
        self.fft_len = int(n_points)
        self.mmio.write(self.REG_FFT_LEN, self.fft_len)

    # =========================================================================
    # 3. Status & Diagnostic Queries
    # =========================================================================

    @property
    def is_enabled(self) -> bool:
        """Returns True if hardware masking is currently active (not bypassed)."""
        return bool(self.mmio.read(self.REG_CTRL) & self.BIT_ENABLE)

    @property
    def mode(self) -> str:
        """Returns the active filter mode name ('lowpass', 'highpass', 'bandpass', 'notch', or 'bypass')."""
        ctrl = self.mmio.read(self.REG_CTRL)
        if not (ctrl & self.BIT_ENABLE):
            return "bypass"
        mode_bits = (ctrl >> 1) & 0x03
        return self.REVERSE_MODE_MAP.get(mode_bits, "unknown")

    def get_config(self) -> Dict:
        """Returns active hardware filter configuration details."""
        k_start = self.mmio.read(self.REG_BIN_START) & 0xFFFF
        k_stop = self.mmio.read(self.REG_BIN_STOP) & 0xFFFF
        n = self.mmio.read(self.REG_FFT_LEN) & 0xFFFF
        df = self.fs / float(n) if n > 0 else 0.0

        return {
            "enabled": self.is_enabled,
            "mode": self.mode,
            "bin_start": k_start,
            "bin_stop": k_stop,
            "freq_start_hz": k_start * df,
            "freq_stop_hz": k_stop * df,
            "fft_len": n,
            "bin_resolution_hz": df,
            "fs_hz": self.fs
        }

    def __repr__(self) -> str:
        cfg = self.get_config()
        if not cfg["enabled"]:
            return f"<SpectralMaskDriver: BYPASS, N={cfg['fft_len']}, fs={cfg['fs_hz']:.0f}Hz>"
        return (
            f"<SpectralMaskDriver: Mode={cfg['mode'].upper()}, "
            f"Range=[{cfg['freq_start_hz']:.1f}Hz -> {cfg['freq_stop_hz']:.1f}Hz] "
            f"(Bins {cfg['bin_start']}..{cfg['bin_stop']}), N={cfg['fft_len']}>"
        )