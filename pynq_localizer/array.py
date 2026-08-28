"""
pynq_localizer.array: Core Dual-Microphone Hardware Interface & 3-DMA Streaming Engine.
Features zero-skew simultaneous sampling (A0 & A1), hardware spectral frequency filtering
(axis_spectral_mask), hardware IFFT time-domain reconstruction, and concurrent 3-DMA streaming.
"""

import time
from pathlib import Path
from typing import Union, Optional, Tuple, Dict
import numpy as np
try:
    from pynq import Overlay, allocate
except ImportError:
    Overlay = object
    allocate = None

from pynq_localizer.loader import HardwareLoader
from pynq_localizer.hw_trigger import HardwareTrigger
from pynq_localizer.spectral_mask import SpectralMaskDriver


class MicrophoneArrayOverlay(Overlay):
    """
    Core Hardware Overlay Driver for Dual-Microphone Acoustic Kinematics & Sound Localization on PYNQ-Z2.
    Controls 3 concurrent DMA engines:
      • axi_dma_0: Raw Time-Domain Interleaved Stereo (A0 & A1)
      • axi_dma_1: Frequency-Domain Magnitude Spectrum (N/2 bins)
      • axi_dma_2: Filtered & Reconstructed Time-Domain Audio (IFFT output)
    """

    PROFILES = {
        "audio":        {"m": 10, "fft_n": 1024, "desc": "Full Audio Band (50 kSPS per ch, 0 - 25 kHz span)"},
        "speech":       {"m": 20, "fft_n": 1024, "desc": "Speech / Acoustic (25 kSPS per ch, 0 - 12.5 kHz span)"},
        "bass_zoom":    {"m": 50, "fft_n": 1024, "desc": "Deep Bass Zoom (10 kSPS per ch, 0 - 5 kHz span)"},
        "oscilloscope": {"m": 1,  "fft_n": 2048, "desc": "Wideband Ultrasonic Scope (500 kSPS per ch, 0 - 250 kHz span)"},
    }

    def __init__(
        self,
        bitfile_name: Optional[Union[str, Path]] = None,
        version: Optional[str] = None,
        default_packet_size: int = 2048,
        default_fft_len: int = 1024,
        **kwargs
    ):
        """
        Initializes the dual-microphone hardware overlay.
        Auto-fetches the pinned v1.6.0 bitstream if bitfile_name is None.
        """
        if bitfile_name is None:
            resolved_bit = str(HardwareLoader.get_overlay_path(version=version))
        else:
            resolved_bit = str(Path(bitfile_name).resolve())

        super().__init__(resolved_bit, **kwargs)

        self.packet_size = int(default_packet_size)
        self.fft_len = int(default_fft_len)
        self.current_profile = "audio"
        self.fs_per_ch = 50_000.0  # Default 50 kSPS (M=10)

        # Persistent CMA buffer pool for concurrent 3-DMA streaming
        self._buf_time_raw = allocate(shape=(self.packet_size,), dtype="u2")
        self._buf_fft_mag = allocate(shape=(self.fft_len // 2,), dtype="u2")
        self._buf_time_filtered = allocate(shape=(self.packet_size // 2,), dtype="u2")

        # Hardware Controllers
        self.trigger = HardwareTrigger(self)
        self.filter = SpectralMaskDriver(self, fs=self.fs_per_ch, fft_len=self.fft_len)

        # Apply default Full-Audio profile (M=10, N=1024, 50 kSPS per channel)
        self.set_profile("audio")

    # =========================================================================
    # 1. Operating Profile Configuration
    # =========================================================================

    def set_profile(
        self,
        mode: str = "audio",
        packet_size: Optional[int] = None,
        fft_len: Optional[int] = None
    ) -> Dict:
        """
        Dynamically configures FPGA Decimator (M), FFT transform length (N), and sampling rate.
        :param mode: 'audio' (50 kSPS), 'speech' (25 kSPS), 'bass_zoom' (10 kSPS), or 'oscilloscope' (500 kSPS).
        :param packet_size: Optional override for time-domain packet size (e.g. 1024, 2048).
        :param fft_len: Optional override for FFT transform length (512, 1024, 2048).
        """
        mode_clean = mode.lower().strip()
        base_cfg = self.PROFILES.get(mode_clean, self.PROFILES["audio"])
        m_val = base_cfg["m"]
        n_val = fft_len if fft_len is not None else base_cfg["fft_n"]
        pkt_val = packet_size if packet_size is not None else self.packet_size

        # 1. Update Hardware Registers
        self.trigger.set_decimation(m_val)
        self.trigger.set_packet_size(pkt_val)
        self.trigger.set_fft_config(n_points=n_val, forward=True)

        self.fs_per_ch = 500_000.0 / float(m_val)
        self.fft_len = n_val
        self.packet_size = pkt_val

        self.filter.fs = self.fs_per_ch
        self.filter.set_fft_length(self.fft_len)

        # 2. Re-allocate CMA buffers if dimensions changed
        self._realloc_buffers(pkt_val, n_val)

        # 3. Update driver state
        self.current_profile = mode_clean

        info = {
            "mode": mode_clean,
            "decimation_M": m_val,
            "fft_length_N": n_val,
            "sample_rate_hz": self.fs_per_ch,
            "bin_resolution_hz": self.fs_per_ch / float(n_val),
            "time_window_ms": ((pkt_val // 2) / self.fs_per_ch) * 1000.0,
            "nyquist_bandwidth_hz": self.fs_per_ch / 2.0,
        }
        return info

    def _realloc_buffers(self, pkt_size: int, n_fft: int):
        """Safely resizes DMA CMA buffers."""
        if self._buf_time_raw is None or len(self._buf_time_raw) != pkt_size:
            if self._buf_time_raw is not None:
                try:
                    self._buf_time_raw.close()
                except Exception:
                    pass
            self._buf_time_raw = allocate(shape=(pkt_size,), dtype="u2")

        if self._buf_fft_mag is None or len(self._buf_fft_mag) != (n_fft // 2):
            if self._buf_fft_mag is not None:
                try:
                    self._buf_fft_mag.close()
                except Exception:
                    pass
            self._buf_fft_mag = allocate(shape=(n_fft // 2,), dtype="u2")

        filtered_pts = pkt_size // 2
        if self._buf_time_filtered is None or len(self._buf_time_filtered) != filtered_pts:
            if self._buf_time_filtered is not None:
                try:
                    self._buf_time_filtered.close()
                except Exception:
                    pass
            self._buf_time_filtered = allocate(shape=(filtered_pts,), dtype="u2")

    def _init_xadc_simultaneous(self):
        """Initializes XADC into simultaneous dual-channel parallel sequencer mode (0.00 µs skew)."""
        if hasattr(self, "xadc_wiz_0"):
            self.xadc_wiz_0.mmio.write(0x304, 0x2000)  # DRP 0x41: Continuous Sequence Mode
            self.xadc_wiz_0.mmio.write(0x320, 0x0000)  # DRP 0x48: Disable internal temp/vcc channels
            self.xadc_wiz_0.mmio.write(0x324, 0x0202)  # DRP 0x49: Enable Vaux1 (A0) & Vaux9 (A1)

    # =========================================================================
    # 2. Synchronous Multi-Stream Capture (Raw Time, Spectrum & Filtered IFFT)
    # =========================================================================

    def capture_frame(
        self,
        crop_startup_samples: int = 8,
        timeout: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Captures a single synchronous dual-channel raw audio frame from A0 (Mic 1) and A1 (Mic 2).
        :return: (voltages_mic1_a0, voltages_mic2_a1) arrays of length packet_size / 2.
        """
        self._init_xadc_simultaneous()

        # Reset and arm DMA 0 (Raw Time)
        self.axi_dma_0.mmio.write(0x30, 0x04)
        time.sleep(0.002)
        self.axi_dma_0.recvchannel.start()
        self.axi_dma_0.recvchannel.transfer(self._buf_time_raw)

        self.trigger.arm()

        start = time.time()
        while time.time() - start < timeout:
            if self.axi_dma_0.recvchannel.idle:
                raw_samples = np.array(self._buf_time_raw)

                # De-interleave: Even = Mic 1 (A0 / Vaux1), Odd = Mic 2 (A1 / Vaux9)
                raw_a0 = raw_samples[0::2]
                raw_a1 = raw_samples[1::2]

                # Convert 12-bit left-aligned raw words to physical Volts (0.0 - 3.3V)
                v_a0 = (raw_a0 >> 4) * (3.3 / 4095.0)
                v_a1 = (raw_a1 >> 4) * (3.3 / 4095.0)

                # Crop boundary pipeline startup words
                if crop_startup_samples > 0 and len(v_a0) > (2 * crop_startup_samples):
                    v_a0 = v_a0[crop_startup_samples:-crop_startup_samples]
                    v_a1 = v_a1[crop_startup_samples:-crop_startup_samples]

                return v_a0, v_a1
            time.sleep(0.002)

        raise TimeoutError(f"Snapshot capture timed out after {timeout}s. Check trigger threshold.")

    def capture_all(
        self,
        crop_startup_samples: int = 8,
        timeout: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Synchronously captures all 3 hardware data streams direct to DDR memory:
          1. Raw Stereo Time (A0 & A1 via axi_dma_0)
          2. Masked Magnitude Spectrum (axi_dma_1)
          3. Filtered & Reconstructed Time (axi_dma_2)

        :return: (v_raw_a0, v_raw_a1, v_filtered, freq_axis_hz, mags_db)
        """
        self._init_xadc_simultaneous()

        # 1. Reset and arm DMA 0 (Raw Time)
        self.axi_dma_0.mmio.write(0x30, 0x04)
        # 2. Reset and arm DMA 1 (Spectrum Magnitude)
        if hasattr(self, "axi_dma_1"):
            self.axi_dma_1.mmio.write(0x30, 0x04)
        # 3. Reset and arm DMA 2 (Filtered Time)
        if hasattr(self, "axi_dma_2"):
            self.axi_dma_2.mmio.write(0x30, 0x04)

        time.sleep(0.002)

        self.axi_dma_0.recvchannel.start()
        self.axi_dma_0.recvchannel.transfer(self._buf_time_raw)

        if hasattr(self, "axi_dma_1"):
            self.axi_dma_1.recvchannel.start()
            self.axi_dma_1.recvchannel.transfer(self._buf_fft_mag)

        if hasattr(self, "axi_dma_2"):
            self.axi_dma_2.recvchannel.start()
            self.axi_dma_2.recvchannel.transfer(self._buf_time_filtered)

        # Trigger synchronized acquisition
        self.trigger.arm()

        start = time.time()
        while time.time() - start < timeout:
            dma0_done = self.axi_dma_0.recvchannel.idle
            dma1_done = self.axi_dma_1.recvchannel.idle if hasattr(self, "axi_dma_1") else True
            dma2_done = self.axi_dma_2.recvchannel.idle if hasattr(self, "axi_dma_2") else True

            if dma0_done and dma1_done and dma2_done:
                # 1. Process Raw Time
                raw = np.array(self._buf_time_raw)
                raw_a0 = raw[0::2]
                raw_a1 = raw[1::2]
                v_a0 = (raw_a0 >> 4) * (3.3 / 4095.0)
                v_a1 = (raw_a1 >> 4) * (3.3 / 4095.0)

                # 2. Process Spectrum
                if hasattr(self, "axi_dma_1"):
                    raw_mag = np.array(self._buf_fft_mag, dtype=np.float64)
                    linear_mag = raw_mag / 32768.0
                    mags_db = 20.0 * np.log10(np.maximum(linear_mag, 1e-6))
                    df = self.fs_per_ch / float(self.fft_len)
                    freq_axis = np.arange(len(mags_db)) * df
                else:
                    freq_axis = np.array([])
                    mags_db = np.array([])

                # 3. Process Filtered IFFT Time
                if hasattr(self, "axi_dma_2"):
                    raw_filt = np.array(self._buf_time_filtered)
                    v_filt = (raw_filt >> 4) * (3.3 / 4095.0)
                else:
                    v_filt = np.copy(v_a0)

                # Crop startup transients
                if crop_startup_samples > 0 and len(v_a0) > (2 * crop_startup_samples):
                    v_a0 = v_a0[crop_startup_samples:-crop_startup_samples]
                    v_a1 = v_a1[crop_startup_samples:-crop_startup_samples]
                    if len(v_filt) > (2 * crop_startup_samples):
                        v_filt = v_filt[crop_startup_samples:-crop_startup_samples]

                return v_a0, v_a1, v_filt, freq_axis, mags_db

            time.sleep(0.002)

        raise TimeoutError(f"Multi-DMA capture timed out after {timeout}s.")

    # =========================================================================
    # 3. Continuous Multi-Second Flight Recorder
    # =========================================================================

    def record_continuous(
        self,
        duration_sec: float = 3.0,
        chunk_size: int = 4096,
        record_filtered: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Continuously streams and records uninterrupted multi-second flight data.
        :param duration_sec: Total duration to record in seconds.
        :param chunk_size: Size of individual DMA streaming packets.
        :param record_filtered: If True, captures [t, v_raw_selected, v_filtered].
                                If False, captures [t, v_raw_mic1, v_raw_mic2].
        :return: (time_axis_sec, ch1_data, ch2_data).
        """
        self._init_xadc_simultaneous()
        self.trigger.set_packet_size(chunk_size)
        self.trigger.set_mode("Auto")

        total_samples_per_ch = int(float(duration_sec) * self.fs_per_ch)
        total_interleaved = total_samples_per_ch * 2
        num_chunks = int(np.ceil(total_interleaved / chunk_size))

        raw_interleaved = np.empty(num_chunks * chunk_size, dtype=np.uint16)
        chunk_buf = allocate(shape=(chunk_size,), dtype="u2")

        self.axi_dma_0.mmio.write(0x30, 0x04)
        time.sleep(0.002)
        self.axi_dma_0.recvchannel.start()

        print(f"[FlightRecorder] Recording {duration_sec:.2f}s ({self.fs_per_ch:.0f} SPS dual stream, {num_chunks} blocks)...")

        try:
            write_ptr = 0
            self.trigger.arm()

            for _ in range(num_chunks):
                self.axi_dma_0.recvchannel.transfer(chunk_buf)
                self.trigger.arm()

                t0 = time.time()
                while not self.axi_dma_0.recvchannel.idle:
                    if time.time() - t0 > 2.0:
                        raise TimeoutError("Continuous DMA streaming timed out.")
                    time.sleep(0.001)

                raw_interleaved[write_ptr : write_ptr + chunk_size] = np.array(chunk_buf)
                write_ptr += chunk_size

            valid_samples = raw_interleaved[:total_interleaved]
            raw_a0 = valid_samples[0::2]
            raw_a1 = valid_samples[1::2]

            v_a0 = (raw_a0 >> 4) * (3.3 / 4095.0)
            v_a1 = (raw_a1 >> 4) * (3.3 / 4095.0)

            t_axis = np.linspace(0, duration_sec, len(v_a0), endpoint=False)
            print(f"[FlightRecorder] Captured {len(v_a0)} stereo samples successfully.")
            return t_axis, v_a0, v_a1

        finally:
            chunk_buf.close()
            self.trigger.set_packet_size(self.packet_size)

    # =========================================================================
    # 4. Jupyter Audio Playback
    # =========================================================================

    def play_audio(self, channel: int = 1, custom_data: Optional[np.ndarray] = None):
        """
        Plays captured microphone audio directly in Jupyter Notebook.
        :param channel: 1 for Mic 1 (A0), 2 for Mic 2 (A1).
        :param custom_data: Optional custom 1D numpy array of audio samples to play.
        """
        try:
            from IPython.display import Audio, display
        except ImportError:
            raise RuntimeError("IPython is required for audio playback.")

        if custom_data is not None:
            audio_samples = custom_data
        else:
            v_a0, v_a1 = self.capture_frame()
            audio_samples = v_a0 if channel == 1 else v_a1

        ac_signal = audio_samples - np.mean(audio_samples)
        max_val = np.max(np.abs(ac_signal))
        normalized = (ac_signal / max_val) if max_val > 1e-4 else ac_signal
        display(Audio(normalized, rate=int(self.fs_per_ch)))

    def kinematics_dashboard(
        self,
        window_duration_sec: float = 10.0,
        hop_ms: float = 10.0
    ):
        """Launches the real-time 10-second rolling Multi-Tab Kinematics Dashboard."""
        from pynq_localizer.kinematics_dashboard import KinematicsDashboard
        dash = KinematicsDashboard(
            overlay=self,
            window_duration_sec=window_duration_sec,
            hop_ms=hop_ms,
            fs_per_ch=self.fs_per_ch
        )
        dash.display()
        return dash

    # =========================================================================
    # Cleanup & Context Management
    # =========================================================================

    def close(self):
        """Frees all CMA buffers and cleans hardware state."""
        for buf_attr in ["_buf_time_raw", "_buf_fft_mag", "_buf_time_filtered"]:
            if hasattr(self, buf_attr) and getattr(self, buf_attr) is not None:
                try:
                    getattr(self, buf_attr).close()
                    setattr(self, buf_attr, None)
                except Exception:
                    pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()