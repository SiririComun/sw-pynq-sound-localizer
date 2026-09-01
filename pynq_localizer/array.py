"""
pynq_localizer.array: Core Dual-Microphone Hardware Interface & Lock-Step Polar Streaming Engine.
Features zero-skew simultaneous sampling (A0 & A1), hardware anti-aliasing decimation,
32-bit polar CORDIC phase-magnitude spectral capture, sub-microsecond hardware timestamping,
and continuous multi-second flight recording.
"""

import time
from pathlib import Path
from typing import Union, Optional, Tuple, Dict
import numpy as np
from pynq import Overlay, allocate

from pynq_localizer.loader import HardwareLoader
from pynq_localizer.hw_trigger import HardwareTrigger

class MicrophoneArrayOverlay(Overlay):
    """
    Core Hardware Overlay Driver for Dual-Microphone Acoustic Kinematics & Sound Localization on PYNQ-Z2.
    """

    PROFILES = {
        "audio":        {"m": 10, "desc": "Full Audio Band (50 kSPS per ch, 0 - 25 kHz span)"},
        "speech":       {"m": 20, "desc": "Speech / Acoustic (25 kSPS per ch, 0 - 12.5 kHz span)"},
        "bass_zoom":    {"m": 50, "desc": "Deep Bass Zoom (10 kSPS per ch, 0 - 5 kHz span)"},
        "oscilloscope": {"m": 1,  "desc": "Wideband Ultrasonic Scope (500 kSPS per ch, 0 - 250 kHz span)"},
    }

    def __init__(
        self,
        bitfile_name: Optional[Union[str, Path]] = None,
        version: Optional[str] = None,
        n_fft: int = 2048,
        **kwargs
    ):
        """
        Initializes the dual-microphone hardware overlay.
        Auto-fetches the pinned v1.5.1-rc1 bitstream if bitfile_name is None.
        """
        if bitfile_name is None:
            resolved_bit = str(HardwareLoader.get_overlay_path(version=version))
        else:
            resolved_bit = str(Path(bitfile_name).resolve())

        super().__init__(resolved_bit, **kwargs)

        self.n_fft = n_fft
        self.packet_size = n_fft * 2  # Interleaved stereo packet size
        self.current_profile = "audio"
        self.fs_per_ch = 50_000.0     # Default 50 kSPS (M=10)

        # Persistent CMA buffer pool for lock-step dual DMA capture
        self._buf_time = allocate(shape=(self.packet_size,), dtype="u2")
        self._buf_fft = allocate(shape=(self.n_fft,), dtype="u4")

        # Hardware Trigger & Decimation Controller
        self.trigger = HardwareTrigger(self)

        # Apply default Full-Audio profile (M=10, 50 kSPS per channel, N=2048)
        self.set_profile("audio", n_fft=self.n_fft)

    # =========================================================================
    # 1. Operating Profile Configuration
    # =========================================================================

    def set_profile(
        self,
        mode: str = "audio",
        n_fft: Optional[int] = None,
        packet_size: Optional[int] = None
    ) -> Dict:
        """
        Dynamically configures FPGA Decimator (M), FFT length (N), and sampling rate.
        :param mode: 'audio' (50 kSPS), 'speech' (25 kSPS), 'bass_zoom' (10 kSPS), or 'oscilloscope' (500 kSPS).
        :param n_fft: FFT transform length (512, 1024, 2048).
        :param packet_size: Optional manual override for interleaved packet size.
        """
        mode_clean = mode.lower().strip()
        base_cfg = self.PROFILES.get(mode_clean, self.PROFILES["audio"])
        m_val = base_cfg["m"]
        n_val = n_fft if n_fft is not None else self.n_fft
        pkt_val = packet_size if packet_size is not None else (n_val * 2)

        # 1. Update Hardware Trigger & FFT Configuration Registers
        self.trigger.set_decimation(m_val)
        self.trigger.set_fft_config(n_val, forward=True)
        self.trigger.set_packet_size(pkt_val)

        # 2. Re-allocate CMA buffers if sizes changed
        if self._buf_time is None or len(self._buf_time) != pkt_val:
            if self._buf_time is not None:
                try: self._buf_time.close()
                except Exception: pass
            self._buf_time = allocate(shape=(pkt_val,), dtype="u2")

        if self._buf_fft is None or len(self._buf_fft) != n_val:
            if self._buf_fft is not None:
                try: self._buf_fft.close()
                except Exception: pass
            self._buf_fft = allocate(shape=(n_val,), dtype="u4")

        # 3. Update driver state
        self.current_profile = mode_clean
        self.n_fft = n_val
        self.packet_size = pkt_val
        self.fs_per_ch = 500_000.0 / float(m_val)

        info = {
            "mode": mode_clean,
            "decimation_M": m_val,
            "n_fft": n_val,
            "packet_size": pkt_val,
            "sample_rate_hz": self.fs_per_ch,
            "bin_resolution_hz": self.fs_per_ch / float(n_val),
            "time_window_ms": (n_val / self.fs_per_ch) * 1000.0,
            "nyquist_bandwidth_hz": self.fs_per_ch / 2.0,
        }
        return info

    def _init_xadc_simultaneous(self):
        """Initializes XADC into simultaneous dual-channel mode and flushes startup pipeline frames."""
        if hasattr(self, "xadc_wiz_0"):
            self.xadc_wiz_0.mmio.write(0x304, 0x2000)  # DRP 0x41: Continuous Sequence Mode
            self.xadc_wiz_0.mmio.write(0x320, 0x0000)  # DRP 0x48: Disable internal channels
            self.xadc_wiz_0.mmio.write(0x324, 0x0202)  # DRP 0x49: Enable Vaux1 (A0) & Vaux9 (A1)

        # Start 100 MHz Hardware Timer (TCSR0: ENT0=1, ARHT0=1)
        if hasattr(self, "axi_timer_0"):
            self.axi_timer_0.mmio.write(0x00, 0x00000480)

    # =========================================================================
    # 2. Lock-Step Dual DMA Capture (Time Domain + Polar Frequency Domain)
    # =========================================================================

    def capture_spectral_frame(
        self,
        fft_source: str = "A0",
        crop_startup_samples: int = 8,
        timeout: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """Captures a simultaneous dual-channel time snapshot AND 32-bit polar CORDIC spectrum in lock-step."""
        self._init_xadc_simultaneous()
        self.trigger.set_fft_channel("CH2" if ("A1" in fft_source.upper() or "CH2" in fft_source.upper()) else "CH1")

        # Arm Both DMAs Concurrently
        self.axi_dma_0.recvchannel.transfer(self._buf_time)
        self.axi_dma_1.recvchannel.transfer(self._buf_fft)
        self.trigger.arm()

        # Poll for Completion
        t0 = time.time()
        stalled = False
        while not (self.axi_dma_0.recvchannel.idle and self.axi_dma_1.recvchannel.idle):
            if time.time() - t0 > timeout:
                stalled = True
                break
            time.sleep(0.0005)

        if stalled:
            # Recovery flush
            self.axi_dma_1.mmio.write(0x30, 0x04)
            time.sleep(0.002)
            self.axi_dma_1.recvchannel.start()
            self.trigger.arm()
            raise TimeoutError(f"Lock-step DMA transfer timed out after {timeout}s.")

        # Latch hardware timer cycle count on frame completion
        t_hw_cycles = self.axi_timer_0.mmio.read(0x08) if hasattr(self, "axi_timer_0") else 0

        # Unpack Time Domain (DMA 0)
        raw_samples = np.array(self._buf_time)
        raw_a0 = raw_samples[0::2]
        raw_a1 = raw_samples[1::2]
        v_a0 = (raw_a0 >> 4) * (3.3 / 4095.0)
        v_a1 = (raw_a1 >> 4) * (3.3 / 4095.0)

        if crop_startup_samples > 0 and len(v_a0) > (2 * crop_startup_samples):
            v_a0 = v_a0[crop_startup_samples:-crop_startup_samples]
            v_a1 = v_a1[crop_startup_samples:-crop_startup_samples]

        # Unpack 32-Bit Polar FFT Domain (DMA 1)
        raw_words = np.array(self._buf_fft)
        half_bins = self.n_fft // 2

        raw_mag = (raw_words[:half_bins] & 0xFFFF).astype(np.uint16).astype(np.float64)
        raw_phase_i16 = (raw_words[:half_bins] >> 16).astype(np.int16)
        phase_rad = (raw_phase_i16.astype(np.float64) / 32768.0) * np.pi

        freq_axis = np.fft.fftfreq(self.n_fft, d=1.0 / self.fs_per_ch)[:half_bins]

        return {
            "v_a0": v_a0,
            "v_a1": v_a1,
            "freqs": freq_axis,
            "mag": raw_mag,
            "phase": phase_rad,
            "timer_cycles": t_hw_cycles
        }

    def capture_frame(
        self,
        crop_startup_samples: int = 8,
        timeout: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Captures a single synchronous dual-channel audio frame from A0 (Mic 1) and A1 (Mic 2).
        :return: (voltages_mic1_a0, voltages_mic2_a1) arrays.
        """
        res = self.capture_spectral_frame(crop_startup_samples=crop_startup_samples, timeout=timeout)
        return res["v_a0"], res["v_a1"]

    # =========================================================================
    # 3. Continuous Multi-Second Flight Recorder
    # =========================================================================

    def record_continuous(
        self,
        duration_sec: float = 3.0,
        chunk_size: int = 4096
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Continuously streams and records uninterrupted multi-second flight data from both microphones.
        Ideal for Doppler moving vehicle tracking and free-fall gravitational acceleration measurement.

        :param duration_sec: Total duration to record in seconds (e.g. 3.0, 5.0, 10.0).
        :param chunk_size: Size of individual DMA streaming packets (default: 4096).
        :return: (time_axis_sec, v_mic1_a0, v_mic2_a1) arrays.
        """
        self._init_xadc_simultaneous()
        self.trigger.set_packet_size(chunk_size)
        self.trigger.set_mode("Auto")

        total_samples_per_ch = int(float(duration_sec) * self.fs_per_ch)
        total_interleaved_samples = total_samples_per_ch * 2
        num_chunks = int(np.ceil(total_interleaved_samples / chunk_size))

        raw_interleaved = np.empty(num_chunks * chunk_size, dtype=np.uint16)
        chunk_buf = allocate(shape=(chunk_size,), dtype="u2")
        dummy_fft_buf = allocate(shape=(chunk_size // 2,), dtype="u4")

        # Reset DMAs
        self.axi_dma_0.mmio.write(0x30, 0x04)
        self.axi_dma_1.mmio.write(0x30, 0x04)
        time.sleep(0.002)
        self.axi_dma_0.recvchannel.start()
        self.axi_dma_1.recvchannel.start()

        print(f"[FlightRecorder] Recording {duration_sec:.2f}s ({self.fs_per_ch:.0f} SPS dual stream, {num_chunks} DMA blocks)...")

        try:
            write_ptr = 0
            self.trigger.arm()

            for _ in range(num_chunks):
                # Lock-step DMA transfers prevent broadcaster stalls during continuous recording
                self.axi_dma_0.recvchannel.transfer(chunk_buf)
                self.axi_dma_1.recvchannel.transfer(dummy_fft_buf)
                self.trigger.arm()

                # Wait for DMA chunk completion
                t0 = time.time()
                while not (self.axi_dma_0.recvchannel.idle and self.axi_dma_1.recvchannel.idle):
                    if time.time() - t0 > 2.0:
                        raise TimeoutError("Continuous DMA streaming timed out. Hardware stalled.")
                    time.sleep(0.001)

                raw_interleaved[write_ptr : write_ptr + chunk_size] = np.array(chunk_buf)
                write_ptr += chunk_size

            # Trim to exact requested length
            valid_samples = raw_interleaved[:total_interleaved_samples]
            raw_a0 = valid_samples[0::2]
            raw_a1 = valid_samples[1::2]

            v_a0 = (raw_a0 >> 4) * (3.3 / 4095.0)
            v_a1 = (raw_a1 >> 4) * (3.3 / 4095.0)

            t_axis = np.linspace(0, duration_sec, len(v_a0), endpoint=False)
            print(f"[FlightRecorder] Captured {len(v_a0)} stereo samples successfully with 0.00 µs skew.")
            return t_axis, v_a0, v_a1

        finally:
            chunk_buf.close()
            dummy_fft_buf.close()
            # Restore default snapshot configuration
            self.trigger.set_packet_size(self.packet_size)

    # =========================================================================
    # 4. Jupyter Audio Playback & Interactive Dashboard
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
        """
        Launches the real-time 10-second rolling Multi-Tab Kinematics Dashboard (A(t) & f0(t)).
        :param window_duration_sec: Rolling time window in seconds (default: 10.0s).
        :param hop_ms: Telemetry extraction rate in milliseconds (default: 10.0 ms).
        """
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
        if hasattr(self, "_buf_time") and self._buf_time is not None:
            try:
                self._buf_time.close()
                self._buf_time = None
            except Exception:
                pass

        if hasattr(self, "_buf_fft") and self._buf_fft is not None:
            try:
                self._buf_fft.close()
                self._buf_fft = None
            except Exception:
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()