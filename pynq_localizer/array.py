"""
pynq_localizer.array: Core Dual-Microphone Hardware Interface & Continuous Flight Streaming Engine.
Features zero-skew simultaneous sampling (A0 & A1), hardware anti-aliasing decimation,
snapshot capture, and multi-second continuous flight recording.
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
        default_packet_size: int = 2048,
        **kwargs
    ):
        """
        Initializes the dual-microphone hardware overlay.
        Auto-fetches the pinned v1.5.0 bitstream if bitfile_name is None.
        """
        if bitfile_name is None:
            resolved_bit = str(HardwareLoader.get_overlay_path(version=version))
        else:
            resolved_bit = str(Path(bitfile_name).resolve())

        super().__init__(resolved_bit, **kwargs)

        self.packet_size = default_packet_size
        self.current_profile = "audio"
        self.fs_per_ch = 50_000.0  # Default 50 kSPS (M=10)

        # Persistent CMA buffer pool for fast snapshot capture
        self._buf_time = allocate(shape=(self.packet_size,), dtype="u2")

        # Hardware Trigger & Decimation Controller
        self.trigger = HardwareTrigger(self)

        # Apply default Full-Audio profile (M=10, 50 kSPS per channel)
        self.set_profile("audio")

    # =========================================================================
    # 1. Operating Profile Configuration
    # =========================================================================

    def set_profile(self, mode: str = "audio", packet_size: Optional[int] = None) -> Dict:
        """
        Dynamically configures FPGA Decimator (M) and sampling rate.
        :param mode: 'audio' (50 kSPS), 'speech' (25 kSPS), 'bass_zoom' (10 kSPS), or 'oscilloscope' (500 kSPS).
        :param packet_size: Optional manual override for snapshot packet size (e.g. 1024, 2048, 4096).
        """
        mode_clean = mode.lower().strip()
        base_cfg = self.PROFILES.get(mode_clean, self.PROFILES["audio"])
        m_val = base_cfg["m"]
        pkt_val = packet_size if packet_size is not None else self.packet_size

        # 1. Update Hardware Trigger Registers
        self.trigger.set_decimation(m_val)
        self.trigger.set_packet_size(pkt_val)

        # 2. Re-allocate snapshot buffer if packet size changed
        if self._buf_time is None or len(self._buf_time) != pkt_val:
            if self._buf_time is not None:
                try:
                    self._buf_time.close()
                except Exception:
                    pass
            self._buf_time = allocate(shape=(pkt_val,), dtype="u2")

        # 3. Update driver state
        self.current_profile = mode_clean
        self.packet_size = pkt_val
        self.fs_per_ch = 500_000.0 / float(m_val)

        info = {
            "mode": mode_clean,
            "decimation_M": m_val,
            "sample_rate_hz": self.fs_per_ch,
            "time_window_ms": ((pkt_val // 2) / self.fs_per_ch) * 1000.0,
            "nyquist_bandwidth_hz": self.fs_per_ch / 2.0,
        }
        return info

    def _init_xadc_simultaneous(self):
        """Initializes XADC into simultaneous dual-channel parallel sequencer mode (0.00 µs skew)."""
        if hasattr(self, "xadc_wiz_0"):
            self.xadc_wiz_0.mmio.write(0x304, 0x2000)  # DRP 0x41: Continuous Sequence Mode
            self.xadc_wiz_0.mmio.write(0x320, 0x0000)  # DRP 0x48: Disable internal temp/vcc channels
            self.xadc_wiz_0.mmio.write(0x324, 0x0202)  # DRP 0x49: Enable Vaux1 (A0) & Vaux9 (A1)

    # =========================================================================
    # 2. Synchronous Snapshot Capture
    # =========================================================================

    def capture_frame(
        self,
        crop_startup_samples: int = 8,
        timeout: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Captures a single synchronous dual-channel audio frame from A0 (Mic 1) and A1 (Mic 2).
        :return: (voltages_mic1_a0, voltages_mic2_a1) arrays of length packet_size / 2.
        """
        self._init_xadc_simultaneous()

        # Reset DMA channel
        self.axi_dma_0.mmio.write(0x30, 0x04)
        time.sleep(0.002)
        self.axi_dma_0.recvchannel.start()

        # Arm DMA transfer
        self.axi_dma_0.recvchannel.transfer(self._buf_time)
        self.trigger.arm()

        # Poll with timeout
        start = time.time()
        while time.time() - start < timeout:
            if self.axi_dma_0.recvchannel.idle:
                raw_samples = np.array(self._buf_time)

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

        raise TimeoutError(f"Snapshot capture timed out after {timeout} seconds. Check trigger threshold.")

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

        # Pre-allocate output arrays in system memory
        raw_interleaved = np.empty(num_chunks * chunk_size, dtype=np.uint16)
        chunk_buf = allocate(shape=(chunk_size,), dtype="u2")

        # Reset DMA
        self.axi_dma_0.mmio.write(0x30, 0x04)
        time.sleep(0.002)
        self.axi_dma_0.recvchannel.start()

        print(f"[FlightRecorder] Recording {duration_sec:.2f}s ({self.fs_per_ch:.0f} SPS dual stream, {num_chunks} DMA blocks)...")

        try:
            write_ptr = 0
            self.trigger.arm()

            for _ in range(num_chunks):
                self.axi_dma_0.recvchannel.transfer(chunk_buf)
                self.trigger.arm()

                # Wait for DMA chunk completion
                t0 = time.time()
                while not self.axi_dma_0.recvchannel.idle:
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
            # Restore default snapshot packet size
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

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()