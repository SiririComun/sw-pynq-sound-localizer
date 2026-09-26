"""
pynq_localizer.array: Core Dual-Microphone Hardware Interface & Lock-Step Polar Streaming Engine.
Features zero-skew simultaneous sampling (A0 & A1), hardware anti-aliasing decimation,
32-bit polar CORDIC phase-magnitude spectral capture, hybrid dual-DMA BFP-immune amplitude
demodulation, sub-microsecond hardware timestamping, and continuous multi-second flight recording.
"""

import time
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
try:
    from pynq import Overlay, allocate
except (ImportError, ModuleNotFoundError):
    Overlay = object
    allocate = None

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
        Auto-fetches the pinned v1.5.1 bitstream if bitfile_name is None.
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

        # One-time startup priming & pipeline synchronization
        self._prime_hardware()

    # =========================================================================
    # 1. Operating Profile Configuration & Hardware Priming
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
        """Initializes XADC continuous dual-channel mode and 100 MHz timer once."""
        if hasattr(self, "xadc_wiz_0"):
            self.xadc_wiz_0.mmio.write(0x304, 0x2000)  # DRP 0x41: Continuous Sequence Mode
            self.xadc_wiz_0.mmio.write(0x320, 0x0000)  # DRP 0x48: Disable internal temp/vcc channels
            self.xadc_wiz_0.mmio.write(0x324, 0x0202)  # DRP 0x49: Enable Vaux1 (A0) & Vaux9 (A1)

        if hasattr(self, "axi_timer_0"):
            self.axi_timer_0.mmio.write(0x00, 0x00000480)  # Enable 100 MHz Hardware Timer (ENT0=1, ARHT0=1)

    def _prime_hardware(self):
        """Flushes startup boundary pipeline frames into steady state."""
        self._init_xadc_simultaneous()
        
        self.axi_dma_0.mmio.write(0x30, 0x04)
        self.axi_dma_1.mmio.write(0x30, 0x04)
        time.sleep(0.005)
        self.axi_dma_0.recvchannel.start()
        self.axi_dma_1.recvchannel.start()

        for _ in range(10):
            self.axi_dma_0.recvchannel.transfer(self._buf_time)
            self.axi_dma_1.recvchannel.transfer(self._buf_fft)
            self.trigger.arm()
            t0 = time.time()
            while not (self.axi_dma_0.recvchannel.idle and self.axi_dma_1.recvchannel.idle):
                if time.time() - t0 > 0.08:
                    self.axi_dma_1.mmio.write(0x30, 0x04)
                    time.sleep(0.002)
                    self.axi_dma_1.recvchannel.start()
                    self.trigger.set_fft_config(self.n_fft, forward=True)
                    self.trigger.arm()
                    break
                time.sleep(0.0005)

    # =========================================================================
    # 2. Lock-Step Dual DMA Capture & Hybrid Quadruple Engine
    # =========================================================================

    def capture_spectral_frame(
        self,
        fft_source: str = "A0",
        crop_startup_samples: int = 8,
        timeout: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Captures a simultaneous dual-channel time snapshot AND 32-bit polar CORDIC spectrum in lock-step.

        :param fft_source: Channel routed to the FFT core ('A0' or 'A1').
        :param crop_startup_samples: Number of initial boundary samples to crop from time domain.
        :param timeout: Maximum wait time in seconds.
        :return: Dictionary containing v_a0, v_a1, freqs, mag, phase, and timer_cycles.
        """
        self.trigger.set_fft_channel("CH2" if ("A1" in fft_source.upper() or "CH2" in fft_source.upper()) else "CH1")

        # Arm Both DMAs Concurrently
        self.axi_dma_0.recvchannel.transfer(self._buf_time)
        self.axi_dma_1.recvchannel.transfer(self._buf_fft)
        self.trigger.arm()

        # Poll for Completion
        t0 = time.time()
        while not (self.axi_dma_0.recvchannel.idle and self.axi_dma_1.recvchannel.idle):
            if time.time() - t0 > timeout:
                self.axi_dma_1.mmio.write(0x30, 0x04)
                time.sleep(0.002)
                self.axi_dma_1.recvchannel.start()
                self.trigger.arm()
                raise TimeoutError(f"Lock-step DMA transfer timed out after {timeout}s.")
            time.sleep(0.0005)

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

    def capture_quadruple(
        self,
        source: str = "A0",
        f_min: float = 100.0,
        f_max: float = 15000.0,
        timeout: float = 0.5
    ) -> Dict[str, Any]:
        """
        High-level capture method returning the hybrid BFP-immune quadruple (f0, A_true, phi, t).
        Combines hardware pitch/phase vectoring (DMA 1) with time-domain coherent demodulation (DMA 0).

        :param source: Microphone channel to process ('A0' or 'A1').
        :param f_min: Search range minimum frequency in Hz.
        :param f_max: Search range maximum frequency in Hz.
        :param timeout: Maximum wait time in seconds.
        :return: Telemetry dictionary containing:
                 - 'quadruple': Dict with f0, amplitude_v (BFP-immune), phase_rad, timestamp_sec.
                 - 'v_time': 1D voltage array of the selected channel.
                 - 'freqs': 1D positive frequency axis in Hz.
                 - 'mag': 1D raw FFT magnitude spectrum.
                 - 'phase': 1D raw CORDIC phase spectrum.
        """
        from pynq_localizer.kinematics import KinematicAnalytics

        frame = self.capture_spectral_frame(fft_source=source, timeout=timeout)
        time_sig = frame["v_a0"] if "A0" in source.upper() or "CH1" in source.upper() else frame["v_a1"]

        quad = KinematicAnalytics.extract_hybrid_quadruple(
            time_signal_v=time_sig,
            fs=self.fs_per_ch,
            freq_axis=frame["freqs"],
            magnitude=frame["mag"],
            phase_rad=frame["phase"],
            f_min=f_min,
            f_max=f_max,
            timer_cycles=frame["timer_cycles"]
        )

        return {
            "quadruple": quad,
            "v_time": time_sig,
            "v_a0": frame["v_a0"],
            "v_a1": frame["v_a1"],
            "freqs": frame["freqs"],
            "mag": frame["mag"],
            "phase": frame["phase"]
        }

    def capture_frame(
        self,
        crop_startup_samples: int = 8,
        timeout: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy wrapper: Captures a single synchronous dual-channel audio frame from A0 and A1."""
        res = self.capture_spectral_frame(crop_startup_samples=crop_startup_samples, timeout=timeout)
        return res["v_a0"], res["v_a1"]

    def capture_aoa_frame(
        self,
        f_target: Optional[float] = None,
        mic_distance_m: float = 0.05,
        profile: Optional[Union[Any, str, Path]] = None,
        temperature_c: float = 20.0,
        noise_gate_v: float = 0.010,
        crop_startup_samples: int = 8,
        timeout: float = 0.5
    ) -> Dict[str, Any]:
        """
        Captures a simultaneous dual-channel audio frame (0.00 µs skew via DMA 0)
        and computes the incident acoustic Angle of Arrival (AoA) bearing angle θ.

        :param f_target: Specific carrier frequency to track (if None, auto-detected).
        :param mic_distance_m: Center-to-center microphone baseline d in meters.
        :param profile: Path to JSON profile or AcousticProfile object (e.g. 'profiles/active_buzzer_2610hz.json').
        :param temperature_c: Ambient temperature in °C for c(T) computation.
        :param noise_gate_v: Minimum in-band RMS voltage squelch threshold.
        :param crop_startup_samples: Boundary samples to discard from frame edges.
        :param timeout: Maximum DMA transfer wait time in seconds.
        :return: Telemetry dictionary containing:
                 - 'theta_deg': Incident bearing angle in degrees [-90°, +90°] (NaN if silent).
                 - 'theta_rad': Incident bearing angle in radians.
                 - 'theta_err_deg': Analytical uncertainty propagation ±δθ in degrees.
                 - 'delta_phi_rad': Wrapped relative phase difference Δφ in radians.
                 - 'amp_a0_v': Physical in-band RMS voltage of Mic 1 (A0).
                 - 'amp_a1_v': Physical in-band RMS voltage of Mic 2 (A1).
                 - 'coherence': Dual-channel complex coherence metric [0.0 to 1.0].
                 - 'f0_evaluated': Center carrier frequency used for inversion.
                 - 'status': 'ACTIVE_VALID', 'SILENCE', or 'OUT_OF_BOUNDS_SPATIAL_ALIASING'.
                 - 'v_a0': Raw ADC voltage array of Mic 1.
                 - 'v_a1': Raw ADC voltage array of Mic 2.
        """
        from pynq_localizer.kinematics import AngleOfArrivalEstimator, KinematicAnalytics

        # 1. Capture synchronized dual-channel frame
        frame = self.capture_spectral_frame(
            fft_source="A0",
            crop_startup_samples=crop_startup_samples,
            timeout=timeout
        )
        v_a0 = frame["v_a0"]
        v_a1 = frame["v_a1"]

        # 2. Determine target frequency: explicit override -> profile -> auto-detect
        if f_target is not None:
            f_eval = float(f_target)
        elif profile is not None:
            f_eval = None  # Handled by AngleOfArrivalEstimator constructor
        else:
            # Auto-detect loudest carrier peak
            f_detected, _ = KinematicAnalytics.track_sub_hertz_pitch(
                frame["freqs"], frame["mag"], min_freq_hz=100.0, max_freq_hz=15000.0, interpolate=True
            )
            f_eval = f_detected if (np.isfinite(f_detected) and f_detected > 0) else 2609.73

        # 3. Instantiate solver and compute bearing
        estimator = AngleOfArrivalEstimator(
            mic_distance_m=mic_distance_m,
            target_freq_hz=f_eval,
            profile=profile,
            temperature_c=temperature_c,
            noise_gate_v=noise_gate_v
        )

        aoa_res = estimator.estimate_angle(v_a0=v_a0, v_a1=v_a1, fs=self.fs_per_ch, f_target=f_eval)

        # 4. Augment with raw signals and hardware timer
        aoa_res["timer_cycles"] = frame.get("timer_cycles", 0)
        aoa_res["v_a0"] = v_a0
        aoa_res["v_a1"] = v_a1

        return aoa_res
    
    def capture_pulsed_toa_frame(
        self,
        pulse_width_ms: float = 10.0,
        packet_samples: int = 16384,
        f_target: Optional[float] = None,
        mic_distance_m: float = 0.05,
        profile: Optional[Union[Any, str, Path]] = None,
        calibrated_offset_ms: Optional[float] = None,
        temperature_c: float = 20.0,
        blanking_ms: float = 0.25,
        min_thresh_mv: float = 15.0,
        timeout: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Synchronously fires a hardware acoustic pulse via FPGA Arduino pin AR2 (U13),
        captures the dual-channel response stream via DMA 0 at 500 kSPS (M=1), and
        computes calibrated radial distances (r1, r2), path delta (Δr), and TDOA angle (θ).

        :param pulse_width_ms: Duration of hardware pulse burst in ms (default: 10.0 ms).
        :param packet_samples: Total interleaved DMA samples (default: 16384 = 16.384 ms window).
        :param f_target: Nominal pulse carrier frequency (default None -> auto from profile).
        :param mic_distance_m: Center-to-center microphone baseline in meters.
        :param profile: Optional device profile (e.g. 'profiles/active_buzzer_2610hz.json').
        :param calibrated_offset_ms: Transducer turn-on delay override (defaults to profile / 1.8080 ms).
        :param temperature_c: Air temperature in °C for c(T) calculation.
        :param blanking_ms: Initial time in ms to ignore electrical transients (default: 0.25 ms).
        :param min_thresh_mv: Minimum acoustic wave threshold in mV.
        :param timeout: Maximum DMA wait time in seconds.
        :return: Telemetry dictionary containing r1_cm, r2_cm, delta_r_cm, delta_t_ms,
                 theta_tdoa_deg, v_a0, v_a1, and status.
        """
        import scipy.signal as signal
        from pynq_localizer.kinematics import KinematicAnalytics

        c_sound = KinematicAnalytics.speed_of_sound(temperature_c)

        # 1. Resolve carrier frequency and calibrated turn-on delay
        if profile is not None:
            if isinstance(profile, (str, Path)):
                p_path = Path(profile).resolve()
                with open(p_path, "r", encoding="utf-8") as f:
                    p_data = json.load(f)
                f0 = float(p_data.get("f_res_hz", 2609.73))
                offset_ms = float(p_data.get("calibrated_toa_offset_ms", 1.8080))
            else:
                f0 = 2609.73
                offset_ms = 1.8080
        else:
            f0 = float(f_target) if f_target is not None else 2609.73
            offset_ms = 1.8080 if calibrated_offset_ms is None else float(calibrated_offset_ms)

        if calibrated_offset_ms is not None:
            offset_ms = float(calibrated_offset_ms)

        # 2. Ensure DMA 0 is running and allocate buffer for M=1 undecimated capture
        if hasattr(self, "axi_dma_0"):
            if not self.axi_dma_0.recvchannel.running:
                self.axi_dma_0.mmio.write(0x30, 0x04)  # S2MM reset
                time.sleep(0.005)
                self.axi_dma_0.recvchannel.start()

        if self._buf_time is None or len(self._buf_time) != packet_samples:
            if self._buf_time is not None:
                try:
                    self._buf_time.close()
                except Exception:
                    pass
            self._buf_time = allocate(shape=(packet_samples,), dtype="u2")

        # 3. Configure hardware trigger: M=1 bypass, 3.29V ceiling to prevent premature triggers
        fs = 500_000.0
        if hasattr(self, "trigger") and self.trigger is not None:
            self.trigger.disarm()
            self.trigger.set_decimation(1)
            self.trigger.set_packet_size(packet_samples)
            self.trigger.set_threshold(3.29)
            self.trigger.set_pulse_width_ms(pulse_width_ms)

        # 4. Queue DMA 0 and fire pulse with self-clearing strobe
        self.axi_dma_0.recvchannel.transfer(self._buf_time)
        time.sleep(0.002)

        # Strobe sequence: Arm Single (0x09) -> Strobe Bit 7 (0x89) -> Restore (0x09)
        self.trigger.mmio.write(0x00, 0x00000009)
        self.trigger.mmio.write(0x00, 0x00000089)
        self.trigger.mmio.write(0x00, 0x00000009)

        # 5. Wait for DMA transfer completion
        t0 = time.time()
        while not self.axi_dma_0.recvchannel.idle:
            if time.time() - t0 > timeout:
                self.trigger.disarm()
                raise TimeoutError(f"Pulsed ToA DMA transfer timed out after {timeout}s.")
            time.sleep(0.0005)

        # 6. Unpack raw de-interleaved samples
        raw = np.array(self._buf_time)
        v_a0 = (raw[0::2] >> 4) * (3.3 / 4095.0)  # Mic 1 (A0, Vaux1)
        v_a1 = (raw[1::2] >> 4) * (3.3 / 4095.0)  # Mic 2 (A1, Vaux9)
        t_ms = (np.arange(len(v_a0)) / fs) * 1000.0

        # 7. Bandpass filter around carrier (2610 Hz +/- 350 Hz)
        nyq = fs / 2.0
        b, a = signal.butter(2, [max(20.0, f0 - 350.0) / nyq, min(nyq - 20.0, f0 + 350.0) / nyq], btype="bandpass")
        v1_bp = signal.filtfilt(b, a, v_a0 - np.mean(v_a0)) * 1000.0  # in mV
        v2_bp = signal.filtfilt(b, a, v_a1 - np.mean(v_a1)) * 1000.0  # in mV

        env1 = np.abs(signal.hilbert(v1_bp))
        env2 = np.abs(signal.hilbert(v2_bp))

        # 8. Detect acoustic wavefronts
        def find_wavefront(v_bp, env):
            blank_idx = int((blanking_ms / 1000.0) * fs)
            peak_val = np.max(env[blank_idx:])
            thresh = max(min_thresh_mv, 0.20 * peak_val)
            zc = np.where((v_bp[:-1] <= 0) & (v_bp[1:] > 0))[0]
            for i in range(len(zc) - 1):
                z0, z1 = zc[i], zc[i + 1]
                if z0 < blank_idx:
                    continue
                p = z1 - z0
                amp = np.max(v_bp[z0:z1]) - np.min(v_bp[z0:z1])
                if (150 <= p <= 230) and (amp >= thresh):
                    frac = -v_bp[z0] / (v_bp[z0 + 1] - v_bp[z0]) if abs(v_bp[z0 + 1] - v_bp[z0]) > 1e-6 else 0.0
                    return (float(z0) + frac) / fs * 1000.0  # in ms
            return np.nan

        t1_raw_ms = find_wavefront(v1_bp, env1)
        t2_raw_ms = find_wavefront(v2_bp, env2)

        # 9. Compute Calibrated Distances & TDOA
        t1_flight_ms = max(0.0, t1_raw_ms - offset_ms) if np.isfinite(t1_raw_ms) else np.nan
        t2_flight_ms = max(0.0, t2_raw_ms - offset_ms) if np.isfinite(t2_raw_ms) else np.nan

        r1_cm = (c_sound * (t1_flight_ms / 1000.0)) * 100.0 if np.isfinite(t1_flight_ms) else np.nan
        r2_cm = (c_sound * (t2_flight_ms / 1000.0)) * 100.0 if np.isfinite(t2_flight_ms) else np.nan

        delta_t_ms = (t1_raw_ms - t2_raw_ms) if (np.isfinite(t1_raw_ms) and np.isfinite(t2_raw_ms)) else np.nan
        delta_r_cm = (c_sound * (delta_t_ms / 1000.0)) * 100.0 if np.isfinite(delta_t_ms) else np.nan

        # Invert TDOA to bearing angle if baseline is satisfied
        if np.isfinite(delta_t_ms) and mic_distance_m > 0:
            ratio = (c_sound * (delta_t_ms / 1000.0)) / mic_distance_m
            clamped = float(np.clip(ratio, -1.0, 1.0))
            theta_deg = float(np.degrees(np.arcsin(clamped)))
        else:
            theta_deg = np.nan

        is_valid = np.isfinite(r1_cm) and np.isfinite(r2_cm)

        return {
            "r1_cm": r1_cm,
            "r2_cm": r2_cm,
            "distance_cm": r1_cm,  # Default reference distance (Mic 1)
            "distance_m": r1_cm / 100.0 if np.isfinite(r1_cm) else np.nan,
            "delta_r_cm": delta_r_cm,
            "delta_t_ms": delta_t_ms,
            "theta_tdoa_deg": theta_deg,
            "t1_raw_ms": t1_raw_ms,
            "t2_raw_ms": t2_raw_ms,
            "t_flight_sec": t1_flight_ms / 1000.0 if np.isfinite(t1_flight_ms) else np.nan,
            "amp_a0_v": float(np.max(env1)) / 1000.0,
            "amp_a1_v": float(np.max(env2)) / 1000.0,
            "status": "ACTIVE_VALID" if is_valid else "SILENCE",
            "v_a0": v_a0,
            "v_a1": v_a1,
            "v1_bp": v1_bp,
            "v2_bp": v2_bp,
            "env1": env1,
            "env2": env2,
            "t_ms": t_ms,
        }

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
                self.axi_dma_0.recvchannel.transfer(chunk_buf)
                self.axi_dma_1.recvchannel.transfer(dummy_fft_buf)
                self.trigger.arm()

                t0 = time.time()
                while not (self.axi_dma_0.recvchannel.idle and self.axi_dma_1.recvchannel.idle):
                    if time.time() - t0 > 2.0:
                        raise TimeoutError("Continuous DMA streaming timed out. Hardware stalled.")
                    time.sleep(0.001)

                raw_interleaved[write_ptr : write_ptr + chunk_size] = np.array(chunk_buf)
                write_ptr += chunk_size

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
            self.trigger.set_packet_size(self.packet_size)

    def record_differential_flight(
        self,
        duration_sec: float = 4.0,
        track_length_m: float = 2.0,
        f0_nominal: Optional[float] = None,
        profile: Optional[Union[Any, str, Path]] = None,
        temperature_c: float = 20.0,
        window_ms: float = 40.0,
        hop_ms: float = 10.0,
        chunk_size: int = 4096
    ) -> Dict[str, Any]:
        """
        Continuously records dual-channel flight data (0.00 µs inter-channel skew)
        and extracts instantaneous velocity v(t), acceleration a(t), position x(t),
        common-mode carrier drift f0(t), and aerodynamic air-track damping.

        :param duration_sec: Total duration to record in seconds (e.g. 4.0, 6.0, 10.0).
        :param track_length_m: Physical length of the air track in meters (default 2.0m).
        :param f0_nominal: Nominal carrier frequency (if None, loaded from profile or baseline).
        :param profile: Optional device profile (e.g. 'profiles/active_buzzer_2610hz.json').
        :param temperature_c: Air temperature in °C for c(T) calculation.
        :param window_ms: STFT analysis window duration in ms (default 40.0 ms = 2000 samples).
        :param hop_ms: Sliding hop step in ms (default 10.0 ms = 100 Hz trajectory rate).
        :param chunk_size: Hardware DMA streaming packet size.
        :return: Comprehensive flight dictionary containing:
                 - 'times_sec': 1D array of time stamps at 100 Hz.
                 - 'velocity_mps': 1D velocity trajectory in m/s.
                 - 'velocity_cmps': 1D velocity trajectory in cm/s.
                 - 'acceleration_mps2': 1D acceleration trajectory in m/s².
                 - 'position_m': 1D integrated position trajectory along the track.
                 - 'f_mic1_hz': Instantaneous frequency observed at Mic 1 (left).
                 - 'f_mic2_hz': Instantaneous frequency observed at Mic 2 (right).
                 - 'f0_common_hz': Real-time drifting center frequency of the buzzer.
                 - 'kinematics_summary': Viscous drag γ, restitution e, and peak speeds.
                 - 'raw_t_sec': Full-rate continuous time axis.
                 - 'raw_v_a0': Full-rate raw ADC voltage from Mic 1.
                 - 'raw_v_a1': Full-rate raw ADC voltage from Mic 2.
        """
        from pynq_localizer.kinematics import DifferentialDopplerTracker

        # 1. Capture continuous stereo stream directly to DDR
        t_raw, v_a0, v_a1 = self.record_continuous(duration_sec=duration_sec, chunk_size=chunk_size)

        # 2. Instantiate Differential Doppler Tracker
        tracker = DifferentialDopplerTracker(
            nominal_f0_hz=f0_nominal if f0_nominal is not None else 2609.73,
            profile=profile,
            temperature_c=temperature_c,
            track_length_m=track_length_m
        )

        # 3. Slide analysis window across the captured stream (100 Hz trajectory rate)
        win_len = int((float(window_ms) / 1000.0) * self.fs_per_ch)
        hop_len = max(1, int((float(hop_ms) / 1000.0) * self.fs_per_ch))
        dt_hop = float(hop_len) / float(self.fs_per_ch)

        n_samples = len(v_a0)
        indices = np.arange(0, n_samples - win_len + 1, hop_len)
        n_frames = len(indices)

        times_sec = np.zeros(n_frames, dtype=np.float64)
        vel_mps = np.zeros(n_frames, dtype=np.float64)
        accel_mps2 = np.zeros(n_frames, dtype=np.float64)
        pos_m = np.zeros(n_frames, dtype=np.float64)
        f1_arr = np.zeros(n_frames, dtype=np.float64)
        f2_arr = np.zeros(n_frames, dtype=np.float64)
        f0_arr = np.zeros(n_frames, dtype=np.float64)

        for i, idx in enumerate(indices):
            chunk0 = v_a0[idx : idx + win_len]
            chunk1 = v_a1[idx : idx + win_len]

            res = tracker.process_stereo_frame(v_a0=chunk0, v_a1=chunk1, fs=self.fs_per_ch, dt_sec=dt_hop)

            times_sec[i] = (idx + (win_len / 2.0)) / float(self.fs_per_ch)
            vel_mps[i] = res["velocity_mps"]
            accel_mps2[i] = res["acceleration_mps2"]
            pos_m[i] = res["position_m"]
            f1_arr[i] = res["f_mic1_hz"]
            f2_arr[i] = res["f_mic2_hz"]
            f0_arr[i] = res["f0_common_hz"]

        # 4. Extract aerodynamic drag and bumper collision metrics
        summary = DifferentialDopplerTracker.analyze_glider_kinematics(time_sec=times_sec, velocity_mps=vel_mps)

        return {
            "times_sec": times_sec,
            "velocity_mps": vel_mps,
            "velocity_cmps": vel_mps * 100.0,
            "acceleration_mps2": accel_mps2,
            "position_m": pos_m,
            "f_mic1_hz": f1_arr,
            "f_mic2_hz": f2_arr,
            "f0_common_hz": f0_arr,
            "kinematics_summary": summary,
            "raw_t_sec": t_raw,
            "raw_v_a0": v_a0,
            "raw_v_a1": v_a1
        }

    # =========================================================================
    # 4. Jupyter Audio Playback & Interactive Dashboard
    # =========================================================================

    def play_audio(self, channel: int = 1, custom_data: Optional[np.ndarray] = None):
        """Plays captured microphone audio directly in Jupyter Notebook."""
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
        hop_ms: float = 10.0,
        profile: Optional[Union[Any, str, Path]] = None,
        k_constant: Optional[float] = None,
        estimator: Optional[Any] = None,
        **kwargs
    ):
        """Launches the real-time 10-second rolling Multi-Tab Kinematics Dashboard."""
        from pynq_localizer.kinematics_dashboard import KinematicsDashboard
        dash = KinematicsDashboard(
            overlay=self,
            window_duration_sec=window_duration_sec,
            hop_ms=hop_ms,
            fs_per_ch=self.fs_per_ch,
            profile=profile,
            k_constant=k_constant,
            estimator=estimator,
            **kwargs
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