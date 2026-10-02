"""
pynq_localizer.array: Core Dual-Microphone Hardware Interface & Lock-Step Polar Streaming Engine.
Features zero-skew simultaneous sampling (A0 & A1), hardware anti-aliasing decimation (M=1, 10, 20, 50),
32-bit polar CORDIC phase-magnitude spectral capture, hybrid dual-DMA BFP-immune demodulation,
sub-microsecond hardware timestamping, quasi-anechoic direct-pulse gating, dual-channel metric ranging,
and continuous multi-second flight recording for Doppler kinematics.
"""

import time
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import json
import scipy.signal as signal

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
        Auto-fetches the pinned hardware bitstream if bitfile_name is None.
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
    ) -> Dict[str, Any]:
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

        return {
            "mode": mode_clean,
            "decimation_M": m_val,
            "n_fft": n_val,
            "packet_size": pkt_val,
            "sample_rate_hz": self.fs_per_ch,
            "bin_resolution_hz": self.fs_per_ch / float(n_val),
            "time_window_ms": (n_val / self.fs_per_ch) * 1000.0,
            "nyquist_bandwidth_hz": self.fs_per_ch / 2.0,
        }

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
        """
        from pynq_localizer.kinematics import KinematicAnalytics

        frame = self.capture_spectral_frame(fft_source=source, timeout=timeout)
        time_sig = frame["v_a0"] if ("A0" in source.upper() or "CH1" in source.upper()) else frame["v_a1"]

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
        """
        from pynq_localizer.kinematics import AngleOfArrivalEstimator, KinematicAnalytics

        frame = self.capture_spectral_frame(
            fft_source="A0",
            crop_startup_samples=crop_startup_samples,
            timeout=timeout
        )
        v_a0 = frame["v_a0"]
        v_a1 = frame["v_a1"]

        if f_target is not None:
            f_eval = float(f_target)
        elif profile is not None:
            f_eval = None
        else:
            f_detected, _ = KinematicAnalytics.track_sub_hertz_pitch(
                frame["freqs"], frame["mag"], min_freq_hz=100.0, max_freq_hz=15000.0, interpolate=True
            )
            f_eval = f_detected if (np.isfinite(f_detected) and f_detected > 0) else KinematicAnalytics.DEFAULT_F0_HZ

        estimator = AngleOfArrivalEstimator(
            mic_distance_m=mic_distance_m,
            target_freq_hz=f_eval,
            profile=profile,
            temperature_c=temperature_c,
            noise_gate_v=noise_gate_v
        )

        aoa_res = estimator.estimate_angle(v_a0=v_a0, v_a1=v_a1, fs=self.fs_per_ch, f_target=f_eval)

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
        calibrated_offset_m1_ms: Optional[float] = None,
        calibrated_offset_m2_ms: Optional[float] = None,
        temperature_c: float = 20.0,
        blanking_ms: float = 2.00,
        min_thresh_mv: float = 15.0,
        gate_cycles: int = 3,
        timeout: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Synchronously fires an acoustic pulse on AR2 (U13), captures the dual-channel
        response via DMA 0 at 500 kSPS (M=1), reads hardware direct-energy registers,
        and computes:
          • Calibrated ToA flight distances (r1, r2) with independent channel offsets
          • Exact near-field 2D Cartesian multilateration (x, y) with modulo-lambda parity
          • TDOA bearing (theta_tdoa_deg) and Autodyne direct phase (theta_phase_deg)
          • Normalized Energy-difference bearing (theta_energy_deg)
          • Dual-channel metric distance inversions via Amplitude (k_A/A) and Energy (sqrt(k_E/E))
          • Pre-trigger RMS noise floors and dynamic Signal-to-Noise Ratios (SNR)
        """
        from pynq_localizer.kinematics import KinematicAnalytics, AcousticProfile

        c_sound = KinematicAnalytics.speed_of_sound(temperature_c)

        # 1. Resolve Profile Parameters (Carrier f0, Dual Offsets, Dual k_A, Dual k_E)
        f0 = KinematicAnalytics.DEFAULT_F0_HZ
        p_off1 = KinematicAnalytics.DEFAULT_TOA_OFFSET_MS
        p_off2 = KinematicAnalytics.DEFAULT_TOA_OFFSET_MS
        ka1, ka2 = 0.0196, 0.0196
        ke1, ke2 = None, None

        if profile is not None:
            if isinstance(profile, (str, Path)):
                p_prof = AcousticProfile.from_json(profile)
            elif isinstance(profile, AcousticProfile):
                p_prof = profile
            else:
                p_prof = None

            if p_prof is not None:
                f0 = float(p_prof.frequencies[0])
                p_off1 = p_prof.calibrated_toa_offset_m1_ms
                p_off2 = p_prof.calibrated_toa_offset_m2_ms
                ka1 = float(p_prof.k_values[0])
                ka2 = float(p_prof.k_values_m2[0])
                ke1 = float(p_prof.k_energy_m1[0])
                ke2 = float(p_prof.k_energy_m2[0])

        if f_target is not None:
            f0 = float(f_target)

        # Priority resolution for independent offsets: explicit override -> profile
        off1_ms = float(calibrated_offset_m1_ms) if calibrated_offset_m1_ms is not None else (
            float(calibrated_offset_ms) if calibrated_offset_ms is not None else p_off1
        )
        off2_ms = float(calibrated_offset_m2_ms) if calibrated_offset_m2_ms is not None else (
            float(calibrated_offset_ms) if calibrated_offset_ms is not None else p_off2
        )

        adc_scale = 3.3 / 4095.0
        if ke1 is None: ke1 = (ka1 / adc_scale) ** 2
        if ke2 is None: ke2 = (ka2 / adc_scale) ** 2

        # 2. Ensure DMA 0 is running and allocate M=1 undecimated buffer
        if hasattr(self, "axi_dma_0"):
            if not self.axi_dma_0.recvchannel.running:
                self.axi_dma_0.mmio.write(0x30, 0x04)
                time.sleep(0.005)
                self.axi_dma_0.recvchannel.start()

        if self._buf_time is None or len(self._buf_time) != packet_samples:
            if self._buf_time is not None:
                try: self._buf_time.close()
                except Exception: pass
            self._buf_time = allocate(shape=(packet_samples,), dtype="u2")

        # 3. Configure Hardware Trigger Unit
        fs = 500_000.0
        n_gate_target = int(gate_cycles * round(fs / f0))

        if hasattr(self, "trigger") and self.trigger is not None:
            self.trigger.disarm()
            self.trigger.set_decimation(1)  # M=1 Bypass Mode (500 kSPS)
            self.trigger.set_packet_size(packet_samples)
            self.trigger.set_threshold(3.29)
            self.trigger.set_pulse_width_ms(pulse_width_ms)
            if hasattr(self.trigger, "set_direct_gate_samples"):
                self.trigger.set_direct_gate_samples(n_gate_target)

        # 4. Queue DMA 0 and Fire Pulse with Self-Clearing Strobe
        self.axi_dma_0.recvchannel.transfer(self._buf_time)
        time.sleep(0.002)

        # Strobe sequence: Arm Single (0x09) -> Strobe Bit 7 (0x89) -> Restore (0x09)
        self.trigger.mmio.write(0x00, 0x00000009)
        self.trigger.mmio.write(0x00, 0x00000089)
        self.trigger.mmio.write(0x00, 0x00000009)

        # 5. Await DMA Transfer Completion
        t0 = time.time()
        while not self.axi_dma_0.recvchannel.idle:
            if time.time() - t0 > timeout:
                self.trigger.disarm()
                raise TimeoutError(f"Pulsed ToA DMA transfer timed out after {timeout}s.")
            time.sleep(0.0005)

        # Read hardware direct-energy registers directly from PL
        hw_e1, hw_e2 = 0, 0
        if hasattr(self, "trigger") and self.trigger is not None and hasattr(self.trigger, "get_direct_energy_counts"):
            hw_e1, hw_e2 = self.trigger.get_direct_energy_counts()

        # 6. Unpack De-interleaved Samples (DMA 0)
        raw = np.array(self._buf_time)
        v_a0 = (raw[0::2] >> 4) * adc_scale  # Mic 1 (A0, Vaux1)
        v_a1 = (raw[1::2] >> 4) * adc_scale  # Mic 2 (A1, Vaux9)
        t_ms = (np.arange(len(v_a0)) / fs) * 1000.0

        # Pre-trigger baseline noise floor extraction (first 500 samples before pulse arrival)
        noise_samples = min(500, len(v_a0) // 8)
        noise1_v = float(np.std(v_a0[:noise_samples]))
        noise2_v = float(np.std(v_a1[:noise_samples]))
        noise1_mv = noise1_v * 1000.0
        noise2_mv = noise2_v * 1000.0

        # 7. Bandpass Filter around Carrier (f0 ± 350 Hz)
        nyq = fs / 2.0
        b, a = signal.butter(2, [max(20.0, f0 - 350.0) / nyq, min(nyq - 20.0, f0 + 350.0) / nyq], btype="bandpass")
        v1_bp = signal.filtfilt(b, a, v_a0 - np.mean(v_a0)) * 1000.0  # in mV
        v2_bp = signal.filtfilt(b, a, v_a1 - np.mean(v_a1)) * 1000.0  # in mV

        env1 = np.abs(KinematicAnalytics.hilbert_transform(v1_bp))
        env2 = np.abs(KinematicAnalytics.hilbert_transform(v2_bp))

        # 8. Detect Acoustic Wavefronts (Sub-Sample Linear Zero-Crossing Interpolation)
        def find_wavefront(v_bp, env):
            blank_idx = int((blanking_ms / 1000.0) * fs)
            if len(env) <= blank_idx:
                return np.nan, 0
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
                    return (float(z0) + frac) / fs * 1000.0, int(z0)
            return np.nan, 0

        t1_raw_ms, idx1_wf = find_wavefront(v1_bp, env1)
        t2_raw_ms, idx2_wf = find_wavefront(v2_bp, env2)

        # 9. Compute Independent Flight Times & Radial Distances
        t1_flight_ms = max(0.0, t1_raw_ms - off1_ms) if np.isfinite(t1_raw_ms) else np.nan
        t2_flight_ms = max(0.0, t2_raw_ms - off2_ms) if np.isfinite(t2_raw_ms) else np.nan

        r1_m = (c_sound * (t1_flight_ms / 1000.0)) if np.isfinite(t1_flight_ms) else np.nan
        r2_m = (c_sound * (t2_flight_ms / 1000.0)) if np.isfinite(t2_flight_ms) else np.nan

        delta_t_ms = (t1_raw_ms - t2_raw_ms) if (np.isfinite(t1_raw_ms) and np.isfinite(t2_raw_ms)) else np.nan
        raw_delta_r_m = (c_sound * (delta_t_ms / 1000.0)) if np.isfinite(delta_t_ms) else np.nan

        # O(1) Non-Iterative Modulo-Lambda Symmetric Wrap
        lambda_m = c_sound / f0
        if np.isfinite(raw_delta_r_m):
            delta_r_corr_m = (raw_delta_r_m + (lambda_m / 2.0)) % lambda_m - (lambda_m / 2.0)
        else:
            delta_r_corr_m = np.nan

        delta_r_cm = delta_r_corr_m * 100.0 if np.isfinite(delta_r_corr_m) else np.nan

        # Solve Exact 2D Near-Field Position (x, y)
        loc2d = KinematicAnalytics.solve_2d_multilateration(
            r1_m=r1_m if np.isfinite(r1_m) else 0.0,
            r2_m=r2_m if np.isfinite(r2_m) else 0.0,
            d_m=mic_distance_m,
            f0=f0,
            c_sound=c_sound,
            wrap_modulo_lambda=True,
            delta_r_m=delta_r_corr_m
        )

        r1_cm = r1_m * 100.0 if np.isfinite(r1_m) else np.nan
        r2_cm = r2_m * 100.0 if np.isfinite(r2_m) else np.nan
        range_cm = loc2d["range_m"] * 100.0 if np.isfinite(loc2d["range_m"]) else np.nan
        x_cm = loc2d["x_m"] * 100.0 if np.isfinite(loc2d["x_m"]) else np.nan
        y_cm = loc2d["y_m"] * 100.0 if np.isfinite(loc2d["y_m"]) else np.nan

        # TDOA Bearing Angle
        if np.isfinite(delta_r_corr_m):
            sin_tdoa = np.clip(delta_r_corr_m / mic_distance_m, -1.0, 1.0)
            theta_tdoa_deg = float(np.degrees(np.arcsin(sin_tdoa)))
        else:
            theta_tdoa_deg = np.nan

        # 10. Direct-Path Wavefront Extractions & 3-Method Solvers
        if np.isfinite(t1_raw_ms) and np.isfinite(t2_raw_ms):
            # Direct energy & amplitude: Each microphone integrates starting at its OWN wavefront arrival
            gated0 = KinematicAnalytics.extract_gated_direct_fourier(
                signal_v=v_a0, fs=fs, f0=f0, n_cycles=gate_cycles, start_idx=idx1_wf
            )
            gated1 = KinematicAnalytics.extract_gated_direct_fourier(
                signal_v=v_a1, fs=fs, f0=f0, n_cycles=gate_cycles, start_idx=idx2_wf
            )

            amp_direct_a0 = gated0["amplitude_v"]
            amp_direct_a1 = gated1["amplitude_v"]
            e_direct_a0 = gated0["energy_v2"]
            e_direct_a1 = gated1["energy_v2"]

            # Direct Phase Interferometry: Common synchronized window starting at max(idx1_wf, idx2_wf)
            sync_start = max(idx1_wf, idx2_wf)
            sync_end = min(len(v_a0), sync_start + n_gate_target)
            phase_direct = KinematicAnalytics.extract_dual_coherent_phase(
                signal_a0_v=v_a0[sync_start:sync_end],
                signal_a1_v=v_a1[sync_start:sync_end],
                fs=fs,
                target_freq_hz=f0,
                remove_dc=True
            )
            dphi_direct_rad = phase_direct["delta_phi_rad"]
            sin_phase = np.clip((c_sound * dphi_direct_rad) / (2.0 * np.pi * f0 * mic_distance_m), -1.0, 1.0)
            theta_phase_deg = float(np.degrees(np.arcsin(sin_phase)))
            coherence_val = phase_direct["coherence"]

            # Signal-to-Multipath Ratio (SMR)
            tail_s0 = idx1_wf + n_gate_target
            tail_e0 = min(len(v_a0), tail_s0 + int(0.0015 * fs))
            e_tail0 = float(np.sum((v_a0[tail_s0:tail_e0] - np.mean(v_a0[tail_s0:tail_e0])) ** 2)) if tail_s0 < len(v_a0) else 1e-6
            smr_a0_db = float(10.0 * np.log10(max(e_direct_a0, 1e-9) / max(e_tail0, 1e-9)))

            tail_s1 = idx2_wf + n_gate_target
            tail_e1 = min(len(v_a1), tail_s1 + int(0.0015 * fs))
            e_tail1 = float(np.sum((v_a1[tail_s1:tail_e1] - np.mean(v_a1[tail_s1:tail_e1])) ** 2)) if tail_s1 < len(v_a1) else 1e-6
            smr_a1_db = float(10.0 * np.log10(max(e_direct_a1, 1e-9) / max(e_tail1, 1e-9)))

            # Dual-Channel Metric Inversions: Amplitude (kA / A) and Energy (sqrt(kE / E))
            r_amp_m1_cm = (ka1 / amp_direct_a0 * 100.0) if amp_direct_a0 > 0.001 else np.nan
            r_amp_m2_cm = (ka2 / amp_direct_a1 * 100.0) if amp_direct_a1 > 0.001 else np.nan

            r_energy_m1_cm = (np.sqrt(ke1 / max(e_direct_a0, 1e-9)) * 100.0) if e_direct_a0 > 1e-6 else np.nan
            r_energy_m2_cm = (np.sqrt(ke2 / max(e_direct_a1, 1e-9)) * 100.0) if e_direct_a1 > 1e-6 else np.nan

            # Gain-Normalized Energy Monopulse Bearing
            e1_norm = e_direct_a0 / ke1 if (ke1 is not None and ke1 > 0) else e_direct_a0
            e2_norm = e_direct_a1 / ke2 if (ke2 is not None and ke2 > 0) else e_direct_a1

            e_bearing = KinematicAnalytics.calculate_energy_bearing(
                energy_mic1=e1_norm, energy_mic2=e2_norm,
                range_m=loc2d["range_m"], mic_distance_m=mic_distance_m
            )
            theta_energy_deg = e_bearing["theta_energy_deg"]

            # Dynamic SNR
            snr1_db = float(20.0 * np.log10(max(amp_direct_a0, 1e-6) / max(noise1_v, 1e-6)))
            snr2_db = float(20.0 * np.log10(max(amp_direct_a1, 1e-6) / max(noise2_v, 1e-6)))
        else:
            amp_direct_a0, amp_direct_a1 = 0.0, 0.0
            e_direct_a0, e_direct_a1 = 0.0, 0.0
            smr_a0_db, smr_a1_db = 0.0, 0.0
            r_amp_m1_cm, r_amp_m2_cm = np.nan, np.nan
            r_energy_m1_cm, r_energy_m2_cm = np.nan, np.nan
            theta_phase_deg = np.nan
            dphi_direct_rad = np.nan
            theta_energy_deg = np.nan
            coherence_val = 0.0
            snr1_db, snr2_db = 0.0, 0.0

        is_valid = np.isfinite(r1_cm) and np.isfinite(r2_cm) and (loc2d["status"] != "SILENCE")

        return {
            # Radial & Multilateration Distances
            "r1_cm": r1_cm,
            "r2_cm": r2_cm,
            "distance_cm": range_cm,
            "distance_m": loc2d["range_m"],
            "delta_r_cm": delta_r_cm,
            "delta_t_ms": delta_t_ms,
            "x_cm": x_cm,
            "y_cm": y_cm,
            # Bearing Angles (TDOA, Phase, Energy)
            "theta_deg": loc2d["theta_deg"],
            "theta_tdoa_deg": theta_tdoa_deg,
            "theta_phase_deg": theta_phase_deg,
            "theta_energy_deg": theta_energy_deg,
            "delta_phi_rad": dphi_direct_rad,
            # Dual Inverted Metric Distances
            "distance_energy_cm": r_energy_m1_cm,  # Backward-compatible alias (Mic 1)
            "dist_amp_m1_cm": r_amp_m1_cm,
            "dist_amp_m2_cm": r_amp_m2_cm,
            "dist_energy_m1_cm": r_energy_m1_cm,
            "dist_energy_m2_cm": r_energy_m2_cm,
            # Direct Amplitudes & Energies
            "amp_direct_a0_v": amp_direct_a0,
            "amp_direct_a1_v": amp_direct_a1,
            "energy_direct_a0": e_direct_a0,
            "energy_direct_a1": e_direct_a1,
            "hw_energy_mic1": hw_e1,
            "hw_energy_mic2": hw_e2,
            # Noise Floors & Signal Quality
            "noise_mic1_mv": noise1_mv,
            "noise_mic2_mv": noise2_mv,
            "snr_mic1_db": snr1_db,
            "snr_mic2_db": snr2_db,
            "smr_a0_db": smr_a0_db,
            "smr_a1_db": smr_a1_db,
            # Raw Timestamps & Signal Envelopes
            "t1_raw_ms": t1_raw_ms,
            "t2_raw_ms": t2_raw_ms,
            "t_flight_sec": t1_flight_ms / 1000.0 if np.isfinite(t1_flight_ms) else np.nan,
            "amp_a0_v": float(np.max(env1)) / 1000.0,
            "amp_a1_v": float(np.max(env2)) / 1000.0,
            "status": loc2d["status"] if is_valid else "SILENCE",
            "v_a0": v_a0,
            "v_a1": v_a1,
            "v1_bp": v1_bp,
            "v2_bp": v2_bp,
            "env1": env1,
            "env2": env2,
            "t_ms": t_ms,
        }
    
    # =========================================================================
    # 3. Continuous Multi-Second Flight Recorder & Velocity Kinematics
    # =========================================================================

    def record_continuous(
        self,
        duration_sec: float = 3.0,
        chunk_size: int = 4096
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Continuously streams and records uninterrupted multi-second flight data from both microphones.
        Features zero-skew simultaneous sampling (A0 and A1) and hardware anti-aliasing decimation.

        :param duration_sec: Total duration to record in seconds (e.g. 3.0, 5.0, 6.0).
        :param chunk_size: Size of individual interleaved DMA streaming packets (must be an integer 
                           multiple of 4096 to preserve 2048-point lock-step FFT alignment, default: 4096).
        :return: (time_axis_sec, v_mic1_a0, v_mic2_a1) arrays in physical Volts.
        """
        self._init_xadc_simultaneous()

        # 1. Hardware Decimation Guard: Enforce M=10 (50 kSPS per channel)
        # Prevents previous M=1 bypass experiments (ToA) from leaving a stale decimation rate in hardware
        if hasattr(self, "trigger") and self.trigger is not None:
            self.trigger.set_decimation(10)
            self.fs_per_ch = 50_000.0

        # 2. Hardware Alignment Guard: Enforce packet size to integer multiples of 2 * N_FFT (4096)
        # The demux splits 2:1, so 4096 interleaved samples delivers exactly 2048 samples to xfft_0
        n_min_pkt = self.n_fft * 2  # 4096 for N=2048
        if chunk_size < n_min_pkt or (chunk_size % n_min_pkt != 0):
            safe_chunk = max(n_min_pkt, int(round(chunk_size / float(n_min_pkt))) * n_min_pkt)
        else:
            safe_chunk = int(chunk_size)

        self.trigger.set_packet_size(safe_chunk)
        self.trigger.set_mode("Auto")

        total_samples_per_ch = int(float(duration_sec) * self.fs_per_ch)
        total_interleaved_samples = total_samples_per_ch * 2
        num_chunks = int(np.ceil(total_interleaved_samples / float(safe_chunk)))

        raw_interleaved = np.empty(num_chunks * safe_chunk, dtype=np.uint16)
        chunk_buf = allocate(shape=(safe_chunk,), dtype="u2")
        dummy_fft_buf = allocate(shape=(safe_chunk // 2,), dtype="u4")

        # 3. DMA S2MM Reset & Startup Flush
        self.axi_dma_0.mmio.write(0x30, 0x04)
        self.axi_dma_1.mmio.write(0x30, 0x04)
        time.sleep(0.005)
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
                        # Clean up DMA channels before raising to prevent hardware hang
                        self.axi_dma_0.mmio.write(0x30, 0x04)
                        self.axi_dma_1.mmio.write(0x30, 0x04)
                        raise TimeoutError(
                            "Continuous DMA streaming timed out. Hardware stalled. "
                            f"(chunk_size={safe_chunk}, n_fft={self.n_fft})"
                        )
                    time.sleep(0.001)

                raw_interleaved[write_ptr : write_ptr + safe_chunk] = np.array(chunk_buf)
                write_ptr += safe_chunk

            valid_samples = raw_interleaved[:total_interleaved_samples]
            raw_a0 = valid_samples[0::2]
            raw_a1 = valid_samples[1::2]

            # 4. Unpack 12-bit ADC raw integer codes to calibrated physical Volts
            v_a0 = (raw_a0 >> 4) * (3.3 / 4095.0)
            v_a1 = (raw_a1 >> 4) * (3.3 / 4095.0)

            t_axis = np.linspace(0.0, duration_sec, len(v_a0), endpoint=False)
            print(f"[FlightRecorder] Captured {len(v_a0)} stereo samples successfully with 0.00 µs skew.")
            return t_axis, v_a0, v_a1

        finally:
            if chunk_buf is not None:
                try: chunk_buf.close()
                except Exception: pass
            if dummy_fft_buf is not None:
                try: dummy_fft_buf.close()
                except Exception: pass
            if hasattr(self, "trigger") and self.trigger is not None:
                self.trigger.set_packet_size(self.packet_size)
                
    def record_differential_flight(
        self,
        duration_sec: float = 4.0,
        track_length_m: float = 1.0,
        f0_nominal: Optional[float] = None,
        profile: Optional[Union[Any, str, Path]] = None,
        temperature_c: float = 20.0,
        window_ms: float = 40.0,
        hop_ms: float = 10.0,
        chunk_size: int = 4096
    ) -> Dict[str, Any]:
        """
        Continuously records dual-channel flight data (0.00 µs inter-channel skew)
        and extracts instantaneous velocities:
          • Differential velocity: v_diff(t) = c · (f2 - f1) / (f1 + f2)  [Common-mode drift canceled]
          • Single-mic velocity Left:  v_mic1(t) = -c · (f1 - f0) / f0
          • Single-mic velocity Right: v_mic2(t) = +c · (f2 - f0) / f0
          • Mean single-mic velocity:  v_mean_single(t) = 0.5 * (v_mic1 + v_mic2)
          • Instantaneous acceleration a(t), position x(t), dynamic in-band amplitudes (A1, A2),
            common-mode carrier drift f0(t), aerodynamic drag γ, and bumper restitution e.
        """
        from pynq_localizer.kinematics import DifferentialDopplerTracker, KinematicAnalytics

        # 1. Capture continuous stereo stream directly to DDR
        t_raw, v_a0, v_a1 = self.record_continuous(duration_sec=duration_sec, chunk_size=chunk_size)

        # 2. Resolve nominal carrier frequency
        if f0_nominal is not None:
            f_ref = float(f0_nominal)
        elif profile is not None:
            f_ref = None  # Handled by DifferentialDopplerTracker constructor
        else:
            f_ref = KinematicAnalytics.DEFAULT_F0_HZ

        # 3. Instantiate Differential Doppler Tracker
        tracker = DifferentialDopplerTracker(
            nominal_f0_hz=f_ref if f_ref is not None else KinematicAnalytics.DEFAULT_F0_HZ,
            profile=profile,
            temperature_c=temperature_c,
            track_length_m=track_length_m
        )

        # 4. Slide analysis window across the captured stream (100 Hz trajectory rate)
        win_len = int((float(window_ms) / 1000.0) * self.fs_per_ch)
        hop_len = max(1, int((float(hop_ms) / 1000.0) * self.fs_per_ch))
        dt_hop = float(hop_len) / float(self.fs_per_ch)

        n_samples = len(v_a0)
        indices = np.arange(0, n_samples - win_len + 1, hop_len)
        n_frames = len(indices)

        times_sec = np.zeros(n_frames, dtype=np.float64)
        vel_mps = np.zeros(n_frames, dtype=np.float64)
        v_m1_mps = np.zeros(n_frames, dtype=np.float64)
        v_m2_mps = np.zeros(n_frames, dtype=np.float64)
        v_mean_s_mps = np.zeros(n_frames, dtype=np.float64)
        accel_mps2 = np.zeros(n_frames, dtype=np.float64)
        pos_m = np.zeros(n_frames, dtype=np.float64)
        f1_arr = np.zeros(n_frames, dtype=np.float64)
        f2_arr = np.zeros(n_frames, dtype=np.float64)
        f0_arr = np.zeros(n_frames, dtype=np.float64)
        amp1_mv = np.zeros(n_frames, dtype=np.float64)
        amp2_mv = np.zeros(n_frames, dtype=np.float64)

        for i, idx in enumerate(indices):
            chunk0 = v_a0[idx : idx + win_len]
            chunk1 = v_a1[idx : idx + win_len]

            res = tracker.process_stereo_frame(v_a0=chunk0, v_a1=chunk1, fs=self.fs_per_ch, dt_sec=dt_hop)

            times_sec[i] = (idx + (win_len / 2.0)) / float(self.fs_per_ch)
            vel_mps[i] = res["velocity_mps"]
            v_m1_mps[i] = res["v_mic1_mps"]
            v_m2_mps[i] = res["v_mic2_mps"]
            v_mean_s_mps[i] = res["v_mean_single_mps"]
            accel_mps2[i] = res["acceleration_mps2"]
            pos_m[i] = res["position_m"]
            f1_arr[i] = res["f_mic1_hz"]
            f2_arr[i] = res["f_mic2_hz"]
            f0_arr[i] = res["f0_common_hz"]
            amp1_mv[i] = res["amp_mic1_v"] * 1000.0
            amp2_mv[i] = res["amp_mic2_v"] * 1000.0

        # 5. Extract aerodynamic drag and collision metrics
        summary = DifferentialDopplerTracker.analyze_glider_kinematics(time_sec=times_sec, velocity_mps=vel_mps)

        return {
            "times_sec": times_sec,
            # Differential Velocities
            "velocity_mps": vel_mps,
            "velocity_cmps": vel_mps * 100.0,
            # Single-Microphone Velocities
            "v_mic1_mps": v_m1_mps,
            "v_mic1_cmps": v_m1_mps * 100.0,
            "v_mic2_mps": v_m2_mps,
            "v_mic2_cmps": v_m2_mps * 100.0,
            "v_mean_single_mps": v_mean_s_mps,
            "v_mean_single_cmps": v_mean_s_mps * 100.0,
            # Kinematics & Spectral Trajectories
            "acceleration_mps2": accel_mps2,
            "position_m": pos_m,
            "f_mic1_hz": f1_arr,
            "f_mic2_hz": f2_arr,
            "f0_common_hz": f0_arr,
            "amp_mic1_mv": amp1_mv,
            "amp_mic2_mv": amp2_mv,
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
        """Launches the real-time rolling Multi-Tab Kinematics Dashboard."""
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
    # 5. Cleanup & Context Management
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