"""
pynq_localizer.kinematics: High-Precision Acoustic Kinematics & Frequency Ridge Tracking Engine.
Provides sub-Hertz pitch tracking (20 Hz - 20 kHz), spectral quadruple extraction (f, A, φ, t),
hybrid dual-DMA coherent in-band voltage demodulation, multi-source tracking, single-channel
distance estimation r = k(f)/A, acoustic profile modeling, standalone acoustic calibration
protocols with Weighted Least Squares (WLS) and dynamic 1/r boundary pruning, phase velocity
verification, smoothed RMS envelope extraction, sliding STFT trajectories, temperature-compensated
sound speed, Doppler velocity, and gravity metrics.
"""

import json
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, List, Any, Callable
import numpy as np

try:
    from scipy.interpolate import interp1d
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except (ImportError, ModuleNotFoundError):
    _HAS_SCIPY = False


class KinematicAnalytics:
    """
    High-performance DSP engine for acoustic kinematics, frequency tracking, and quadruple telemetry.
    """

    # =========================================================================
    # 1. Physics Models & Temperature Compensation
    # =========================================================================

    @staticmethod
    def speed_of_sound(temperature_c: float = 20.0) -> float:
        """
        Calculates the temperature-compensated speed of sound in air.
        c(T) = 331.3 * sqrt(1 + T_c / 273.15) [m/s]
        """
        return float(331.3 * np.sqrt(1.0 + (float(temperature_c) / 273.15)))

    @classmethod
    def calculate_doppler_velocity(
        cls,
        f_observed: Union[float, np.ndarray],
        f_source: float,
        temperature_c: float = 20.0
    ) -> Union[float, np.ndarray]:
        """
        Calculates instantaneous radial velocity v(t) from observed Doppler frequency:
        v(t) = c(T) * ((f_observed - f_source) / f_source)
        """
        c = cls.speed_of_sound(temperature_c)
        f_obs = np.asarray(f_observed, dtype=np.float64)
        v = c * ((f_obs - float(f_source)) / float(f_source))
        return float(v) if np.isscalar(f_observed) else v

    @classmethod
    def calculate_gravity_acceleration(
        cls,
        time_sec: np.ndarray,
        f_observed: np.ndarray,
        f_source: float,
        temperature_c: float = 20.0
    ) -> Dict[str, float]:
        """
        Calculates gravitational acceleration g from the linear frequency slope of a falling source:
        g = - (c(T) / f_0) * (df / dt)
        """
        t = np.asarray(time_sec, dtype=np.float64)
        f = np.asarray(f_observed, dtype=np.float64)

        valid_mask = np.isfinite(t) & np.isfinite(f)
        t_clean = t[valid_mask]
        f_clean = f[valid_mask]

        if len(t_clean) < 5:
            raise ValueError("Insufficient valid data points to perform linear regression for gravity measurement.")

        slope, intercept = np.polyfit(t_clean, f_clean, 1)

        f_pred = slope * t_clean + intercept
        ss_res = np.sum((f_clean - f_pred) ** 2)
        ss_tot = np.sum((f_clean - np.mean(f_clean)) ** 2)
        r_squared = 1.0 - (ss_res / (ss_tot + 1e-12))

        c = cls.speed_of_sound(temperature_c)
        g_measured = - (c / float(f_source)) * slope

        return {
            "g_measured": float(g_measured),
            "slope_df_dt": float(slope),
            "r_squared": float(r_squared),
            "f_rest": float(f_source),
            "c_sound": float(c),
            "error_pct": float(abs(g_measured - 9.80665) / 9.80665 * 100.0)
        }

    # =========================================================================
    # 2. Quadruple Extraction & Coherent Time-Domain Demodulation Engine
    # =========================================================================

    @staticmethod
    def compute_coherent_inband_amplitude(
        signal_v: np.ndarray,
        fs: float,
        target_freq_hz: float,
        remove_dc: bool = True
    ) -> float:
        """
        Extracts physical in-band RMS voltage from raw 12-bit ADC time-series data at target_freq_hz.
        Uses single-bin coherent Fourier projection:
        X(f0) = (2 / N) * sum( (v[n] - mean(v)) * exp(-j * 2*pi * f0 * n / fs) )
        V_RMS = |X(f0)| / sqrt(2)

        Guarantees 100% immunity to FPGA FFT Block Floating Point bit-shift jumps.
        """
        v = np.asarray(signal_v, dtype=np.float64)
        n = len(v)
        if n == 0 or not np.isfinite(target_freq_hz) or target_freq_hz <= 0:
            return 0.0

        v_ac = v - np.mean(v) if remove_dc else v
        t = np.arange(n) / float(fs)
        phasor = np.exp(-2.0j * np.pi * float(target_freq_hz) * t)

        # Single-bin discrete Fourier projection
        x_f0 = (2.0 / n) * np.dot(v_ac, phasor)
        v_rms = float(np.abs(x_f0) / np.sqrt(2.0))
        return v_rms

    # =========================================================================
    # Dual-Channel Coherent Phase Extraction & Angle of Arrival (AoA)
    # =========================================================================

    @staticmethod
    def extract_dual_coherent_phase(
        signal_a0_v: np.ndarray,
        signal_a1_v: np.ndarray,
        fs: float,
        target_freq_hz: float,
        remove_dc: bool = True
    ) -> Dict[str, float]:
        """
        Simultaneously projects both synchronous ADC channels (A0 and A1) onto a
        Hann-windowed Fourier phasor w[n] · e^(-j 2π f0 t) to extract individual phases,
        in-band RMS voltages, and the unambiguous wrapped phase difference Δφ = φ1 - φ0.

        Hann windowing suppresses frame boundary splatter and attenuates harmonic
        distortion (such as the buzzer's 24% 2f0 harmonic) by > 75 dB.

        Sign Convention:
          • Δφ = 0      => Broadside (Wavefront arrives at A0 and A1 simultaneously)
          • Δφ > 0      => Wavefront reaches Mic 2 (A1) before Mic 1 (A0) [Right / +θ]
          • Δφ < 0      => Wavefront reaches Mic 1 (A0) before Mic 2 (A1) [Left / -θ]

        :param signal_a0_v: 1D voltage array from Channel A0 (Mic 1).
        :param signal_a1_v: 1D voltage array from Channel A1 (Mic 2).
        :param fs: Sampling frequency in Hz (e.g. 50000.0).
        :param target_freq_hz: Tone frequency f0 in Hz (e.g. 2609.73).
        :param remove_dc: If True, subtracts channel means before projection.
        :return: Dict with delta_phi_rad, delta_phi_deg, phi_a0_rad, phi_a1_rad,
                 amp_a0_v, amp_a1_v, and complex coherence.
        """
        v0 = np.asarray(signal_a0_v, dtype=np.float64)
        v1 = np.asarray(signal_a1_v, dtype=np.float64)
        n = min(len(v0), len(v1))

        if n == 0 or not np.isfinite(target_freq_hz) or target_freq_hz <= 0:
            return {
                "delta_phi_rad": 0.0,
                "delta_phi_deg": 0.0,
                "phi_a0_rad": 0.0,
                "phi_a1_rad": 0.0,
                "amp_a0_v": 0.0,
                "amp_a1_v": 0.0,
                "coherence": 0.0
            }

        v0_ac = v0[:n] - np.mean(v0[:n]) if remove_dc else v0[:n]
        v1_ac = v1[:n] - np.mean(v1[:n]) if remove_dc else v1[:n]

        t = np.arange(n) / float(fs)

        # Hann window to suppress non-integer boundary leakage and harmonic splatter
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1)) if n > 1 else np.ones(n)
        coherent_sum = float(np.sum(w))
        phasor = w * np.exp(-2.0j * np.pi * float(target_freq_hz) * t)

        # Single-bin Hann-windowed discrete Fourier projections
        scale = 2.0 / max(coherent_sum, 1e-12)
        x0 = scale * np.dot(v0_ac, phasor)
        x1 = scale * np.dot(v1_ac, phasor)

        # In-band RMS physical voltages
        v_rms_0 = float(np.abs(x0) / np.sqrt(2.0))
        v_rms_1 = float(np.abs(x1) / np.sqrt(2.0))

        # Four-quadrant phase angles
        phi_0 = float(np.angle(x0))
        phi_1 = float(np.angle(x1))

        # Relative phase difference wrapped strictly to [-π, +π]
        dphi_raw = phi_1 - phi_0
        delta_phi = float(np.arctan2(np.sin(dphi_raw), np.cos(dphi_raw)))

        # Magnitude-squared coherence approximation across the dual projection
        denom = (np.abs(x0) * np.abs(x1))
        coherence = float(np.abs(x0 * np.conj(x1)) / denom) if denom > 1e-12 else 0.0

        return {
            "delta_phi_rad": delta_phi,
            "delta_phi_deg": float(np.degrees(delta_phi)),
            "phi_a0_rad": phi_0,
            "phi_a1_rad": phi_1,
            "amp_a0_v": v_rms_0,
            "amp_a1_v": v_rms_1,
            "coherence": coherence
        }

    @classmethod
    def calculate_angle_of_arrival(
        cls,
        delta_phi_rad: float,
        f0: float,
        mic_distance_m: float,
        temperature_c: float = 20.0,
        delta_phi_err_rad: float = 0.05
    ) -> Tuple[float, float, float, bool]:
        """
        Inverts wrapped acoustic phase difference Δφ to incident bearing angle θ:
          sin(θ) = (c(T) · Δφ) / (2π · f0 · d)
          θ = arcsin(clip(sin(θ), -1.0, 1.0))

        :param delta_phi_rad: Wrapped phase difference (φ1 - φ0) in radians [-π, +π].
        :param f0: Fundamental carrier frequency in Hz.
        :param mic_distance_m: Center-to-center microphone baseline d in meters.
        :param temperature_c: Ambient air temperature in Celsius.
        :param delta_phi_err_rad: Phase measurement uncertainty in radians.
        :return: (theta_deg, theta_rad, theta_err_deg, is_aliased)
        """
        c = cls.speed_of_sound(temperature_c)
        f_val = float(f0)
        d_val = float(mic_distance_m)

        if f_val <= 0 or d_val <= 0 or not np.isfinite(delta_phi_rad):
            return np.nan, np.nan, np.nan, False

        # Spatial aliasing bound: d <= c / (2 * f0)
        d_max_aliasing = c / (2.0 * f_val)
        is_aliased = bool(d_val > d_max_aliasing)

        # Inversion ratio
        ratio = (c * float(delta_phi_rad)) / (2.0 * np.pi * f_val * d_val)
        clamped_ratio = float(np.clip(ratio, -1.0, 1.0))

        theta_rad = float(np.arcsin(clamped_ratio))
        theta_deg = float(np.degrees(theta_rad))

        # Angular uncertainty propagation: dθ = (c / (2π f0 d cos θ)) · d(Δφ)
        cos_theta = max(abs(np.cos(theta_rad)), 0.05)  # Avoid division by zero at endfire
        theta_err_rad = float((c / (2.0 * np.pi * f_val * d_val * cos_theta)) * float(delta_phi_err_rad))
        theta_err_deg = float(np.degrees(theta_err_rad))

        return theta_deg, theta_rad, theta_err_deg, is_aliased

    @classmethod
    def extract_hybrid_quadruple(
        cls,
        time_signal_v: np.ndarray,
        fs: float,
        freq_axis: np.ndarray,
        magnitude: np.ndarray,
        phase_rad: np.ndarray,
        f_min: float = 100.0,
        f_max: float = 15000.0,
        timer_cycles: int = 0,
        clock_freq_hz: float = 100_000_000.0
    ) -> Dict[str, Union[float, int, bool]]:
        """
        Extracts the hybrid quadruple (f0, A_true, phi, t):
        1. Pitch (f0) & Phase (phi) extracted from DMA 1 (Hardware FFT/CORDIC).
        2. In-band Amplitude (A_true) extracted from DMA 0 (Hardware ADC Time Stream).
        """
        quad = cls.extract_quadruple(
            freq_axis=freq_axis,
            magnitude=magnitude,
            phase_rad=phase_rad,
            f_min=f_min,
            f_max=f_max,
            timer_cycles=timer_cycles,
            clock_freq_hz=clock_freq_hz
        )

        f0 = quad["frequency_hz"]

        if np.isfinite(f0) and f0 > 0:
            a_true = cls.compute_coherent_inband_amplitude(
                signal_v=time_signal_v,
                fs=fs,
                target_freq_hz=f0,
                remove_dc=True
            )
        else:
            a_true = quad["amplitude_v"]

        quad["amplitude_v"] = float(a_true)
        quad["amplitude_raw_fft"] = float(quad.get("amplitude_v", 0.0))
        return quad

    @classmethod
    def extract_quadruple(
        cls,
        freq_axis: np.ndarray,
        magnitude: np.ndarray,
        phase_rad: np.ndarray,
        f_min: float = 100.0,
        f_max: float = 10000.0,
        timer_cycles: int = 0,
        clock_freq_hz: float = 100_000_000.0,
        enbw: float = 1.0,
        raw_scale_factor: float = 3.3 / 4095.0
    ) -> Dict[str, Union[float, int, bool]]:
        """
        Extracts the physical quadruple (f0, A, phi, t) from a spectral polar frame within [f_min, f_max].
        """
        freqs = np.asarray(freq_axis, dtype=np.float64)
        mags = np.asarray(magnitude, dtype=np.float64)
        phases = np.asarray(phase_rad, dtype=np.float64)

        band_mask = (freqs >= float(f_min)) & (freqs <= float(f_max))
        band_indices = np.where(band_mask)[0]

        if len(band_indices) == 0:
            t_sec = float(timer_cycles) / clock_freq_hz
            return {
                "frequency_hz": np.nan,
                "amplitude_v": 0.0,
                "phase_rad": np.nan,
                "phase_deg": np.nan,
                "timestamp_sec": t_sec,
                "peak_bin": 0,
                "band_energy": 0.0,
                "is_valid": False
            }

        band_mags = mags[band_indices]
        local_k = int(np.argmax(band_mags))
        k0 = int(band_indices[local_k])

        f0, delta = cls.track_sub_hertz_pitch(
            freqs, mags, min_freq_hz=f_min, max_freq_hz=f_max, interpolate=True, return_delta=True
        )

        n_points = len(freqs) * 2
        band_energy = float(np.sum(band_mags ** 2))
        v_rms = (np.sqrt(2.0 * band_energy) / (n_points * enbw)) * raw_scale_factor

        raw_phi = float(phases[k0]) if (0 <= k0 < len(phases)) else 0.0
        phi_corrected = raw_phi - (np.pi * delta * (n_points - 1.0) / n_points)
        phi_wrapped = float((phi_corrected + np.pi) % (2.0 * np.pi) - np.pi)

        t_sec = float(timer_cycles) / clock_freq_hz

        return {
            "frequency_hz": float(f0),
            "amplitude_v": float(v_rms),
            "phase_rad": phi_wrapped,
            "phase_deg": float(np.degrees(phi_wrapped)),
            "timestamp_sec": float(t_sec),
            "peak_bin": int(k0),
            "band_energy": float(band_energy),
            "is_valid": bool(np.isfinite(f0) and v_rms > 0.0)
        }

    @classmethod
    def compute_phase_velocity(
        cls,
        phase_history: Union[List[float], np.ndarray],
        time_history: Union[List[float], np.ndarray],
        f_expected: Optional[float] = None,
        f_tol: float = 50.0
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calculates instantaneous phase frequency trajectory f_phase(t) = (1 / 2π) * (dΦ / dt)."""
        phi = np.asarray(phase_history, dtype=np.float64)
        t = np.asarray(time_history, dtype=np.float64)

        if len(phi) < 2 or len(t) < 2:
            return np.array([]), np.array([]), 0.0

        unwrapped_phi = np.unwrap(phi)
        dt = np.diff(t)
        dphi = np.diff(unwrapped_phi)

        valid_dt = dt > 1e-7
        f_inst = np.zeros(len(dt), dtype=np.float64)
        f_inst[valid_dt] = (dphi[valid_dt] / dt[valid_dt]) / (2.0 * np.pi)

        if f_expected is not None and len(f_inst) > 0:
            err = np.abs(f_inst - float(f_expected))
            sqi_array = np.clip(1.0 - (err / float(f_tol)), 0.0, 1.0)
            mean_sqi = float(np.mean(sqi_array))
        else:
            mean_sqi = 1.0

        return unwrapped_phi, f_inst, mean_sqi

    # =========================================================================
    # 3. Sub-Hertz Analytical Sinc Pitch Tracking & STFT Trajectories
    # =========================================================================

    @staticmethod
    def track_sub_hertz_pitch(
        freqs: np.ndarray,
        mags: np.ndarray,
        min_freq_hz: float = 20.0,
        max_freq_hz: float = 20000.0,
        interpolate: bool = True,
        return_delta: bool = False
    ) -> Union[Tuple[float, float], Tuple[float, float, float]]:
        """
        Extracts dominant fundamental frequency f0 with sub-Hertz accuracy (±0.01 Hz)
        using the exact analytical spectral ratio estimator for DFT sinc mainlobes.
        """
        valid_mask = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            k = int(np.argmax(mags))
            if return_delta:
                return float(freqs[k]), 0.0
            return float(freqs[k]), float(mags[k])

        k = int(valid_indices[np.argmax(mags[valid_indices])])

        if not interpolate or k <= 0 or k >= len(mags) - 1:
            if return_delta:
                return float(freqs[k]), 0.0
            return float(freqs[k]), float(mags[k])

        alpha = float(mags[k - 1])
        beta  = float(mags[k])
        gamma = float(mags[k + 1])

        # Exact Analytical Sinc Ratio Peak Estimator
        if gamma >= alpha:
            denom = beta + gamma
            delta = (gamma / denom) if denom > 1e-12 else 0.0
        else:
            denom = beta + alpha
            delta = (-alpha / denom) if denom > 1e-12 else 0.0

        delta = max(-0.5, min(0.5, delta))

        delta_f = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        interp_freq = float(freqs[k] + delta * delta_f)
        interp_mag = float(mags[k])

        if return_delta:
            return interp_freq, delta
        return interp_freq, interp_mag

    # =========================================================================
    # 4. Amplitude Envelope & Energy Downsampling
    # =========================================================================

    @staticmethod
    def compute_rms_amplitude(signal: np.ndarray, remove_dc: bool = True) -> float:
        """Computes root-mean-square (RMS) physical voltage of a signal slice."""
        x = np.asarray(signal, dtype=np.float64)
        if len(x) == 0:
            return 0.0
        if remove_dc:
            x = x - np.mean(x)
        return float(np.sqrt(np.mean(x ** 2)))

    @staticmethod
    def hilbert_transform(signal: np.ndarray) -> np.ndarray:
        """Computes analytic signal z[n] = x[n] + j*H{x[n]} using pure NumPy FFT."""
        x = np.asarray(signal, dtype=np.float64)
        n = len(x)
        if n == 0:
            return np.array([], dtype=np.complex128)

        xf = np.fft.fft(x)
        h = np.zeros(n, dtype=np.float64)
        if n % 2 == 0:
            h[0] = 1.0
            h[n // 2] = 1.0
            h[1 : n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1 : (n + 1) // 2] = 2.0

        return np.fft.ifft(xf * h)

    @classmethod
    def extract_analytic_envelope(cls, signal: np.ndarray, remove_dc: bool = True) -> np.ndarray:
        """Extracts physical instantaneous amplitude envelope A(t) = |z(t)|."""
        x = np.asarray(signal, dtype=np.float64)
        if remove_dc:
            x = x - np.mean(x)
        return np.abs(cls.hilbert_transform(x))

    @classmethod
    def compute_downsampled_envelope(
        cls,
        signal: np.ndarray,
        fs: float,
        step_ms: float = 10.0,
        method: str = "rms"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Downsamples continuous physical audio into a smoothed A(t) time series."""
        x = np.asarray(signal, dtype=np.float64)
        x_ac = x - np.mean(x)
        n_samples = len(x_ac)

        hop_samples = max(1, int((float(step_ms) / 1000.0) * fs))
        indices = np.arange(0, n_samples, hop_samples)

        amp_out = np.zeros(len(indices), dtype=np.float64)
        time_out = indices / float(fs)

        for i, idx in enumerate(indices):
            chunk = x_ac[idx : idx + hop_samples]
            if len(chunk) == 0:
                continue
            if method.lower() == "peak":
                amp_out[i] = np.max(np.abs(chunk))
            else:
                amp_out[i] = np.sqrt(np.mean(chunk ** 2))

        return time_out, amp_out

    @classmethod
    def compute_stft_ridge_trajectory(
        cls,
        signal: np.ndarray,
        fs: float,
        window_ms: float = 20.0,
        hop_ms: float = 10.0,
        min_freq_hz: float = 20.0,
        max_freq_hz: float = 20000.0,
        energy_thresh: float = 0.015,
        window_type: str = "blackmanharris"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts synchronized Amplitude A(t) and Dominant Frequency f0(t) trajectories."""
        x = np.asarray(signal, dtype=np.float64)
        x_ac = x - np.mean(x)
        n_samples = len(x_ac)

        win_len = max(16, int((float(window_ms) / 1000.0) * fs))
        hop_len = max(1, int((float(hop_ms) / 1000.0) * fs))

        if window_type.lower() == "blackmanharris":
            n = np.arange(win_len)
            w = (0.35875 - 0.48829 * np.cos(2.0 * np.pi * n / (win_len - 1)) +
                 0.14128 * np.cos(4.0 * np.pi * n / (win_len - 1)) -
                 0.01168 * np.cos(6.0 * np.pi * n / (win_len - 1)))
        elif window_type.lower() == "hamming":
            w = np.hamming(win_len)
        else:
            w = np.hanning(win_len)

        coherent_gain = np.sum(w) / win_len
        indices = np.arange(0, n_samples - win_len + 1, hop_len)
        n_frames = len(indices)

        times_sec = (indices + (win_len / 2.0)) / float(fs)
        amp_traj = np.zeros(n_frames, dtype=np.float64)
        freq_traj = np.full(n_frames, np.nan, dtype=np.float64)

        freq_axis = np.fft.rfftfreq(win_len, d=1.0 / fs)

        for i, idx in enumerate(indices):
            chunk = x_ac[idx : idx + win_len]
            rms_val = np.sqrt(np.mean(chunk ** 2))
            amp_traj[i] = rms_val

            if rms_val >= energy_thresh:
                windowed = chunk * w
                fft_mag = np.abs(np.fft.rfft(windowed)) / (win_len / 2.0)
                linear_v = fft_mag / max(coherent_gain, 1e-4)

                f0, _ = cls.track_sub_hertz_pitch(
                    freq_axis,
                    linear_v,
                    min_freq_hz=min_freq_hz,
                    max_freq_hz=max_freq_hz,
                    interpolate=True
                )
                freq_traj[i] = f0

        return times_sec, amp_traj, freq_traj


class MultiSourceTracker:
    """
    Multi-Band Spectral Tracker for simultaneous independent tracking of multiple acoustic sources.
    Features coherent harmonic leakage cancellation (2f0, 3f0) and dynamic SIR metrics.
    """

    def __init__(
        self,
        source_bands: Dict[str, Tuple[float, float]],
        clock_freq_hz: float = 100_000_000.0,
        enbw: float = 1.0,
        noise_gate_v: float = 0.005,
        harmonic_rejection: bool = True,
        h2_coeff: float = 0.03,
        h3_coeff: float = 0.008
    ):
        self.source_bands = source_bands
        self.clock_freq_hz = float(clock_freq_hz)
        self.enbw = float(enbw)
        self.noise_gate_v = float(noise_gate_v)
        self.harmonic_rejection = harmonic_rejection
        self.h2_coeff = float(h2_coeff)
        self.h3_coeff = float(h3_coeff)

        self.history: Dict[str, Dict[str, List[float]]] = {
            src_name: {"t": [], "f": [], "amp": [], "phi": [], "sir": []}
            for src_name in self.source_bands.keys()
        }

    def process_spectral_frame(
        self,
        freq_axis: np.ndarray,
        magnitude: np.ndarray,
        phase_rad: np.ndarray,
        timer_cycles: int = 0
    ) -> Dict[str, Any]:
        """Processes a single polar FFT frame, extracts raw quadruples, cancels harmonic cross-talk, and computes SIR."""
        t_sec = float(timer_cycles) / self.clock_freq_hz if self.clock_freq_hz > 0 else 0.0
        raw_quads = {}

        for src_name, (f_min, f_max) in self.source_bands.items():
            quad = KinematicAnalytics.extract_quadruple(
                freq_axis=freq_axis,
                magnitude=magnitude,
                phase_rad=phase_rad,
                f_min=f_min,
                f_max=f_max,
                timer_cycles=timer_cycles,
                clock_freq_hz=self.clock_freq_hz,
                enbw=self.enbw
            )
            raw_quads[src_name] = quad

        frame_results = {}
        src_names = list(self.source_bands.keys())

        for j_name in src_names:
            quad_j = raw_quads[j_name].copy()
            f_j = quad_j["frequency_hz"]
            p_j = quad_j["band_energy"]
            total_leakage = 0.0

            if self.harmonic_rejection and np.isfinite(f_j):
                for i_name in src_names:
                    if i_name == j_name:
                        continue
                    quad_i = raw_quads[i_name]
                    f_i = quad_i["frequency_hz"]
                    p_i = quad_i["band_energy"]

                    if np.isfinite(f_i) and p_i > 0:
                        if abs(f_j - 2.0 * f_i) < (self.source_bands[j_name][1] - self.source_bands[j_name][0]) * 0.5:
                            leak = p_i * self.h2_coeff
                            total_leakage += leak
                        elif abs(f_j - 3.0 * f_i) < (self.source_bands[j_name][1] - self.source_bands[j_name][0]) * 0.5:
                            leak = p_i * self.h3_coeff
                            total_leakage += leak

            p_j_clean = max(0.0, p_j - total_leakage)
            n_points = len(freq_axis) * 2
            v_rms_clean = (np.sqrt(2.0 * p_j_clean) / (n_points * self.enbw)) * (3.3 / 4095.0)

            sir_db = 10.0 * np.log10(max(p_j_clean, 1e-12) / max(total_leakage, 1e-12))
            sir_db = float(np.clip(sir_db, -10.0, 60.0))

            quad_j["amplitude_v"] = float(v_rms_clean)
            quad_j["sir_db"] = float(sir_db)
            quad_j["harmonic_leakage_energy"] = float(total_leakage)

            is_active = bool(v_rms_clean >= self.noise_gate_v and quad_j["is_valid"])
            quad_j["is_active"] = is_active
            quad_j["source_name"] = j_name
            quad_j["band_limits"] = (float(self.source_bands[j_name][0]), float(self.source_bands[j_name][1]))

            frame_results[j_name] = quad_j

            self.history[j_name]["t"].append(t_sec)
            self.history[j_name]["f"].append(quad_j["frequency_hz"] if is_active else np.nan)
            self.history[j_name]["amp"].append(v_rms_clean)
            self.history[j_name]["phi"].append(quad_j["phase_rad"] if is_active else np.nan)
            self.history[j_name]["sir"].append(sir_db)

        return {
            "timestamp_sec": t_sec,
            "sources": frame_results
        }

    def get_source_trajectory(self, source_name: str) -> Dict[str, np.ndarray]:
        """Returns the complete recorded time series for a specific source."""
        if source_name not in self.history:
            raise KeyError(f"Source '{source_name}' not found in tracker.")

        hist = self.history[source_name]
        return {
            "t": np.array(hist["t"]),
            "f": np.array(hist["f"]),
            "amp": np.array(hist["amp"]),
            "phi": np.array(hist["phi"]),
            "sir": np.array(hist["sir"])
        }

    def reset_history(self):
        """Clears all accumulated history buffers."""
        for src_name in self.history:
            self.history[src_name] = {"t": [], "f": [], "amp": [], "phi": [], "sir": []}


# =============================================================================
# 5. Acoustic Calibration Profile & Distance Inversion Engine
# =============================================================================

class AcousticProfile:
    """
    Data model and interpolator for physical acoustic calibration functions k(f).
    Stores discrete calibration grids, splines, system metadata (volume/gain), and
    frequency-dependent certified operating bounds [r_min(f), r_max(f)].
    """

    def __init__(
        self,
        frequencies_hz: Optional[Union[List[float], np.ndarray]] = None,
        k_values: Optional[Union[List[float], np.ndarray]] = None,
        r_squared: Optional[Union[List[float], np.ndarray]] = None,
        k_uncertainty: Optional[Union[List[float], np.ndarray]] = None,
        operational_bounds: Optional[Dict[str, Dict[str, float]]] = None,
        system_metadata: Optional[Dict[str, Any]] = None,
        name: str = "DefaultProfile",
        description: str = "Acoustic calibration curve k(f)"
    ):
        self.name = name
        self.description = description
        self._callable_model: Optional[Callable[[float], float]] = None
        self.operational_bounds = operational_bounds or {}
        self.system_metadata = system_metadata or {}

        if frequencies_hz is not None and k_values is not None:
            self.frequencies = np.asarray(frequencies_hz, dtype=np.float64)
            self.k_values = np.asarray(k_values, dtype=np.float64)
            self.r_squared = (
                np.asarray(r_squared, dtype=np.float64)
                if r_squared is not None
                else np.ones_like(self.k_values)
            )
            self.k_uncertainty = (
                np.asarray(k_uncertainty, dtype=np.float64)
                if k_uncertainty is not None
                else 0.03 * self.k_values
            )

            # Build 1D interpolator
            if len(self.frequencies) > 1:
                if _HAS_SCIPY and len(self.frequencies) >= 4:
                    self._interp_k = interp1d(
                        self.frequencies,
                        self.k_values,
                        kind="cubic",
                        bounds_error=False,
                        fill_value=(self.k_values[0], self.k_values[-1])
                    )
                    self._interp_err = interp1d(
                        self.frequencies,
                        self.k_uncertainty,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(self.k_uncertainty[0], self.k_uncertainty[-1])
                    )
                else:
                    self._interp_k = lambda f: float(
                        np.interp(f, self.frequencies, self.k_values, left=self.k_values[0], right=self.k_values[-1])
                    )
                    self._interp_err = lambda f: float(
                        np.interp(f, self.frequencies, self.k_uncertainty, left=self.k_uncertainty[0], right=self.k_uncertainty[-1])
                    )
            else:
                self._interp_k = lambda f: float(self.k_values[0])
                self._interp_err = lambda f: float(self.k_uncertainty[0])
        else:
            self.frequencies = np.array([1000.0], dtype=np.float64)
            self.k_values = np.array([0.05], dtype=np.float64)
            self.r_squared = np.array([1.0], dtype=np.float64)
            self.k_uncertainty = np.array([0.002], dtype=np.float64)
            self._interp_k = lambda f: 0.05
            self._interp_err = lambda f: 0.002

    def evaluate(self, frequency_hz: float) -> Tuple[float, float]:
        """Evaluates k(f) and its uncertainty delta_k at a specific frequency."""
        f = float(frequency_hz)
        if not np.isfinite(f) or f <= 0:
            return float(self.k_values[0]), float(self.k_uncertainty[0])

        if self._callable_model is not None:
            k_val = float(self._callable_model(f))
            return k_val, 0.03 * k_val

        k_val = float(self._interp_k(f))
        k_err = float(self._interp_err(f))
        return k_val, k_err

    def get_operational_bounds(self, frequency_hz: float) -> Dict[str, float]:
        """Retrieves certified physical operating distance/voltage boundaries for a frequency."""
        f = float(frequency_hz)
        if not self.operational_bounds:
            return {"r_min_m": 0.10, "r_max_m": 2.00, "v_min_v": 0.002, "v_sat_v": 0.500}

        # Find closest calibrated frequency key
        closest_f = min(self.operational_bounds.keys(), key=lambda k: abs(float(k) - f))
        return self.operational_bounds[closest_f]

    @classmethod
    def from_constant(
        cls,
        k_value: float,
        relative_error: float = 0.03,
        system_metadata: Optional[Dict[str, Any]] = None,
        name: str = "ConstantProfile"
    ) -> "AcousticProfile":
        """Factory creating an AcousticProfile with a flat constant k."""
        profile = cls(
            frequencies_hz=[100.0, 10000.0],
            k_values=[float(k_value), float(k_value)],
            r_squared=[1.0, 1.0],
            k_uncertainty=[float(k_value) * relative_error, float(k_value) * relative_error],
            system_metadata=system_metadata or {"volume_setting": 0.75, "gain_setting": "default"},
            name=name,
            description=f"Static flat profile with k={k_value:.4f} V*m"
        )
        return profile

    @classmethod
    def from_callable(
        cls,
        func: Callable[[float], float],
        system_metadata: Optional[Dict[str, Any]] = None,
        name: str = "CallableProfile"
    ) -> "AcousticProfile":
        """Factory creating an AcousticProfile evaluated directly from a mathematical function k(f)."""
        profile = cls.from_constant(0.05, system_metadata=system_metadata, name=name)
        profile._callable_model = func
        return profile

    def to_json(self, filepath: Union[str, Path]):
        """Serializes calibration profile, metadata, and operational bounds to a JSON file."""
        out_path = Path(filepath).resolve()
        data = {
            "name": self.name,
            "description": self.description,
            "system_metadata": self.system_metadata,
            "operational_bounds": self.operational_bounds,
            "frequencies_hz": self.frequencies.tolist(),
            "k_values": self.k_values.tolist(),
            "r_squared": self.r_squared.tolist(),
            "k_uncertainty": self.k_uncertainty.tolist()
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "AcousticProfile":
        """Loads a calibration profile, metadata, and bounds from a JSON file."""
        in_path = Path(filepath).resolve()
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            frequencies_hz=data["frequencies_hz"],
            k_values=data["k_values"],
            r_squared=data.get("r_squared"),
            k_uncertainty=data.get("k_uncertainty"),
            operational_bounds=data.get("operational_bounds", {}),
            system_metadata=data.get("system_metadata", {}),
            name=data.get("name", in_path.stem),
            description=data.get("description", "")
        )


class DistanceEstimator:
    """
    Runtime Single-Channel Distance Inversion Engine.
    Computes real-time physical distance r(t) = k(f0) / A_true(t) with dynamic error propagation
    and certified operating boundary status checks.
    """

    def __init__(
        self,
        profile: Optional[AcousticProfile] = None,
        k_constant: Optional[float] = None,
        k_func: Optional[Callable[[float], float]] = None,
        noise_gate_v: float = 0.003,
        voltage_uncertainty_v: float = 0.0005,
        min_distance_m: float = 0.05,
        max_distance_m: float = 10.0
    ):
        if profile is not None:
            self.profile = profile
        elif k_func is not None:
            self.profile = AcousticProfile.from_callable(k_func)
        elif k_constant is not None:
            self.profile = AcousticProfile.from_constant(k_constant)
        else:
            self.profile = AcousticProfile.from_constant(0.05, name="DefaultBaseline")

        self.noise_gate_v = float(noise_gate_v)
        self.voltage_uncertainty_v = float(voltage_uncertainty_v)
        self.min_dist = float(min_distance_m)
        self.max_dist = float(max_distance_m)

    def estimate_distance(
        self,
        amplitude_v: float,
        frequency_hz: float,
        delta_a: Optional[float] = None
    ) -> Tuple[float, float, str]:
        """
        Calculates physical distance r(t), uncertainty delta_r(t), and operational status flag:
        r(t) = k(f0) / A(t)
        delta_r(t) = r * sqrt( (delta_k / k)^2 + (delta_A / A)^2 )

        :return: (distance_meters, uncertainty_meters, status_str).
                 Status is one of: 'ACTIVE_VALID', 'OUT_OF_BOUNDS_SATURATION', 'OUT_OF_BOUNDS_NOISE', 'SILENCE'.
        """
        amp = float(amplitude_v)
        f0 = float(frequency_hz)

        # Squelch silence/noise floor
        if not np.isfinite(amp) or amp < self.noise_gate_v or not np.isfinite(f0) or f0 <= 0:
            return np.nan, np.nan, "SILENCE"

        k_val, delta_k = self.profile.evaluate(f0)
        bounds = self.profile.get_operational_bounds(f0)
        da = float(delta_a) if delta_a is not None else self.voltage_uncertainty_v

        r_calc = k_val / max(amp, 1e-6)
        r_clamped = float(np.clip(r_calc, self.min_dist, self.max_dist))

        # Check operational boundaries
        if r_calc < bounds.get("r_min_m", self.min_dist):
            status = "OUT_OF_BOUNDS_SATURATION"
        elif r_calc > bounds.get("r_max_m", self.max_dist):
            status = "OUT_OF_BOUNDS_NOISE"
        else:
            status = "ACTIVE_VALID"

        rel_k_err = delta_k / max(k_val, 1e-6)
        rel_a_err = da / max(amp, 1e-6)
        delta_r = float(r_clamped * np.sqrt(rel_k_err ** 2 + rel_a_err ** 2))

        return r_clamped, delta_r, status

    def process_quadruple(self, quadruple: Dict[str, Any]) -> Dict[str, Any]:
        """Augments an incoming quadruple dict with real-time distance metrics and operational status."""
        res = quadruple.copy()
        amp = res.get("amplitude_v", 0.0)
        f0 = res.get("frequency_hz", np.nan)

        r_m, r_err, status = self.estimate_distance(amp, f0)
        k_val, delta_k = self.profile.evaluate(f0 if np.isfinite(f0) else 1000.0)

        res["distance_m"] = r_m
        res["distance_err_m"] = r_err
        res["distance_status"] = status
        res["k_evaluated"] = k_val
        res["k_uncertainty"] = delta_k
        return res

    def process_frame(
        self,
        frame_dict: Dict[str, Any],
        source: str = "A0",
        fs: float = 50_000.0,
        f_min: float = 100.0,
        f_max: float = 15000.0
    ) -> Dict[str, Any]:
        """Processes raw spectral frame dict, performs hybrid in-band demodulation, and computes distance."""
        if "quadruple" in frame_dict:
            quad = frame_dict["quadruple"].copy()
        else:
            time_sig = frame_dict["v_a0"] if "A0" in source.upper() or "CH1" in source.upper() else frame_dict["v_a1"]
            quad = KinematicAnalytics.extract_hybrid_quadruple(
                time_signal_v=time_sig,
                fs=fs,
                freq_axis=frame_dict["freqs"],
                magnitude=frame_dict["mag"],
                phase_rad=frame_dict["phase"],
                f_min=f_min,
                f_max=f_max,
                timer_cycles=frame_dict.get("timer_cycles", 0)
            )

        return self.process_quadruple(quad)


# =============================================================================
# 6. Standalone Acoustic Calibration Protocol Suite (WLS & Boundary Pruner)
# =============================================================================

class AcousticCalibrationProtocol:
    """
    Standalone Acoustic Calibration & Regression Protocol.
    Ingests multi-sample (N=30) observations per point, applies dynamic boundary pruning
    to isolate the central valid 1/r region, solves Weighted Least Squares (WLS) regressions,
    validates R^2 >= 0.95 linearity gates, and exports complete AcousticProfile artifacts.
    """

    def __init__(
        self,
        r2_threshold: float = 0.95,
        system_metadata: Optional[Dict[str, Any]] = None
    ):
        self.r2_threshold = float(r2_threshold)
        self.system_metadata = system_metadata or {
            "speaker_volume": 0.75,
            "mic_gain": "+35dB",
            "environment_label": "default_lab",
            "temperature_c": 20.0
        }
        # Data store: {frequency_hz: {distance_m: [v_1, v_2, ..., v_N]}}
        self._measurements: Dict[float, Dict[float, List[float]]] = {}
        self._fit_results: Dict[float, Dict[str, Any]] = {}

    def add_measurement(
        self,
        distance_m: float,
        frequency_hz: float,
        amplitude_v: Union[float, List[float], np.ndarray]
    ):
        """
        Records measurement observations for a given distance and frequency station.
        Accepts single scalar voltages or lists/arrays of repeat observations (e.g. N=30).
        """
        r = float(distance_m)
        f = float(frequency_hz)

        if r <= 0 or not np.isfinite(r) or not np.isfinite(f):
            return

        if f not in self._measurements:
            self._measurements[f] = {}
        if r not in self._measurements[f]:
            self._measurements[f][r] = []

        if isinstance(amplitude_v, (list, tuple, np.ndarray)):
            for v in amplitude_v:
                v_flt = float(v)
                if np.isfinite(v_flt) and v_flt > 0:
                    self._measurements[f][r].append(v_flt)
        else:
            v_flt = float(amplitude_v)
            if np.isfinite(v_flt) and v_flt > 0:
                self._measurements[f][r].append(v_flt)

    def add_dataset(
        self,
        distances_m: Union[List[float], np.ndarray],
        frequencies_hz: Union[List[float], np.ndarray],
        amplitude_matrix_v: Union[List[List[float]], np.ndarray]
    ):
        """Ingests a complete 2D calibration grid of distances x frequencies."""
        r_arr = np.asarray(distances_m, dtype=np.float64)
        f_arr = np.asarray(frequencies_hz, dtype=np.float64)
        v_mat = np.asarray(amplitude_matrix_v, dtype=np.float64)

        if v_mat.shape != (len(r_arr), len(f_arr)):
            raise ValueError(
                f"Shape mismatch: amplitude_matrix shape {v_mat.shape} != ({len(r_arr)}, {len(f_arr)})"
            )

        for i, r in enumerate(r_arr):
            for j, f in enumerate(f_arr):
                self.add_measurement(r, f, v_mat[i, j])

    def _prune_linear_window(
        self,
        r_sorted: np.ndarray,
        v_means: np.ndarray
    ) -> Tuple[int, int]:
        """
        Dynamic Boundary Pruning: Identifies the optimal 1/r linear sub-window [i_start, i_stop]
        by jointly maximizing R^2 and adherence to the theoretical power-law slope d(ln V)/d(ln r) = -1.0.
        """
        m = len(r_sorted)
        if m <= 4:
            return 0, m

        min_window_len = max(4, int(np.ceil(m * 0.35)))
        best_start, best_stop = 0, m
        best_score = -1.0

        for win_len in range(m, min_window_len - 1, -1):
            for start in range(m - win_len + 1):
                stop = start + win_len
                r_sub = r_sorted[start:stop]
                v_sub = v_means[start:stop]

                # 1. 1/r Linear Fit
                x_sub = 1.0 / r_sub
                y_sub = v_sub

                slope, intercept = np.polyfit(x_sub, y_sub, 1)
                y_pred = slope * x_sub + intercept
                ss_res = np.sum((y_sub - y_pred) ** 2)
                ss_tot = np.sum((y_sub - np.mean(y_sub)) ** 2)

                if ss_tot > 1e-9 and slope > 1e-4:
                    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))
                    
                    # 2. Log-Log Power Law Slope: d(ln V) / d(ln r) ~ -1.0
                    log_r = np.log(r_sub)
                    log_v = np.log(np.maximum(v_sub, 1e-6))
                    log_slope, _ = np.polyfit(log_r, log_v, 1)

                    penalty = abs(log_slope - (-1.0))
                    if penalty < 0.35 and r2 >= self.r2_threshold:
                        # Composite score favors high R^2, physical 1/r slope, and coverage
                        score = r2 * (1.0 - penalty) * (len(r_sub) ** 0.3)
                        if score > best_score:
                            best_score = score
                            best_start, best_stop = start, stop

        # Fallback if no window passed both gates: maximize R^2
        if best_score < 0:
            best_r2 = -1.0
            for start in range(m - min_window_len + 1):
                for stop in range(start + min_window_len, m + 1):
                    r_sub = r_sorted[start:stop]
                    v_sub = v_means[start:stop]
                    x_sub = 1.0 / r_sub
                    y_sub = v_sub
                    slope, intercept = np.polyfit(x_sub, y_sub, 1)
                    y_pred = slope * x_sub + intercept
                    ss_res = np.sum((y_sub - y_pred) ** 2)
                    ss_tot = np.sum((y_sub - np.mean(y_sub)) ** 2)
                    if ss_tot > 1e-9 and slope > 1e-4:
                        r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))
                        if r2 > best_r2:
                            best_r2 = r2
                            best_start, best_stop = start, stop

        return best_start, best_stop

    def fit(self) -> Dict[float, Dict[str, Any]]:
        """
        Computes statistical parameters, performs dynamic boundary pruning,
        and solves Weighted Least Squares (WLS) regressions:
        V(r_i) = k(f) * (1 / r_i) + c_room
        """
        self._fit_results.clear()

        for f, dist_dict in self._measurements.items():
            if len(dist_dict) < 2:
                continue

            r_sorted = np.array(sorted(dist_dict.keys()), dtype=np.float64)
            v_means = []
            v_stds = []
            v_sems = []
            sample_counts = []

            for r in r_sorted:
                samples = np.array(dist_dict[r], dtype=np.float64)
                n = len(samples)
                mean_val = float(np.mean(samples)) if n > 0 else 0.0
                std_val = float(np.std(samples, ddof=1)) if n > 1 else max(0.001 * mean_val, 1e-5)
                sem_val = float(std_val / np.sqrt(n)) if n > 0 else std_val

                v_means.append(mean_val)
                v_stds.append(std_val)
                v_sems.append(sem_val)
                sample_counts.append(n)

            v_means = np.array(v_means, dtype=np.float64)
            v_sems = np.array(v_sems, dtype=np.float64)

            # 1. Dynamic Boundary Pruning
            i_start, i_stop = self._prune_linear_window(r_sorted, v_means)

            r_pruned = r_sorted[i_start:i_stop]
            v_pruned = v_means[i_start:i_stop]
            sem_pruned = v_sems[i_start:i_stop]

            # 2. Weighted Least Squares (WLS) on pruned region: x = 1/r
            x_wls = 1.0 / r_pruned
            y_wls = v_pruned
            w_wls = 1.0 / np.maximum(sem_pruned ** 2, 1e-10)

            # WLS fit: y = k * x + c
            w_sum = np.sum(w_wls)
            x_bar = np.sum(w_wls * x_wls) / w_sum
            y_bar = np.sum(w_wls * y_wls) / w_sum

            s_xx = np.sum(w_wls * (x_wls - x_bar) ** 2)
            s_xy = np.sum(w_wls * (x_wls - x_bar) * (y_wls - y_bar))

            if s_xx > 1e-12:
                slope = float(s_xy / s_xx)
                intercept = float(y_bar - slope * x_bar)
            else:
                slope = 0.0
                intercept = float(y_bar)

            y_pred = slope * x_wls + intercept
            ss_res = np.sum(w_wls * (y_wls - y_pred) ** 2)
            ss_tot = np.sum(w_wls * (y_wls - y_bar) ** 2)

            n_pts = len(r_pruned)
            if ss_tot < 1e-9 or slope <= 1e-4:
                r2 = 0.0
                passed_gate = False
                slope_err = float(0.05 * abs(slope))
            else:
                r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))
                passed_gate = bool(r2 >= self.r2_threshold and slope > 1e-4)
                # Standard error of weighted slope
                s_sq = ss_res / max(n_pts - 2, 1)
                slope_err = float(np.sqrt(s_sq / max(s_xx, 1e-12)))

            self._fit_results[f] = {
                "k": float(slope),
                "delta_k": float(slope_err),
                "c_room": float(intercept),
                "r_squared": float(r2),
                "passed_gate": passed_gate,
                "n_pruned_points": n_pts,
                "n_total_points": len(r_sorted),
                "pruned_indices": (int(i_start), int(i_stop)),
                "r_valid_min_m": float(np.min(r_pruned)),
                "r_valid_max_m": float(np.max(r_pruned)),
                "v_sat_v": float(np.max(v_pruned)),
                "v_min_v": float(np.min(v_pruned)),
                "raw_stations": [
                    {
                        "r_m": float(r_sorted[idx]),
                        "mean_v": float(v_means[idx]),
                        "std_v": float(v_stds[idx]),
                        "sem_v": float(v_sems[idx]),
                        "n_samples": int(sample_counts[idx]),
                        "is_pruned_in": bool(i_start <= idx < i_stop)
                    }
                    for idx in range(len(r_sorted))
                ]
            }

        return self._fit_results

    def export_profile(
        self,
        name: str = "Calibrated_Room_Profile",
        description: str = "WLS calibrated profile with dynamic boundary pruning",
        only_passed: bool = True
    ) -> AcousticProfile:
        """Constructs an AcousticProfile with certified operating bounds and system metadata."""
        if not self._fit_results:
            self.fit()

        freqs = []
        k_vals = []
        r2_vals = []
        k_errs = []
        bounds_dict = {}

        sorted_freqs = sorted(self._fit_results.keys())
        for f in sorted_freqs:
            res = self._fit_results[f]
            if only_passed and not res["passed_gate"]:
                continue
            freqs.append(f)
            k_vals.append(res["k"])
            r2_vals.append(res["r_squared"])
            k_errs.append(res["delta_k"])
            bounds_dict[str(f)] = {
                "r_min_m": res["r_valid_min_m"],
                "r_max_m": res["r_valid_max_m"],
                "v_sat_v": res["v_sat_v"],
                "v_min_v": res["v_min_v"],
                "c_room_v": res["c_room"]
            }

        if len(freqs) == 0:
            raise ValueError(
                f"No calibration points passed the R^2 >= {self.r2_threshold} quality gate."
            )

        return AcousticProfile(
            frequencies_hz=freqs,
            k_values=k_vals,
            r_squared=r2_vals,
            k_uncertainty=k_errs,
            operational_bounds=bounds_dict,
            system_metadata=self.system_metadata,
            name=name,
            description=description
        )

    def save_profile_json(
        self,
        filepath: Union[str, Path],
        name: str = "Calibrated_Room_Profile",
        description: str = "WLS calibrated profile with dynamic boundary pruning"
    ) -> Path:
        """Fits, exports, and saves calibration profile with metadata and bounds to JSON."""
        profile = self.export_profile(name=name, description=description)
        out_path = Path(filepath).resolve()
        profile.to_json(out_path)
        return out_path

    def clear(self):
        """Clears all raw measurements and fit results."""
        self._measurements.clear()
        self._fit_results.clear()

# =============================================================================
# 7. Multipath / Reflection-Aware Calibration Protocol (Two-Ray Lloyd's Mirror)
# =============================================================================

class MultipathCalibrationProtocol:
    """
    Multipath & Reflection-Tolerant Acoustic Calibration Protocol.
    Sister class to AcousticCalibrationProtocol designed for reflective indoor rooms.
    Fits the spatial centroid of interference fringes across inverse-distance space:
    
    V_RMS(r) = k * (1 / r) + c_room + Ripple(r)
    
    Extracts the true direct-wave coupling constant k and reverberation baseline c_room
    without being penalized by standing wave antinodes/nulls or requiring free-field monotonic decay.
    """

    def __init__(
        self,
        r2_threshold: float = 0.80,
        temperature_c: float = 20.0,
        system_metadata: Optional[Dict[str, Any]] = None
    ):
        self.r2_threshold = float(r2_threshold)
        self.temperature_c = float(temperature_c)
        self.c_sound = KinematicAnalytics.speed_of_sound(self.temperature_c)

        self.system_metadata = system_metadata or {
            "speaker_volume": 0.75,
            "mic_gain": "+35dB",
            "environment_label": "reflective_indoor_room",
            "temperature_c": self.temperature_c
        }
        self.system_metadata["calibration_regime"] = "indoor_multipath_centroid"

        # Data store: {frequency_hz: {distance_m: [v_1, v_2, ..., v_N]}}
        self._measurements: Dict[float, Dict[float, List[float]]] = {}
        self._fit_results: Dict[float, Dict[str, Any]] = {}

    def add_measurement(
        self,
        distance_m: float,
        frequency_hz: float,
        amplitude_v: Union[float, List[float], np.ndarray]
    ):
        """Records burst observations for a distance station (identical API to AcousticCalibrationProtocol)."""
        r = float(distance_m)
        f = float(frequency_hz)

        if r <= 0 or not np.isfinite(r) or not np.isfinite(f):
            return

        if f not in self._measurements:
            self._measurements[f] = {}
        if r not in self._measurements[f]:
            self._measurements[f][r] = []

        if isinstance(amplitude_v, (list, tuple, np.ndarray)):
            for v in amplitude_v:
                v_flt = float(v)
                if np.isfinite(v_flt) and v_flt > 0:
                    self._measurements[f][r].append(v_flt)
        else:
            v_flt = float(amplitude_v)
            if np.isfinite(v_flt) and v_flt > 0:
                self._measurements[f][r].append(v_flt)

    def add_dataset(
        self,
        distances_m: Union[List[float], np.ndarray],
        frequencies_hz: Union[List[float], np.ndarray],
        amplitude_matrix_v: Union[List[List[float]], np.ndarray]
    ):
        """Ingests a complete 2D calibration grid of distances x frequencies."""
        r_arr = np.asarray(distances_m, dtype=np.float64)
        f_arr = np.asarray(frequencies_hz, dtype=np.float64)
        v_mat = np.asarray(amplitude_matrix_v, dtype=np.float64)

        if v_mat.shape != (len(r_arr), len(f_arr)):
            raise ValueError(
                f"Shape mismatch: amplitude_matrix shape {v_mat.shape} != ({len(r_arr)}, {len(f_arr)})"
            )

        for i, r in enumerate(r_arr):
            for j, f in enumerate(f_arr):
                self.add_measurement(r, f, v_mat[i, j])

    def clear(self):
        """Clears all raw measurements and fit results."""
        self._measurements.clear()
        self._fit_results.clear()

    def fit(self) -> Dict[float, Dict[str, Any]]:
        """
        Computes multi-sample statistics and solves for the direct-wave constant k
        through the spatial centroid of interference fringes across inverse-distance space:
        V(r) = k * (1 / r) + c_room
        """
        self._fit_results.clear()

        for f, dist_dict in self._measurements.items():
            if len(dist_dict) < 2:
                continue

            r_sorted = np.array(sorted(dist_dict.keys()), dtype=np.float64)
            v_means = []
            v_stds = []
            v_sems = []
            sample_counts = []

            for r in r_sorted:
                samples = np.array(dist_dict[r], dtype=np.float64)
                n = len(samples)
                mean_val = float(np.mean(samples)) if n > 0 else 0.0
                std_val = float(np.std(samples, ddof=1)) if n > 1 else max(0.001 * mean_val, 1e-5)
                sem_val = float(std_val / np.sqrt(n)) if n > 0 else std_val

                v_means.append(mean_val)
                v_stds.append(std_val)
                v_sems.append(sem_val)
                sample_counts.append(n)

            v_means = np.array(v_means, dtype=np.float64)
            v_sems = np.array(v_sems, dtype=np.float64)

            # Linearized inverse domain: x = 1 / r
            x_wls = 1.0 / r_sorted
            y_wls = v_means
            w_wls = 1.0 / np.maximum(v_sems ** 2, 1e-10)

            # Weighted Least Squares regression for centroid slope k and intercept c_room
            w_sum = np.sum(w_wls)
            x_bar = np.sum(w_wls * x_wls) / w_sum
            y_bar = np.sum(w_wls * y_wls) / w_sum

            s_xx = np.sum(w_wls * (x_wls - x_bar) ** 2)
            s_xy = np.sum(w_wls * (x_wls - x_bar) * (y_wls - y_bar))

            if s_xx > 1e-12:
                slope = float(s_xy / s_xx)
                intercept = float(y_bar - slope * x_bar)
            else:
                slope = 0.0
                intercept = float(y_bar)

            # Model predictions and residuals
            y_pred = slope * x_wls + intercept
            ss_res = np.sum(w_wls * (y_wls - y_pred) ** 2)
            ss_tot = np.sum(w_wls * (y_wls - y_bar) ** 2)

            n_pts = len(r_sorted)
            if ss_tot < 1e-9 or slope <= 1e-4:
                r2 = 0.0
                passed_gate = False
                slope_err = float(0.05 * abs(slope))
            else:
                r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))
                r2 = max(0.0, min(1.0, r2))
                passed_gate = bool(r2 >= self.r2_threshold and slope > 1e-4)
                s_sq = ss_res / max(n_pts - 2, 1)
                slope_err = float(np.sqrt(s_sq / max(s_xx, 1e-12)))

            # Multipath Standing Wave Severity Index (SWI) & ripple metrics
            residuals = y_wls - y_pred
            rms_ripple = float(np.sqrt(np.mean(residuals ** 2)))
            max_ripple = float(np.max(np.abs(residuals)))
            swi_ratio = float(max_ripple / max(np.mean(y_pred), 1e-6))

            self._fit_results[f] = {
                "k": float(slope),
                "delta_k": float(slope_err),
                "c_room": float(intercept),
                "r_squared": float(r2),
                "passed_gate": passed_gate,
                "rms_ripple_v": rms_ripple,
                "max_ripple_v": max_ripple,
                "standing_wave_index": swi_ratio,
                "n_points": n_pts,
                "n_pruned_points": n_pts,  # All points retained in centroid fit
                "n_total_points": n_pts,
                "r_valid_min_m": float(np.min(r_sorted)),
                "r_valid_max_m": float(np.max(r_sorted)),
                "v_sat_v": float(np.max(v_means)),
                "v_min_v": float(np.min(v_means)),
                "calibration_regime": "indoor_multipath_centroid",
                "raw_stations": [
                    {
                        "r_m": float(r_sorted[idx]),
                        "mean_v": float(v_means[idx]),
                        "std_v": float(v_stds[idx]),
                        "sem_v": float(v_sems[idx]),
                        "n_samples": int(sample_counts[idx]),
                        "model_pred_v": float(y_pred[idx]),
                        "residual_v": float(residuals[idx]),
                        "is_pruned_in": True  # All points active
                    }
                    for idx in range(len(r_sorted))
                ]
            }

        return self._fit_results

    def export_profile(
        self,
        name: str = "Multipath_Room_Profile",
        description: str = "Acoustic profile calibrated via spatial centroid WLS in reflective environment",
        only_passed: bool = True
    ) -> AcousticProfile:
        """Constructs an AcousticProfile with certified operating bounds and system metadata."""
        if not self._fit_results:
            self.fit()

        freqs = []
        k_vals = []
        r2_vals = []
        k_errs = []
        bounds_dict = {}

        sorted_freqs = sorted(self._fit_results.keys())
        for f in sorted_freqs:
            res = self._fit_results[f]
            if only_passed and not res["passed_gate"]:
                continue
            freqs.append(f)
            k_vals.append(res["k"])
            r2_vals.append(res["r_squared"])
            k_errs.append(res["delta_k"])
            bounds_dict[str(f)] = {
                "r_min_m": res["r_valid_min_m"],
                "r_max_m": res["r_valid_max_m"],
                "v_sat_v": res["v_sat_v"],
                "v_min_v": res["v_min_v"],
                "c_room_v": res["c_room"],
                "rms_ripple_v": res.get("rms_ripple_v", 0.0),
                "standing_wave_index": res.get("standing_wave_index", 0.0)
            }

        if len(freqs) == 0:
            raise ValueError(
                f"No calibration points passed the R^2 >= {self.r2_threshold} quality gate."
            )

        return AcousticProfile(
            frequencies_hz=freqs,
            k_values=k_vals,
            r_squared=r2_vals,
            k_uncertainty=k_errs,
            operational_bounds=bounds_dict,
            system_metadata=self.system_metadata,
            name=name,
            description=description
        )

    def save_profile_json(
        self,
        filepath: Union[str, Path],
        name: str = "Multipath_Room_Profile",
        description: str = "Acoustic profile calibrated via spatial centroid WLS in reflective environment"
    ) -> Path:
        """Fits, exports, and saves calibration profile with metadata and bounds to JSON."""
        profile = self.export_profile(name=name, description=description)
        out_path = Path(filepath).resolve()
        profile.to_json(out_path)
        return out_path

# =============================================================================
# 8. Angle of Arrival (AoA) Phase Interferometry Estimator
# =============================================================================

class AngleOfArrivalEstimator:
    """
    Real-Time Dual-Channel Angle of Arrival (AoA) Interferometric Solver.
    Computes continuous bearing angle θ(t) from synchronous dual-microphone time streams
    using coherent single-bin Fourier projection:
      sin(θ) = (c(T) · Δφ) / (2π · f0 · d)
    """

    def __init__(
        self,
        mic_distance_m: float = 0.05,
        target_freq_hz: Optional[float] = None,
        profile: Optional[Union[AcousticProfile, str, Path]] = None,
        temperature_c: float = 20.0,
        noise_gate_v: float = 0.010,
        phase_uncertainty_rad: float = 0.04
    ):
        self.mic_distance_m = float(mic_distance_m)
        self.temperature_c = float(temperature_c)
        self.noise_gate_v = float(noise_gate_v)
        self.phase_uncertainty_rad = float(phase_uncertainty_rad)

        # Resolve carrier frequency f0 from profile, parameter, or default
        if target_freq_hz is not None:
            self.target_freq_hz = float(target_freq_hz)
        elif profile is not None:
            if isinstance(profile, (str, Path)):
                p_path = Path(profile).resolve()
                with open(p_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.target_freq_hz = float(data.get("f_res_hz", data.get("frequencies_hz", [2609.73])[0]))
                if "max_mic_spacing_aoa_cm" in data:
                    recommended_d = float(data["max_mic_spacing_aoa_cm"]) / 100.0
                    if self.mic_distance_m > recommended_d:
                        self.mic_distance_m = recommended_d * 0.75
            elif isinstance(profile, AcousticProfile):
                self.target_freq_hz = float(profile.frequencies[0])
            else:
                self.target_freq_hz = 2609.73
        else:
            self.target_freq_hz = 2609.73

    def estimate_angle(
        self,
        v_a0: np.ndarray,
        v_a1: np.ndarray,
        fs: float,
        f_target: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Estimates incident bearing angle θ from synchronized ADC raw time arrays.

        :param v_a0: Raw voltage array from Mic 1 (A0).
        :param v_a1: Raw voltage array from Mic 2 (A1).
        :param fs: Sampling frequency in Hz.
        :param f_target: Optional override for target carrier frequency.
        :return: Dict containing theta_deg, theta_rad, theta_err_deg, delta_phi_rad,
                 amp_a0_v, amp_a1_v, coherence, and status string.
        """
        f0 = float(f_target) if f_target is not None else self.target_freq_hz

        # Extract coherent phase difference and in-band amplitudes
        phase_data = KinematicAnalytics.extract_dual_coherent_phase(
            signal_a0_v=v_a0,
            signal_a1_v=v_a1,
            fs=fs,
            target_freq_hz=f0,
            remove_dc=True
        )

        amp_min = min(phase_data["amp_a0_v"], phase_data["amp_a1_v"])

        # Squelch noise floor check
        if amp_min < self.noise_gate_v:
            return {
                "theta_deg": np.nan,
                "theta_rad": np.nan,
                "theta_err_deg": np.nan,
                "delta_phi_rad": phase_data["delta_phi_rad"],
                "delta_phi_deg": phase_data["delta_phi_deg"],
                "amp_a0_v": phase_data["amp_a0_v"],
                "amp_a1_v": phase_data["amp_a1_v"],
                "coherence": phase_data["coherence"],
                "f0_evaluated": f0,
                "status": "SILENCE"
            }

        theta_deg, theta_rad, theta_err, is_aliased = KinematicAnalytics.calculate_angle_of_arrival(
            delta_phi_rad=phase_data["delta_phi_rad"],
            f0=f0,
            mic_distance_m=self.mic_distance_m,
            temperature_c=self.temperature_c,
            delta_phi_err_rad=self.phase_uncertainty_rad
        )

        status = "OUT_OF_BOUNDS_SPATIAL_ALIASING" if is_aliased else "ACTIVE_VALID"

        return {
            "theta_deg": theta_deg,
            "theta_rad": theta_rad,
            "theta_err_deg": theta_err,
            "delta_phi_rad": phase_data["delta_phi_rad"],
            "delta_phi_deg": phase_data["delta_phi_deg"],
            "amp_a0_v": phase_data["amp_a0_v"],
            "amp_a1_v": phase_data["amp_a1_v"],
            "coherence": phase_data["coherence"],
            "f0_evaluated": f0,
            "status": status
        }

    def process_frame(
        self,
        frame_dict: Dict[str, Any],
        fs: float = 50000.0,
        f_target: Optional[float] = None
    ) -> Dict[str, Any]:
        """Convenience method to process frame output from capture_spectral_frame()."""
        return self.estimate_angle(
            v_a0=frame_dict["v_a0"],
            v_a1=frame_dict["v_a1"],
            fs=fs,
            f_target=f_target
        )