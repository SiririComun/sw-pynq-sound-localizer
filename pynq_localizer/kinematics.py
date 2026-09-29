"""
pynq_localizer.kinematics: High-Precision Acoustic Kinematics, Doppler Tracking & Metrology Engine.
Provides sub-Hertz pitch tracking (20 Hz - 20 kHz), spectral quadruple extraction (f, A, φ, t),
hybrid dual-DMA coherent in-band demodulation, multi-source tracking, dual-channel metric distance
estimation (r = k_A / A and r = sqrt(k_E / E)), dual-channel acoustic profile modeling, quasi-anechoic
direct-pulse calibration (zero-intercept 1/r and 1/r² WLS), phase and energy angle-of-arrival solvers,
Tracker optical cross-correlation alignment, theoretical Newtonian air-track simulators, and
statistical metrology aggregation (mean, std, SEM, MAE, RMSE, SNR).
"""

import json
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, List, Any, Callable
import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except (ImportError, ModuleNotFoundError):
    _HAS_PANDAS = False

try:
    import scipy.signal as signal
    from scipy.interpolate import interp1d
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except (ImportError, ModuleNotFoundError):
    _HAS_SCIPY = False


class KinematicAnalytics:
    """
    High-performance DSP and metrology engine for acoustic kinematics,
    energy decay modeling, and interferometric trajectory tracking.
    """

    # Certified nominal system constants
    DEFAULT_F0_HZ: float = 2660.0
    DEFAULT_TOA_OFFSET_MS: float = 1.8080

    # =========================================================================
    # 1. Physics Models & Temperature Compensation
    # =========================================================================

    @staticmethod
    def speed_of_sound(temperature_c: float = 20.0) -> float:
        """
        Calculates the temperature-compensated speed of sound in air:
          c(T) = 331.3 * sqrt(1 + T_c / 273.15) [m/s]
        """
        return float(331.3 * np.sqrt(1.0 + (float(temperature_c) / 273.15)))

    @classmethod
    def calculate_doppler_velocity(
        cls,
        f_observed: Union[float, np.ndarray],
        f_source: float = DEFAULT_F0_HZ,
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
    def calculate_differential_doppler_velocity(
        cls,
        f_mic1: Union[float, np.ndarray],
        f_mic2: Union[float, np.ndarray],
        temperature_c: float = 20.0
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Calculates instantaneous glider velocity v(t) along a 1D air track using two
        counter-positioned microphones (Mic 1 at x=0, Mic 2 at x=L):

          v(t) = c(T) · (f2(t) - f1(t)) / (f1(t) + f2(t))
          f0_common(t) = (f1(t) + f2(t)) / 2

        Sign Convention:
          • v > 0  => Glider moving RIGHT towards Mic 2 (Mic 2 is Blue, Mic 1 is Red)
          • v < 0  => Glider moving LEFT towards Mic 1  (Mic 1 is Blue, Mic 2 is Red)
          • v = 0  => Glider is stationary (f1 = f2 = f0)

        Completely eliminates buzzer oscillator thermal/battery drift because
        drift is common-mode to both sensors and cancels out identically.
        """
        c = cls.speed_of_sound(temperature_c)
        is_scalar = np.isscalar(f_mic1) and np.isscalar(f_mic2)

        f1 = np.asarray(f_mic1, dtype=np.float64)
        f2 = np.asarray(f_mic2, dtype=np.float64)

        denom = f1 + f2
        diff = f2 - f1

        with np.errstate(divide="ignore", invalid="ignore"):
            valid = (denom > 0.0) & np.isfinite(f1) & np.isfinite(f2)
            v = np.where(valid, c * (diff / denom), np.nan)
            f0_common = np.where(valid, 0.5 * denom, np.nan)

        if is_scalar:
            return float(v.item()), float(f0_common.item())
        return v, f0_common

    @classmethod
    def calculate_gravity_acceleration(
        cls,
        time_sec: np.ndarray,
        f_observed: np.ndarray,
        f_source: float = DEFAULT_F0_HZ,
        temperature_c: float = 20.0
    ) -> Dict[str, float]:
        """
        Calculates gravitational acceleration g from the linear frequency slope
        of a freely falling acoustic source:
          g = - (c(T) / f_0) * (df / dt)
        """
        t = np.asarray(time_sec, dtype=np.float64)
        f = np.asarray(f_observed, dtype=np.float64)

        valid_mask = np.isfinite(t) & np.isfinite(f)
        t_clean = t[valid_mask]
        f_clean = f[valid_mask]

        if len(t_clean) < 5:
            raise ValueError("Insufficient valid data points to perform regression for gravity.")

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
    # 2. Fourier Projections, Gated Direct Waves & Energy Metrics
    # =========================================================================

    @staticmethod
    def compute_coherent_inband_amplitude(
        signal_v: np.ndarray,
        fs: float,
        target_freq_hz: float = DEFAULT_F0_HZ,
        remove_dc: bool = True
    ) -> float:
        """
        Extracts physical in-band RMS voltage from raw 12-bit ADC time series at target_freq_hz:
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

        x_f0 = (2.0 / n) * np.dot(v_ac, phasor)
        return float(np.abs(x_f0) / np.sqrt(2.0))

    @staticmethod
    def extract_gated_direct_fourier(
        signal_v: np.ndarray,
        fs: float = 500_000.0,
        f0: float = DEFAULT_F0_HZ,
        n_cycles: int = 3,
        start_idx: int = 0,
        remove_dc: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates single-bin Discrete Fourier Transform over an exact integer multiple
        of carrier periods (K cycles) starting at the wavefront arrival instant.
        Because the integration window spans an exact integer number of periods,
        the Fourier basis functions are strictly orthogonal, eliminating spectral leakage
        and isolating the direct path from room reverberation.
        """
        v = np.asarray(signal_v, dtype=np.float64)
        total_len = len(v)

        if total_len == 0 or not np.isfinite(f0) or f0 <= 0 or fs <= 0:
            return {
                "amplitude_v": 0.0,
                "amplitude_peak_v": 0.0,
                "energy_v2": 0.0,
                "phase_rad": 0.0,
                "phase_deg": 0.0,
                "n_samples_gated": 0,
                "gate_duration_ms": 0.0,
                "is_valid": False
            }

        samples_per_cycle = int(round(float(fs) / float(f0)))
        n_gate = max(1, int(n_cycles) * samples_per_cycle)

        s0 = max(0, int(start_idx))
        s1 = min(total_len, s0 + n_gate)
        slice_v = v[s0:s1]
        n_actual = len(slice_v)

        if n_actual < max(16, samples_per_cycle // 2):
            return {
                "amplitude_v": 0.0,
                "amplitude_peak_v": 0.0,
                "energy_v2": 0.0,
                "phase_rad": 0.0,
                "phase_deg": 0.0,
                "n_samples_gated": n_actual,
                "gate_duration_ms": (n_actual / float(fs)) * 1000.0,
                "is_valid": False
            }

        v_ac = slice_v - np.mean(slice_v) if remove_dc else slice_v

        t_local = np.arange(n_actual) / float(fs)
        phasor = np.exp(-2.0j * np.pi * float(f0) * t_local)

        x_f0 = (2.0 / float(n_actual)) * np.dot(v_ac, phasor)

        v_peak = float(np.abs(x_f0))
        v_rms = v_peak / np.sqrt(2.0)
        phi_rad = float(np.angle(x_f0))
        energy_v2 = float(np.sum(v_ac ** 2))
        gate_ms = (float(n_actual) / float(fs)) * 1000.0

        return {
            "amplitude_v": v_rms,
            "amplitude_peak_v": v_peak,
            "energy_v2": energy_v2,
            "phase_rad": phi_rad,
            "phase_deg": float(np.degrees(phi_rad)),
            "n_samples_gated": n_actual,
            "gate_duration_ms": gate_ms,
            "is_valid": True
        }

    # =========================================================================
    # 3. Dual-Channel Phase Interferometry & Energy Angle of Arrival
    # =========================================================================

    @staticmethod
    def extract_dual_coherent_phase(
        signal_a0_v: np.ndarray,
        signal_a1_v: np.ndarray,
        fs: float,
        target_freq_hz: float = DEFAULT_F0_HZ,
        remove_dc: bool = True
    ) -> Dict[str, float]:
        """
        Projects both synchronous ADC channels (A0 and A1) onto a Hann-windowed
        Fourier phasor to extract individual phases, RMS voltages, and wrapped Δφ.
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
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1)) if n > 1 else np.ones(n)
        coherent_sum = float(np.sum(w))
        phasor = w * np.exp(-2.0j * np.pi * float(target_freq_hz) * t)

        scale = 2.0 / max(coherent_sum, 1e-12)
        x0 = scale * np.dot(v0_ac, phasor)
        x1 = scale * np.dot(v1_ac, phasor)

        v_rms_0 = float(np.abs(x0) / np.sqrt(2.0))
        v_rms_1 = float(np.abs(x1) / np.sqrt(2.0))

        phi_0 = float(np.angle(x0))
        phi_1 = float(np.angle(x1))

        dphi_raw = phi_1 - phi_0
        delta_phi = float(np.arctan2(np.sin(dphi_raw), np.cos(dphi_raw)))

        denom = np.abs(x0) * np.abs(x1)
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
        f0: float = DEFAULT_F0_HZ,
        mic_distance_m: float = 0.05,
        temperature_c: float = 20.0,
        delta_phi_err_rad: float = 0.05
    ) -> Tuple[float, float, float, bool]:
        """
        Inverts wrapped acoustic phase difference Δφ to incident bearing angle θ:
          sin(θ) = (c(T) · Δφ) / (2π · f0 · d)
          θ = arcsin(clip(sin(θ), -1.0, 1.0))
        """
        c = cls.speed_of_sound(temperature_c)
        f_val = float(f0)
        d_val = float(mic_distance_m)

        if f_val <= 0 or d_val <= 0 or not np.isfinite(delta_phi_rad):
            return np.nan, np.nan, np.nan, False

        d_max_aliasing = c / (2.0 * f_val)
        is_aliased = bool(d_val > d_max_aliasing)

        ratio = (c * float(delta_phi_rad)) / (2.0 * np.pi * f_val * d_val)
        clamped_ratio = float(np.clip(ratio, -1.0, 1.0))

        theta_rad = float(np.arcsin(clamped_ratio))
        theta_deg = float(np.degrees(theta_rad))

        cos_theta = max(abs(np.cos(theta_rad)), 0.05)
        theta_err_rad = float((c / (2.0 * np.pi * f_val * d_val * cos_theta)) * float(delta_phi_err_rad))
        theta_err_deg = float(np.degrees(theta_err_rad))

        return theta_deg, theta_rad, theta_err_deg, is_aliased

    @staticmethod
    def calculate_energy_bearing(
        energy_mic1: float,
        energy_mic2: float,
        range_m: float,
        mic_distance_m: float = 0.05
    ) -> Dict[str, float]:
        """
        Computes incident bearing angle using normalized differential acoustic energy:
          r1 ≈ r + (d/2) sin(θ),   r2 ≈ r - (d/2) sin(θ)
          (E2 - E1) / (E1 + E2) ≈ (d / r) * sin(θ)
          sin(θ) = (r / d) * ((E2 - E1) / (E1 + E2))

        Sign Convention:
          • E2 > E1 => Closer to Mic 2 (Right / +θ)
          • E1 > E2 => Closer to Mic 1 (Left / -θ)
        """
        e1 = float(max(0.0, energy_mic1))
        e2 = float(max(0.0, energy_mic2))
        r = float(max(0.01, range_m))
        d = float(max(0.001, mic_distance_m))

        denom = e1 + e2
        if denom <= 1e-12:
            return {
                "theta_energy_deg": np.nan,
                "theta_energy_rad": np.nan,
                "delta_energy_norm": 0.0,
                "energy_ratio_db": 0.0
            }

        norm_diff = (e2 - e1) / denom
        ratio_sin = (r / d) * norm_diff
        sin_clamped = float(np.clip(ratio_sin, -1.0, 1.0))

        th_rad = float(np.arcsin(sin_clamped))
        th_deg = float(np.degrees(th_rad))
        energy_ratio_db = float(10.0 * np.log10(max(e2, 1e-9) / max(e1, 1e-9)))

        return {
            "theta_energy_deg": th_deg,
            "theta_energy_rad": th_rad,
            "delta_energy_norm": float(norm_diff),
            "energy_ratio_db": energy_ratio_db
        }

    # =========================================================================
    # 4. Statistical Metrology & Uncertainty Aggregator
    # =========================================================================

    @staticmethod
    def compute_metrology_statistics(
        measurements: Union[List[float], np.ndarray],
        ground_truth: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Aggregates repeat measurement shots into formal metrological metrics:
        mean (μ), sample standard deviation / jitter (σ), standard error (SEM = σ/√N),
        MAE, RMSE, and bias against ground truth.
        """
        arr = np.asarray(measurements, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        n = len(valid)

        if n == 0:
            return {
                "n_samples": 0,
                "mean": np.nan,
                "std": np.nan,
                "sem": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "bias": np.nan,
                "ground_truth": float(ground_truth) if ground_truth is not None else np.nan
            }

        mean_val = float(np.mean(valid))
        std_val = float(np.std(valid, ddof=1)) if n > 1 else 0.0
        sem_val = float(std_val / np.sqrt(n)) if n > 0 else 0.0

        stats = {
            "n_samples": int(n),
            "mean": mean_val,
            "std": std_val,
            "sem": sem_val,
            "ground_truth": float(ground_truth) if ground_truth is not None else np.nan
        }

        if ground_truth is not None:
            gt = float(ground_truth)
            residuals = valid - gt
            stats["bias"] = float(mean_val - gt)
            stats["mae"] = float(np.mean(np.abs(residuals)))
            stats["rmse"] = float(np.sqrt(np.mean(residuals ** 2)))
            stats["error_pct"] = float(abs(mean_val - gt) / max(abs(gt), 1e-6) * 100.0)
        else:
            stats["bias"] = np.nan
            stats["mae"] = np.nan
            stats["rmse"] = np.nan
            stats["error_pct"] = np.nan

        return stats

    # =========================================================================
    # 5. Hybrid Quadruple & Pitch Telemetry
    # =========================================================================

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
          1. Pitch (f0) & Phase (phi) extracted from DMA 1 (FFT/CORDIC).
          2. Coherent in-band Amplitude (A_true) extracted from DMA 0 (Time Stream).
        Guarantees 100% immunity to FPGA FFT Block Floating Point bit-shifts.
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
        Extracts the physical quadruple (f0, A, phi, t) from polar spectral frames.
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
    # 6. Sinc Ratio Pitch Tracking, Envelopes & Wavefronts
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

    @staticmethod
    def detect_pulse_arrival_time(
        signal_v: np.ndarray,
        fs: float,
        target_freq_hz: float = DEFAULT_F0_HZ,
        threshold_ratio: float = 0.50,
        band_half_width_hz: float = 250.0
    ) -> Dict[str, Any]:
        """
        Extracts the sub-sample arrival timestamp of an acoustic pulse burst using
        Hilbert analytic envelope tracking and linear threshold interpolation.
        """
        v = np.asarray(signal_v, dtype=np.float64)
        n = len(v)
        if n < 32 or not np.isfinite(target_freq_hz) or target_freq_hz <= 0:
            return {"t_arrival_sec": np.nan, "amp_peak_v": 0.0, "snr_db": 0.0, "is_detected": False}

        nyq = fs / 2.0
        low = max(20.0, target_freq_hz - band_half_width_hz) / nyq
        high = min(nyq - 20.0, target_freq_hz + band_half_width_hz) / nyq

        b, a = signal.butter(N=3, Wn=[low, high], btype="bandpass")
        v_clean = signal.filtfilt(b, a, v - np.mean(v))

        env = np.abs(KinematicAnalytics.hilbert_transform(v_clean))
        amp_peak = float(np.max(env))

        if amp_peak < 1e-4:
            return {"t_arrival_sec": np.nan, "amp_peak_v": 0.0, "snr_db": 0.0, "is_detected": False}

        noise_sigma = float(np.std(v_clean[: min(100, n // 4)]))
        snr_db = float(20.0 * np.log10(max(amp_peak, 1e-6) / max(noise_sigma, 1e-6)))

        v_thresh = float(threshold_ratio * amp_peak)
        crossing_indices = np.where(env >= v_thresh)[0]

        if len(crossing_indices) == 0 or snr_db < 6.0:
            return {"t_arrival_sec": np.nan, "amp_peak_v": amp_peak, "snr_db": snr_db, "is_detected": False}

        idx_arr = crossing_indices[0]

        if idx_arr > 0:
            y_prev = env[idx_arr - 1]
            y_curr = env[idx_arr]
            denom = y_curr - y_prev
            frac = (v_thresh - y_prev) / denom if denom > 1e-12 else 0.0
            frac = max(0.0, min(1.0, frac))
            t_sub_sample = (float(idx_arr - 1) + frac) / float(fs)
        else:
            t_sub_sample = float(idx_arr) / float(fs)

        return {
            "t_arrival_sec": t_sub_sample,
            "amp_peak_v": amp_peak,
            "snr_db": snr_db,
            "is_detected": True
        }

    # =========================================================================
    # 7. Exact Near-Field Multilateration & Parity Protection
    # =========================================================================

    @staticmethod
    def solve_2d_multilateration(
        r1_m: float,
        r2_m: float,
        d_m: float,
        f0: float = DEFAULT_F0_HZ,
        c_sound: float = 343.21,
        wrap_modulo_lambda: bool = True,
        delta_r_m: Optional[float] = None
    ) -> Dict[str, Union[float, str]]:
        """
        Solves exact 2D Cartesian and polar acoustic multilateration from dual-channel ranges
        with O(1) non-iterative modulo-lambda cycle-slip parity protection.

        Geometry convention:
          - Origin (0, 0) is the baseline center.
          - Mic 1 (Left / A0) is at (-d/2, 0).
          - Mic 2 (Right / A1) is at (+d/2, 0).
          - Forward axis is +y (theta = 0° at broadside).
          - Transverse axis is +x (theta > 0° to the Right, theta < 0° to the Left).
        """
        r1 = float(r1_m)
        r2 = float(r2_m)
        d = float(d_m)

        if not (np.isfinite(r1) and np.isfinite(r2) and r1 > 0.0 and r2 > 0.0 and d > 0.0):
            return {
                "x_m": np.nan,
                "y_m": np.nan,
                "range_m": np.nan,
                "theta_deg": np.nan,
                "theta_far_deg": np.nan,
                "status": "SILENCE"
            }

        delta_r = float(delta_r_m) if delta_r_m is not None else (r1 - r2)
        is_geometric_anomaly = False

        if wrap_modulo_lambda:
            lambda_m = c_sound / f0 if f0 > 0 else 0.129
            # O(1) Non-Iterative Modulo-Lambda Symmetric Wrap
            delta_r_corr = (delta_r + (lambda_m / 2.0)) % lambda_m - (lambda_m / 2.0)
            if abs(delta_r_corr) > d:
                is_geometric_anomaly = True
                delta_r_clamped = float(np.clip(delta_r_corr, -d, d))
            else:
                delta_r_clamped = delta_r_corr
        else:
            if abs(delta_r) > d:
                is_geometric_anomaly = True
                delta_r_clamped = float(np.clip(delta_r, -d, d))
            else:
                delta_r_clamped = delta_r

        ratio_far = delta_r_clamped / d
        theta_far_rad = float(np.arcsin(np.clip(ratio_far, -1.0, 1.0)))
        theta_far_deg = float(np.degrees(theta_far_rad))

        if is_geometric_anomaly:
            range_approx = 0.5 * (r1 + r2)
            x = range_approx * np.sin(theta_far_rad)
            y = 0.0
            status = "GEOMETRIC_OUT_OF_BOUNDS"
        else:
            r_mean = 0.5 * (r1 + r2)
            r1_eff = r_mean + 0.5 * delta_r_clamped
            r2_eff = r_mean - 0.5 * delta_r_clamped

            x = (r1_eff ** 2 - r2_eff ** 2) / (2.0 * d)
            y_radicand = ((r1_eff ** 2 + r2_eff ** 2) / 2.0) - (x ** 2) - ((d ** 2) / 4.0)

            if y_radicand < 0.0:
                y = 0.0
                status = "OUT_OF_PLANE_COLLAPSE"
            else:
                y = float(np.sqrt(y_radicand))
                status = "ACTIVE_VALID"

        range_center = float(np.sqrt(x ** 2 + y ** 2))
        theta_deg = float(np.degrees(np.arctan2(x, y)))

        return {
            "x_m": float(x),
            "y_m": float(y),
            "range_m": range_center,
            "theta_deg": theta_deg,
            "theta_far_deg": theta_far_deg,
            "status": status
        }

    # =========================================================================
    # 8. Tracker Optical Synchronization & Newtonian Simulators
    # =========================================================================

    @staticmethod
    def align_tracker_ground_truth(
        tracker_csv_path: Union[str, Path],
        t_fpga: np.ndarray,
        v_fpga: np.ndarray
    ) -> Dict[str, Any]:
        """
        Parses Tracker video analysis export (.csv / .txt), synchronizes timestamps
        via velocity cross-correlation, interpolates onto the FPGA 100 Hz timeline,
        and computes MAE, RMSE, and Pearson correlation.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas is required to align Tracker ground-truth data.")

        csv_p = Path(tracker_csv_path).resolve()
        df_tr = pd.read_csv(csv_p, sep=None, engine="python")

        t_cols = [c for c in df_tr.columns if c.strip().lower() in ("t", "time", "t_sec")]
        v_cols = [c for c in df_tr.columns if c.strip().lower() in ("v", "vx", "vel", "velocity", "v_mps")]

        if not t_cols or not v_cols:
            raise ValueError(f"Could not find time/velocity columns in {csv_p.name}. Found: {list(df_tr.columns)}")

        t_tr = df_tr[t_cols[0]].to_numpy(dtype=np.float64)
        v_tr = df_tr[v_cols[0]].to_numpy(dtype=np.float64)

        valid_tr = np.isfinite(t_tr) & np.isfinite(v_tr)
        t_tr = t_tr[valid_tr]
        v_tr = v_tr[valid_tr]

        dt_fpga = float(t_fpga[1] - t_fpga[0]) if len(t_fpga) > 1 else 0.010
        t_tr_resampled = np.arange(t_tr[0], t_tr[-1], dt_fpga)
        v_tr_resampled = np.interp(t_tr_resampled, t_tr, v_tr)

        v_fpga_ac = v_fpga - np.mean(v_fpga[np.isfinite(v_fpga)])
        v_tr_ac = v_tr_resampled - np.mean(v_tr_resampled)

        corr = np.correlate(v_fpga_ac, v_tr_ac, mode="full")
        lags = np.arange(-len(v_tr_resampled) + 1, len(v_fpga))
        best_lag = lags[np.argmax(corr)]
        # Invert lag sign so positive offset shifts delayed camera timeline back to FPGA t0
        time_offset_sec = -float(best_lag * dt_fpga)

        v_tr_aligned = np.interp(t_fpga + time_offset_sec, t_tr, v_tr, left=np.nan, right=np.nan)

        eval_mask = np.isfinite(v_fpga) & np.isfinite(v_tr_aligned)
        if np.sum(eval_mask) > 5:
            mae = float(np.mean(np.abs(v_fpga[eval_mask] - v_tr_aligned[eval_mask])))
            rmse = float(np.sqrt(np.mean((v_fpga[eval_mask] - v_tr_aligned[eval_mask]) ** 2)))
            r_val = float(np.corrcoef(v_fpga[eval_mask], v_tr_aligned[eval_mask])[0, 1])
        else:
            mae, rmse, r_val = np.nan, np.nan, np.nan

        return {
            "v_tracker_aligned_mps": v_tr_aligned,
            "v_tracker_aligned_cmps": v_tr_aligned * 100.0,
            "time_offset_sec": time_offset_sec,
            "mae_mps": mae,
            "rmse_mps": rmse,
            "correlation_r2": (r_val ** 2) if np.isfinite(r_val) else np.nan,
            "n_matched_points": int(np.sum(eval_mask))
        }

    @staticmethod
    def simulate_atwood_glider(
        t_axis: np.ndarray,
        m_glider_kg: float,
        m_hanging_kg: float,
        fall_height_m: float,
        gamma_damping: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates theoretical velocity v(t) and acceleration a(t) for an air-track
        glider propelled by a falling mass (Modified Atwood Machine):
          Phase 1 (Fall): a = (m / (M + m)) * g,   v(t) = a * t
          Phase 2 (Coast): v(t) = v_max * exp(-gamma * t),  a(t) = -gamma * v(t)
        """
        g = 9.80665
        m_tot = m_glider_kg + m_hanging_kg
        a_const = (m_hanging_kg / m_tot) * g if m_tot > 0 else 0.0

        t_impact = np.sqrt(2.0 * fall_height_m / a_const) if a_const > 0 else 0.0
        v_max = a_const * t_impact

        t = np.asarray(t_axis, dtype=np.float64)
        v_theo = np.zeros_like(t)
        a_theo = np.zeros_like(t)

        for i, ti in enumerate(t):
            if ti <= t_impact:
                v_theo[i] = a_const * ti
                a_theo[i] = a_const
            else:
                dt_coast = ti - t_impact
                v_theo[i] = v_max * np.exp(-gamma_damping * dt_coast)
                a_theo[i] = -gamma_damping * v_theo[i]

        return v_theo, a_theo

    @staticmethod
    def simulate_elastic_glider(
        t_axis: np.ndarray,
        m_glider_kg: float,
        k_spring_npm: float,
        delta_x_m: float,
        gamma_damping: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates theoretical velocity and acceleration for an air-track glider
        launched via elastic band recoil:
          E_pot = 0.5 * k * dx²  ==>  v_0 = dx * sqrt(k / M)
          Coasting with viscous drag: v(t) = v_0 * exp(-gamma * t)
        """
        v0 = delta_x_m * np.sqrt(k_spring_npm / m_glider_kg) if m_glider_kg > 0 else 0.0

        t = np.asarray(t_axis, dtype=np.float64)
        v_theo = v0 * np.exp(-gamma_damping * t)
        a_theo = -gamma_damping * v_theo

        return v_theo, a_theo


# =============================================================================
# MultiSourceTracker (Multi-Tone Telemetry Engine)
# =============================================================================

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
        """Processes a single polar FFT frame, cancels harmonic cross-talk, and computes SIR."""
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
                        bw = (self.source_bands[j_name][1] - self.source_bands[j_name][0]) * 0.5
                        if abs(f_j - 2.0 * f_i) < bw:
                            total_leakage += p_i * self.h2_coeff
                        elif abs(f_j - 3.0 * f_i) < bw:
                            total_leakage += p_i * self.h3_coeff

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
# 9. Dual-Microphone Acoustic Profile & Calibration Model
# =============================================================================

class AcousticProfile:
    """
    Data model, interpolator, and serializer for acoustic calibration parameters.
    Supports dual-channel parameters:
      • Amplitude coupling constants: k_A1, k_A2 (in V·m) for p(r) ∝ 1/r
      • Energy coupling constants:    k_E1, k_E2 (in counts²·m²) for I(r) ∝ 1/r²
      • Calibrated ToA offsets:       t_offset1, t_offset2 (in ms)
      • Certified operating distance/voltage bounds: [r_min(f), r_max(f)]
    """

    def __init__(
        self,
        frequencies_hz: Optional[Union[List[float], np.ndarray]] = None,
        k_values: Optional[Union[List[float], np.ndarray]] = None,
        k_values_m2: Optional[Union[List[float], np.ndarray]] = None,
        k_energy_m1: Optional[Union[List[float], np.ndarray]] = None,
        k_energy_m2: Optional[Union[List[float], np.ndarray]] = None,
        r_squared: Optional[Union[List[float], np.ndarray]] = None,
        k_uncertainty: Optional[Union[List[float], np.ndarray]] = None,
        k_uncertainty_m2: Optional[Union[List[float], np.ndarray]] = None,
        operational_bounds: Optional[Dict[str, Dict[str, float]]] = None,
        system_metadata: Optional[Dict[str, Any]] = None,
        name: str = "Active_Buzzer_Profile",
        description: str = "Dual-microphone acoustic calibration profile"
    ):
        self.name = name
        self.description = description
        self._callable_model: Optional[Callable[[float], float]] = None
        self.operational_bounds = operational_bounds or {}
        self.system_metadata = system_metadata or {}

        # Default physical turn-on delays and skew
        self.calibrated_toa_offset_m1_ms = float(self.system_metadata.get("calibrated_toa_offset_m1_ms",
                                                 self.system_metadata.get("calibrated_toa_offset_ms", 1.8080)))
        self.calibrated_toa_offset_m2_ms = float(self.system_metadata.get("calibrated_toa_offset_m2_ms",
                                                 self.calibrated_toa_offset_m1_ms))
        self.inter_channel_skew_us = float((self.calibrated_toa_offset_m1_ms - self.calibrated_toa_offset_m2_ms) * 1000.0)

        # Frequencies grid
        if frequencies_hz is not None:
            self.frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        else:
            self.frequencies = np.array([KinematicAnalytics.DEFAULT_F0_HZ], dtype=np.float64)

        # Channel 1 Amplitude constants k_A1 (V*m)
        if k_values is not None:
            self.k_values = np.asarray(k_values, dtype=np.float64)
        else:
            self.k_values = np.array([0.0196], dtype=np.float64)

        # Channel 2 Amplitude constants k_A2 (V*m)
        if k_values_m2 is not None:
            self.k_values_m2 = np.asarray(k_values_m2, dtype=np.float64)
        else:
            self.k_values_m2 = np.copy(self.k_values)

        # Energy constants k_E (counts^2 * m^2) for E = k_E / r^2
        if k_energy_m1 is not None:
            self.k_energy_m1 = np.asarray(k_energy_m1, dtype=np.float64)
        else:
            # Fallback estimation based on ADC scaling: (V_direct / (3.3 / 4095))^2
            adc_scale = 3.3 / 4095.0
            self.k_energy_m1 = (self.k_values / adc_scale) ** 2

        if k_energy_m2 is not None:
            self.k_energy_m2 = np.asarray(k_energy_m2, dtype=np.float64)
        else:
            self.k_energy_m2 = np.copy(self.k_energy_m1)

        # Regression Quality Gates & Uncertainties
        self.r_squared = (
            np.asarray(r_squared, dtype=np.float64)
            if r_squared is not None
            else np.ones_like(self.k_values)
        )
        self.k_uncertainty = (
            np.asarray(k_uncertainty, dtype=np.float64)
            if k_uncertainty is not None
            else 0.02 * self.k_values
        )
        self.k_uncertainty_m2 = (
            np.asarray(k_uncertainty_m2, dtype=np.float64)
            if k_uncertainty_m2 is not None
            else 0.02 * self.k_values_m2
        )

        # Build interpolators
        self._build_interpolators()

    def _build_interpolators(self):
        """Constructs 1D continuous spline/linear interpolators for both channels."""
        if len(self.frequencies) > 1:
            kind = "cubic" if (_HAS_SCIPY and len(self.frequencies) >= 4) else "linear"
            if _HAS_SCIPY:
                self._interp_k1 = interp1d(
                    self.frequencies, self.k_values, kind=kind,
                    bounds_error=False, fill_value=(self.k_values[0], self.k_values[-1])
                )
                self._interp_k2 = interp1d(
                    self.frequencies, self.k_values_m2, kind=kind,
                    bounds_error=False, fill_value=(self.k_values_m2[0], self.k_values_m2[-1])
                )
                self._interp_ke1 = interp1d(
                    self.frequencies, self.k_energy_m1, kind=kind,
                    bounds_error=False, fill_value=(self.k_energy_m1[0], self.k_energy_m1[-1])
                )
                self._interp_ke2 = interp1d(
                    self.frequencies, self.k_energy_m2, kind=kind,
                    bounds_error=False, fill_value=(self.k_energy_m2[0], self.k_energy_m2[-1])
                )
                self._interp_err1 = interp1d(
                    self.frequencies, self.k_uncertainty, kind="linear",
                    bounds_error=False, fill_value=(self.k_uncertainty[0], self.k_uncertainty[-1])
                )
                self._interp_err2 = interp1d(
                    self.frequencies, self.k_uncertainty_m2, kind="linear",
                    bounds_error=False, fill_value=(self.k_uncertainty_m2[0], self.k_uncertainty_m2[-1])
                )
            else:
                self._interp_k1 = lambda f: float(np.interp(f, self.frequencies, self.k_values))
                self._interp_k2 = lambda f: float(np.interp(f, self.frequencies, self.k_values_m2))
                self._interp_ke1 = lambda f: float(np.interp(f, self.frequencies, self.k_energy_m1))
                self._interp_ke2 = lambda f: float(np.interp(f, self.frequencies, self.k_energy_m2))
                self._interp_err1 = lambda f: float(np.interp(f, self.frequencies, self.k_uncertainty))
                self._interp_err2 = lambda f: float(np.interp(f, self.frequencies, self.k_uncertainty_m2))
        else:
            self._interp_k1 = lambda f: float(self.k_values[0])
            self._interp_k2 = lambda f: float(self.k_values_m2[0])
            self._interp_ke1 = lambda f: float(self.k_energy_m1[0])
            self._interp_ke2 = lambda f: float(self.k_energy_m2[0])
            self._interp_err1 = lambda f: float(self.k_uncertainty[0])
            self._interp_err2 = lambda f: float(self.k_uncertainty_m2[0])

    def evaluate(self, frequency_hz: float, channel: int = 1) -> Tuple[float, float]:
        """
        Evaluates physical amplitude coupling constant k_A(f) and uncertainty delta_k
        for the specified microphone channel (1 for Mic 1 / A0, 2 for Mic 2 / A1).
        """
        f = float(frequency_hz)
        if not np.isfinite(f) or f <= 0:
            f = float(self.frequencies[0])

        if self._callable_model is not None:
            k_val = float(self._callable_model(f))
            return k_val, 0.02 * k_val

        if channel == 2:
            return float(self._interp_k2(f)), float(self._interp_err2(f))
        return float(self._interp_k1(f)), float(self._interp_err1(f))

    def evaluate_energy(self, frequency_hz: float, channel: int = 1) -> Tuple[float, float]:
        """
        Evaluates physical energy coupling constant k_E(f) and uncertainty delta_k_E
        for the specified microphone channel (1 for Mic 1 / A0, 2 for Mic 2 / A1).
        """
        f = float(frequency_hz)
        if not np.isfinite(f) or f <= 0:
            f = float(self.frequencies[0])

        if channel == 2:
            k_e = float(self._interp_ke2(f))
            return k_e, 0.03 * k_e
        k_e = float(self._interp_ke1(f))
        return k_e, 0.03 * k_e

    def get_operational_bounds(self, frequency_hz: float) -> Dict[str, float]:
        """Retrieves certified physical operating boundaries [r_min, r_max, v_sat, v_min]."""
        f = float(frequency_hz)
        if not self.operational_bounds:
            return {"r_min_m": 0.10, "r_max_m": 2.00, "v_min_v": 0.002, "v_sat_v": 0.600}

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
        """Factory creating an AcousticProfile evaluated directly from a mathematical function."""
        profile = cls.from_constant(0.0196, system_metadata=system_metadata, name=name)
        profile._callable_model = func
        return profile

    def to_json(self, filepath: Union[str, Path]):
        """Serializes dual-channel calibration constants, metadata, and bounds to JSON."""
        out_path = Path(filepath).resolve()
        data = {
            "name": self.name,
            "description": self.description,
            "system_metadata": self.system_metadata,
            "operational_bounds": self.operational_bounds,
            "f_res_hz": float(self.frequencies[0]),
            "calibrated_toa_offset_m1_ms": self.calibrated_toa_offset_m1_ms,
            "calibrated_toa_offset_m2_ms": self.calibrated_toa_offset_m2_ms,
            "inter_channel_skew_us": self.inter_channel_skew_us,
            "frequencies_hz": self.frequencies.tolist(),
            "k_values": self.k_values.tolist(),
            "k_values_m2": self.k_values_m2.tolist(),
            "k_energy_m1": self.k_energy_m1.tolist(),
            "k_energy_m2": self.k_energy_m2.tolist(),
            "r_squared": self.r_squared.tolist(),
            "k_uncertainty": self.k_uncertainty.tolist(),
            "k_uncertainty_m2": self.k_uncertainty_m2.tolist()
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "AcousticProfile":
        """Loads a dual-channel calibration profile from JSON."""
        in_path = Path(filepath).resolve()
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("system_metadata", {})
        if "calibrated_toa_offset_m1_ms" in data:
            meta["calibrated_toa_offset_m1_ms"] = float(data["calibrated_toa_offset_m1_ms"])
        if "calibrated_toa_offset_m2_ms" in data:
            meta["calibrated_toa_offset_m2_ms"] = float(data["calibrated_toa_offset_m2_ms"])

        freqs = data.get("frequencies_hz", [data.get("f_res_hz", KinematicAnalytics.DEFAULT_F0_HZ)])
        k1 = data.get("k_values", [data.get("k_direct", 0.0196)])
        k2 = data.get("k_values_m2", k1)
        ke1 = data.get("k_energy_m1", None)
        ke2 = data.get("k_energy_m2", None)

        return cls(
            frequencies_hz=freqs,
            k_values=k1,
            k_values_m2=k2,
            k_energy_m1=ke1,
            k_energy_m2=ke2,
            r_squared=data.get("r_squared"),
            k_uncertainty=data.get("k_uncertainty"),
            k_uncertainty_m2=data.get("k_uncertainty_m2"),
            operational_bounds=data.get("operational_bounds", {}),
            system_metadata=meta,
            name=data.get("name", in_path.stem),
            description=data.get("description", "")
        )


# =============================================================================
# 10. Real-Time Distance Inversion Engine (Amplitude & Energy)
# =============================================================================

class DistanceEstimator:
    """
    Runtime Metric Distance Inversion Engine.
    Computes real-time physical distance via:
      1. Amplitude Inversion: r(t) = k_A(f0) / A_true(t)
      2. Energy Inversion:    r(t) = sqrt(k_E(f0) / E_direct(t))
    Supports dual-channel routing and dynamic uncertainty propagation.
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
            self.profile = AcousticProfile.from_constant(0.0196, name="DefaultBaseline_2660Hz")

        self.noise_gate_v = float(noise_gate_v)
        self.voltage_uncertainty_v = float(voltage_uncertainty_v)
        self.min_dist = float(min_distance_m)
        self.max_dist = float(max_distance_m)

    def estimate_distance(
        self,
        amplitude_v: float,
        frequency_hz: float,
        delta_a: Optional[float] = None,
        channel: int = 1
    ) -> Tuple[float, float, str]:
        """
        Calculates physical distance r(t), uncertainty delta_r(t), and operational status:
          r(t) = k_A(f0) / A(t)
          delta_r(t) = r * sqrt( (delta_k / k)^2 + (delta_A / A)^2 )
        """
        amp = float(amplitude_v)
        f0 = float(frequency_hz)

        if not np.isfinite(amp) or amp < self.noise_gate_v or not np.isfinite(f0) or f0 <= 0:
            return np.nan, np.nan, "SILENCE"

        k_val, delta_k = self.profile.evaluate(f0, channel=channel)
        bounds = self.profile.get_operational_bounds(f0)
        da = float(delta_a) if delta_a is not None else self.voltage_uncertainty_v

        r_calc = k_val / max(amp, 1e-6)
        r_clamped = float(np.clip(r_calc, self.min_dist, self.max_dist))

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

    def estimate_distance_energy(
        self,
        energy_direct: float,
        frequency_hz: float,
        channel: int = 1
    ) -> Tuple[float, float, str]:
        """
        Calculates physical distance r(t) directly from direct line-of-sight energy:
          E(r) = k_E / r²  ==>  r = sqrt(k_E / E)
          delta_r = 0.5 * r * sqrt( (delta_k_E / k_E)^2 + (delta_E / E)^2 )
        """
        e = float(energy_direct)
        f0 = float(frequency_hz)

        if not np.isfinite(e) or e <= 1e-6 or not np.isfinite(f0) or f0 <= 0:
            return np.nan, np.nan, "SILENCE"

        k_e, delta_ke = self.profile.evaluate_energy(f0, channel=channel)

        r_calc = float(np.sqrt(max(0.0, k_e) / e))
        r_clamped = float(np.clip(r_calc, self.min_dist, self.max_dist))

        rel_ke_err = delta_ke / max(k_e, 1e-6)
        rel_e_err = 0.02  # ~2% integration uncertainty
        delta_r = float(0.5 * r_clamped * np.sqrt(rel_ke_err ** 2 + rel_e_err ** 2))

        status = "ACTIVE_VALID" if (self.min_dist <= r_calc <= self.max_dist) else "OUT_OF_BOUNDS"
        return r_clamped, delta_r, status

    def process_quadruple(self, quadruple: Dict[str, Any], channel: int = 1) -> Dict[str, Any]:
        """Augments an incoming quadruple dict with real-time distance metrics."""
        res = quadruple.copy()
        amp = res.get("amplitude_v", 0.0)
        f0 = res.get("frequency_hz", np.nan)

        r_m, r_err, status = self.estimate_distance(amp, f0, channel=channel)
        k_val, delta_k = self.profile.evaluate(f0 if np.isfinite(f0) else KinematicAnalytics.DEFAULT_F0_HZ, channel=channel)

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
        """Processes raw spectral frame, performs hybrid demodulation, and computes distance."""
        channel_num = 2 if ("A1" in source.upper() or "CH2" in source.upper()) else 1

        if "quadruple" in frame_dict:
            quad = frame_dict["quadruple"].copy()
        else:
            time_sig = frame_dict["v_a0"] if channel_num == 1 else frame_dict["v_a1"]
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

        return self.process_quadruple(quad, channel=channel_num) 

# =============================================================================
# 11. Standalone Acoustic Calibration Protocols (WLS, Centroid & Direct Pulse)
# =============================================================================

class AcousticCalibrationProtocol:
    """
    Continuous-Wave Acoustic Calibration Protocol.
    Ingests multi-sample (N=30) observations across an acoustic grid, applies
    dynamic boundary pruning to isolate the valid 1/r region, solves Weighted Least
    Squares (WLS) regressions, validates R² >= 0.95 linearity gates, and exports
    complete AcousticProfile artifacts.
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
        self._measurements: Dict[float, Dict[float, List[float]]] = {}
        self._fit_results: Dict[float, Dict[str, Any]] = {}

    def add_measurement(
        self,
        distance_m: float,
        frequency_hz: float,
        amplitude_v: Union[float, List[float], np.ndarray]
    ):
        """Records multi-sample observations for a given distance and frequency station."""
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
        by jointly maximizing R² and adherence to the theoretical power-law slope d(ln V)/d(ln r) = -1.0.
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

                x_sub = 1.0 / r_sub
                y_sub = v_sub

                slope, intercept = np.polyfit(x_sub, y_sub, 1)
                y_pred = slope * x_sub + intercept
                ss_res = np.sum((y_sub - y_pred) ** 2)
                ss_tot = np.sum((y_sub - np.mean(y_sub)) ** 2)

                if ss_tot > 1e-9 and slope > 1e-4:
                    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))

                    log_r = np.log(r_sub)
                    log_v = np.log(np.maximum(v_sub, 1e-6))
                    log_slope, _ = np.polyfit(log_r, log_v, 1)

                    penalty = abs(log_slope - (-1.0))
                    if penalty < 0.35 and r2 >= self.r2_threshold:
                        score = r2 * (1.0 - penalty) * (len(r_sub) ** 0.3)
                        if score > best_score:
                            best_score = score
                            best_start, best_stop = start, stop

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
        """Solves Weighted Least Squares (WLS) regressions: V(r_i) = k(f) * (1 / r_i) + c_room."""
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

            i_start, i_stop = self._prune_linear_window(r_sorted, v_means)

            r_pruned = r_sorted[i_start:i_stop]
            v_pruned = v_means[i_start:i_stop]
            sem_pruned = v_sems[i_start:i_stop]

            x_wls = 1.0 / r_pruned
            y_wls = v_pruned
            w_wls = 1.0 / np.maximum(sem_pruned ** 2, 1e-10)

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
        """Constructs an AcousticProfile with certified operating bounds."""
        if not self._fit_results:
            self.fit()

        freqs, k_vals, r2_vals, k_errs = [], [], [], []
        bounds_dict = {}

        for f in sorted(self._fit_results.keys()):
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
            raise ValueError(f"No calibration points passed R² >= {self.r2_threshold} gate.")

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
        """Fits, exports, and saves calibration profile to JSON."""
        profile = self.export_profile(name=name, description=description)
        out_path = Path(filepath).resolve()
        profile.to_json(out_path)
        return out_path

    def clear(self):
        """Clears all raw measurements and fit results."""
        self._measurements.clear()
        self._fit_results.clear()


class MultipathCalibrationProtocol:
    """
    Multipath & Reflection-Tolerant Calibration Protocol.
    Fits the spatial centroid of interference fringes across inverse-distance space:
      V_RMS(r) = k * (1 / r) + c_room + Ripple(r)
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

        self._measurements: Dict[float, Dict[float, List[float]]] = {}
        self._fit_results: Dict[float, Dict[str, Any]] = {}

    def add_measurement(
        self,
        distance_m: float,
        frequency_hz: float,
        amplitude_v: Union[float, List[float], np.ndarray]
    ):
        """Records burst observations for a distance station."""
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

    def fit(self) -> Dict[float, Dict[str, Any]]:
        """Solves spatial centroid WLS regression across inverse-distance space."""
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

            x_wls = 1.0 / r_sorted
            y_wls = v_means
            w_wls = 1.0 / np.maximum(v_sems ** 2, 1e-10)

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

            n_pts = len(r_sorted)
            if ss_tot < 1e-9 or slope <= 1e-4:
                r2 = 0.0
                passed_gate = False
                slope_err = float(0.05 * abs(slope))
            else:
                r2 = float(np.clip(1.0 - (ss_res / (ss_tot + 1e-12)), 0.0, 1.0))
                passed_gate = bool(r2 >= self.r2_threshold and slope > 1e-4)
                s_sq = ss_res / max(n_pts - 2, 1)
                slope_err = float(np.sqrt(s_sq / max(s_xx, 1e-12)))

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
                        "is_pruned_in": True
                    }
                    for idx in range(len(r_sorted))
                ]
            }

        return self._fit_results

    def export_profile(
        self,
        name: str = "Multipath_Room_Profile",
        description: str = "Acoustic profile calibrated via spatial centroid WLS",
        only_passed: bool = True
    ) -> AcousticProfile:
        """Constructs an AcousticProfile from spatial centroid fit."""
        if not self._fit_results:
            self.fit()

        freqs, k_vals, r2_vals, k_errs = [], [], [], []
        bounds_dict = {}

        for f in sorted(self._fit_results.keys()):
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
        description: str = "Acoustic profile calibrated via spatial centroid WLS"
    ) -> Path:
        profile = self.export_profile(name=name, description=description)
        out_path = Path(filepath).resolve()
        profile.to_json(out_path)
        return out_path

    def clear(self):
        self._measurements.clear()
        self._fit_results.clear()


# =============================================================================
# 12. Quasi-Anechoic Direct-Pulse Calibration Protocol (Dual Mic: 1/r & 1/r²)
# =============================================================================

class DirectPulseCalibrationProtocol:
    """
    Quasi-Anechoic Direct-Pulse Calibration Protocol.
    Calibrates both Mic 1 (A0) and Mic 2 (A1) in parallel by fitting:
      1. Amplitude decay law: A_direct(r) = k_A * (1 / r)   [Zero-Intercept WLS]
      2. Energy decay law:    E_direct(r) = k_E * (1 / r²)  [Zero-Intercept WLS]

    Because room reflections are locked out by the hardware direct-path gate,
    the steady-state room reverberation floor vanishes identically (c_room = 0).
    """

    def __init__(
        self,
        nominal_f0_hz: float = KinematicAnalytics.DEFAULT_F0_HZ,
        r2_threshold: float = 0.98,
        temperature_c: float = 20.0,
        system_metadata: Optional[Dict[str, Any]] = None
    ):
        self.nominal_f0_hz = float(nominal_f0_hz)
        self.r2_threshold = float(r2_threshold)
        self.temperature_c = float(temperature_c)
        self.c_sound = KinematicAnalytics.speed_of_sound(self.temperature_c)

        self.system_metadata = system_metadata or {
            "emitter_device": "Active_Buzzer_2660Hz",
            "mic_gain": "+35dB",
            "environment_label": "reflective_indoor_bench",
            "temperature_c": self.temperature_c
        }
        self.system_metadata["calibration_regime"] = "quasi_anechoic_direct_pulse"

        # Data store: {channel: {frequency_hz: {distance_m: {"amp": [], "energy": []}}}}
        self._measurements: Dict[int, Dict[float, Dict[float, Dict[str, List[float]]]]] = {
            1: {},
            2: {}
        }
        self._fit_results: Dict[int, Dict[float, Dict[str, Any]]] = {
            1: {},
            2: {}
        }

    def add_measurement(
        self,
        distance_m: float,
        amplitude_v: Union[float, List[float], np.ndarray],
        frequency_hz: Optional[float] = None,
        energy_direct: Optional[Union[float, List[float], np.ndarray]] = None,
        channel: int = 1
    ):
        """Records direct-path pulse observations for a specific microphone channel."""
        r = float(distance_m)
        f = float(frequency_hz) if frequency_hz is not None else self.nominal_f0_hz
        ch = 2 if channel == 2 else 1

        if r <= 0 or not np.isfinite(r) or not np.isfinite(f):
            return

        if f not in self._measurements[ch]:
            self._measurements[ch][f] = {}
        if r not in self._measurements[ch][f]:
            self._measurements[ch][f][r] = {"amp": [], "energy": []}

        # Ingest amplitudes
        if isinstance(amplitude_v, (list, tuple, np.ndarray)):
            for v in amplitude_v:
                v_flt = float(v)
                if np.isfinite(v_flt) and v_flt > 0:
                    self._measurements[ch][f][r]["amp"].append(v_flt)
        else:
            v_flt = float(amplitude_v)
            if np.isfinite(v_flt) and v_flt > 0:
                self._measurements[ch][f][r]["amp"].append(v_flt)

        # Ingest energies if provided
        if energy_direct is not None:
            if isinstance(energy_direct, (list, tuple, np.ndarray)):
                for e in energy_direct:
                    e_flt = float(e)
                    if np.isfinite(e_flt) and e_flt > 0:
                        self._measurements[ch][f][r]["energy"].append(e_flt)
            else:
                e_flt = float(energy_direct)
                if np.isfinite(e_flt) and e_flt > 0:
                    self._measurements[ch][f][r]["energy"].append(e_flt)

    def add_dual_measurement(
        self,
        distance_m: float,
        amplitude_m1_v: Union[float, List[float], np.ndarray],
        amplitude_m2_v: Union[float, List[float], np.ndarray],
        energy_m1: Optional[Union[float, List[float], np.ndarray]] = None,
        energy_m2: Optional[Union[float, List[float], np.ndarray]] = None,
        frequency_hz: Optional[float] = None
    ):
        """Convenience method to record synchronous observations for both Mic 1 and Mic 2."""
        self.add_measurement(distance_m, amplitude_m1_v, frequency_hz=frequency_hz, energy_direct=energy_m1, channel=1)
        self.add_measurement(distance_m, amplitude_m2_v, frequency_hz=frequency_hz, energy_direct=energy_m2, channel=2)

    def fit_channel(self, channel: int = 1) -> Dict[float, Dict[str, Any]]:
        """
        Solves zero-intercept Weighted Least Squares (WLS) regressions for a channel:
          • Amplitude: A_direct(r) = k_A * (1 / r)
          • Energy:    E_direct(r) = k_E * (1 / r²)
        """
        ch = 2 if channel == 2 else 1
        results = {}

        for f, dist_dict in self._measurements[ch].items():
            if len(dist_dict) < 2:
                continue

            r_sorted = np.array(sorted(dist_dict.keys()), dtype=np.float64)
            a_means, a_sems, a_stds = [], [], []
            e_means, e_sems = [], []

            for r in r_sorted:
                amps = np.array(dist_dict[r]["amp"], dtype=np.float64)
                n_a = len(amps)
                m_a = float(np.mean(amps)) if n_a > 0 else 0.0
                s_a = float(np.std(amps, ddof=1)) if n_a > 1 else max(0.001 * m_a, 1e-5)
                sem_a = float(s_a / np.sqrt(n_a)) if n_a > 0 else s_a

                a_means.append(m_a)
                a_stds.append(s_a)
                a_sems.append(sem_a)

                energies = np.array(dist_dict[r]["energy"], dtype=np.float64)
                if len(energies) > 0:
                    m_e = float(np.mean(energies))
                    s_e = float(np.std(energies, ddof=1)) if len(energies) > 1 else max(0.01 * m_e, 1e-5)
                    sem_e = float(s_e / np.sqrt(len(energies)))
                else:
                    # Estimate energy from amplitude if not recorded explicitly
                    adc_scale = 3.3 / 4095.0
                    m_e = (m_a / adc_scale) ** 2
                    sem_e = max(0.02 * m_e, 1e-5)

                e_means.append(m_e)
                e_sems.append(sem_e)

            a_means = np.array(a_means, dtype=np.float64)
            a_sems = np.array(a_sems, dtype=np.float64)
            e_means = np.array(e_means, dtype=np.float64)
            e_sems = np.array(e_sems, dtype=np.float64)

            # -------------------------------------------------------------
            # 1. Zero-Intercept Amplitude WLS: A(r) = k_A * (1/r)
            # -------------------------------------------------------------
            x_inv = 1.0 / r_sorted
            w_a = 1.0 / np.maximum(a_sems ** 2, 1e-10)

            sum_wxy_a = float(np.sum(w_a * x_inv * a_means))
            sum_wxx_a = float(np.sum(w_a * (x_inv ** 2)))
            k_direct = sum_wxy_a / sum_wxx_a if sum_wxx_a > 1e-12 else 0.0

            pred_a = k_direct * x_inv
            ss_res_a = float(np.sum(w_a * ((a_means - pred_a) ** 2)))
            ss_tot_a = float(np.sum(w_a * ((a_means - np.mean(a_means)) ** 2)))
            r2_a = float(np.clip(1.0 - (ss_res_a / max(ss_tot_a, 1e-12)), 0.0, 1.0))

            n_pts = len(r_sorted)
            s_sq_a = ss_res_a / max(n_pts - 1, 1)
            k_err_a = float(np.sqrt(s_sq_a / max(sum_wxx_a, 1e-12)))

            # -------------------------------------------------------------
            # 2. Zero-Intercept Energy WLS: E(r) = k_E * (1/r²)
            # -------------------------------------------------------------
            x_inv2 = 1.0 / (r_sorted ** 2)
            w_e = 1.0 / np.maximum(e_sems ** 2, 1e-10)

            sum_wxy_e = float(np.sum(w_e * x_inv2 * e_means))
            sum_wxx_e = float(np.sum(w_e * (x_inv2 ** 2)))
            k_energy = sum_wxy_e / sum_wxx_e if sum_wxx_e > 1e-12 else 0.0

            pred_e = k_energy * x_inv2
            ss_res_e = float(np.sum(w_e * ((e_means - pred_e) ** 2)))
            ss_tot_e = float(np.sum(w_e * ((e_means - np.mean(e_means)) ** 2)))
            r2_e = float(np.clip(1.0 - (ss_res_e / max(ss_tot_e, 1e-12)), 0.0, 1.0))

            s_sq_e = ss_res_e / max(n_pts - 1, 1)
            k_err_e = float(np.sqrt(s_sq_e / max(sum_wxx_e, 1e-12)))

            # Diagnostic unconstrained room floor check
            w_sum_a = float(np.sum(w_a))
            x_bar_a = float(np.sum(w_a * x_inv) / w_sum_a)
            y_bar_a = float(np.sum(w_a * a_means) / w_sum_a)
            s_xx_uncon = float(np.sum(w_a * ((x_inv - x_bar_a) ** 2)))
            s_xy_uncon = float(np.sum(w_a * (x_inv - x_bar_a) * (a_means - y_bar_a)))
            slope_uncon = s_xy_uncon / s_xx_uncon if s_xx_uncon > 1e-12 else 0.0
            c_room_uncon = y_bar_a - slope_uncon * x_bar_a

            passed_gate = bool(r2_a >= self.r2_threshold and k_direct > 1e-4)

            results[f] = {
                "k": float(k_direct),
                "delta_k": float(k_err_a),
                "k_energy": float(k_energy),
                "delta_k_energy": float(k_err_e),
                "c_room": 0.0,
                "c_room_unconstrained_v": float(c_room_uncon),
                "r_squared": float(r2_a),
                "r_squared_energy": float(r2_e),
                "passed_gate": passed_gate,
                "n_points": n_pts,
                "r_valid_min_m": float(np.min(r_sorted)),
                "r_valid_max_m": float(np.max(r_sorted)),
                "v_sat_v": float(np.max(a_means)),
                "v_min_v": float(np.min(a_means)),
                "raw_stations": [
                    {
                        "r_m": float(r_sorted[i]),
                        "mean_v": float(a_means[i]),
                        "std_v": float(a_stds[i]),
                        "sem_v": float(a_sems[i]),
                        "mean_energy": float(e_means[i]),
                        "sem_energy": float(e_sems[i]),
                        "model_pred_v": float(pred_a[i]),
                        "model_pred_energy": float(pred_e[i])
                    }
                    for i in range(n_pts)
                ]
            }

        self._fit_results[ch] = results
        return results

    def fit(self) -> Dict[str, Dict[float, Dict[str, Any]]]:
        """Solves dual-channel regressions for both Mic 1 and Mic 2."""
        res_m1 = self.fit_channel(channel=1)
        res_m2 = self.fit_channel(channel=2)
        return {
            "mic1": res_m1,
            "mic2": res_m2,
            self.nominal_f0_hz: res_m1.get(self.nominal_f0_hz, {})
        }

    def export_profile(
        self,
        name: str = "Quasi_Anechoic_Pulse_Profile",
        description: str = "Dual-microphone quasi-anechoic profile (1/r and 1/r² models)",
        only_passed: bool = True
    ) -> AcousticProfile:
        """Constructs an AcousticProfile containing calibrated parameters for both microphones."""
        if not self._fit_results[1]:
            self.fit()

        freqs = sorted(self._fit_results[1].keys())
        k_m1, k_m2 = [], []
        ke_m1, ke_m2 = [], []
        r2_vals, err_m1, err_m2 = [], [], []
        bounds_dict = {}

        for f in freqs:
            r1 = self._fit_results[1][f]
            r2 = self._fit_results[2].get(f, r1)

            if only_passed and not r1["passed_gate"]:
                continue

            k_m1.append(r1["k"])
            k_m2.append(r2["k"])
            ke_m1.append(r1["k_energy"])
            ke_m2.append(r2["k_energy"])
            r2_vals.append(r1["r_squared"])
            err_m1.append(r1["delta_k"])
            err_m2.append(r2["delta_k"])

            bounds_dict[str(f)] = {
                "r_min_m": r1["r_valid_min_m"],
                "r_max_m": r1["r_valid_max_m"],
                "v_sat_v": r1["v_sat_v"],
                "v_min_v": r1["v_min_v"],
                "c_room_v": 0.0
            }

        if len(freqs) == 0:
            raise ValueError(f"No calibration points passed R² >= {self.r2_threshold} gate.")

        return AcousticProfile(
            frequencies_hz=freqs,
            k_values=k_m1,
            k_values_m2=k_m2,
            k_energy_m1=ke_m1,
            k_energy_m2=ke_m2,
            r_squared=r2_vals,
            k_uncertainty=err_m1,
            k_uncertainty_m2=err_m2,
            operational_bounds=bounds_dict,
            system_metadata=self.system_metadata,
            name=name,
            description=description
        )

    def save_profile_json(
        self,
        filepath: Union[str, Path],
        name: str = "Quasi_Anechoic_Pulse_Profile",
        description: str = "Dual-microphone quasi-anechoic profile (1/r and 1/r² models)"
    ) -> Path:
        """Fits, exports, and saves dual-microphone profile to JSON."""
        profile = self.export_profile(name=name, description=description)
        out_path = Path(filepath).resolve()
        profile.to_json(out_path)
        return out_path

    def clear(self):
        """Clears all raw measurements and fits for both channels."""
        self._measurements[1].clear()
        self._measurements[2].clear()
        self._fit_results[1].clear()
        self._fit_results[2].clear()

# =============================================================================
# 13. Angle of Arrival (AoA) Phase Interferometry Estimator
# =============================================================================

class AngleOfArrivalEstimator:
    """
    Real-Time Dual-Channel Angle of Arrival (AoA) Interferometric Solver.
    Computes continuous incident bearing angle θ(t) from synchronous dual-microphone
    time streams using coherent single-bin Fourier projection:
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

        if target_freq_hz is not None:
            self.target_freq_hz = float(target_freq_hz)
        elif profile is not None:
            if isinstance(profile, (str, Path)):
                p_path = Path(profile).resolve()
                with open(p_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.target_freq_hz = float(data.get("f_res_hz", data.get("frequencies_hz", [KinematicAnalytics.DEFAULT_F0_HZ])[0]))
            elif isinstance(profile, AcousticProfile):
                self.target_freq_hz = float(profile.frequencies[0])
            else:
                self.target_freq_hz = KinematicAnalytics.DEFAULT_F0_HZ
        else:
            self.target_freq_hz = KinematicAnalytics.DEFAULT_F0_HZ

    def estimate_angle(
        self,
        v_a0: np.ndarray,
        v_a1: np.ndarray,
        fs: float,
        f_target: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Estimates incident bearing angle θ from synchronized ADC raw time arrays.
        """
        f0 = float(f_target) if f_target is not None else self.target_freq_hz

        phase_data = KinematicAnalytics.extract_dual_coherent_phase(
            signal_a0_v=v_a0,
            signal_a1_v=v_a1,
            fs=fs,
            target_freq_hz=f0,
            remove_dc=True
        )

        amp_min = min(phase_data["amp_a0_v"], phase_data["amp_a1_v"])

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
        """Convenience method to process spectral frame output dictionary."""
        return self.estimate_angle(
            v_a0=frame_dict["v_a0"],
            v_a1=frame_dict["v_a1"],
            fs=fs,
            f_target=f_target
        )


# =============================================================================
# 14. Dual-Ended Differential Doppler Air-Track Tracker
# =============================================================================

class DifferentialDopplerTracker:
    """
    Precision Dual-Ended Differential Doppler Kinematic Tracker for 1D Air Tracks.
    Tracks moving acoustic sources (gliders) with common-mode oscillator drift cancellation.
    Exposes single-microphone velocities (v_mic1, v_mic2) alongside drift-free v_diff,
    instantaneous acceleration, position, viscous drag γ, and bumper restitution e.
    """

    def __init__(
        self,
        nominal_f0_hz: float = KinematicAnalytics.DEFAULT_F0_HZ,
        profile: Optional[Union[AcousticProfile, str, Path]] = None,
        temperature_c: float = 20.0,
        track_length_m: float = 1.0,
        initial_position_m: Optional[float] = None,
        noise_gate_v: float = 0.010,
        tracking_half_bandwidth_hz: float = 120.0,
        velocity_deadband_mps: float = 0.003
    ):
        self.temperature_c = float(temperature_c)
        self.track_length_m = float(track_length_m)
        self.position_m = float(initial_position_m) if initial_position_m is not None else (self.track_length_m / 2.0)
        self.noise_gate_v = float(noise_gate_v)
        self.tracking_half_bw = float(tracking_half_bandwidth_hz)
        self.velocity_deadband = float(velocity_deadband_mps)

        if profile is not None:
            if isinstance(profile, (str, Path)):
                p_path = Path(profile).resolve()
                with open(p_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.nominal_f0_hz = float(data.get("f_res_hz", data.get("frequencies_hz", [KinematicAnalytics.DEFAULT_F0_HZ])[0]))
            elif isinstance(profile, AcousticProfile):
                self.nominal_f0_hz = float(profile.frequencies[0])
            else:
                self.nominal_f0_hz = float(nominal_f0_hz)
        else:
            self.nominal_f0_hz = float(nominal_f0_hz)

        self.last_velocity_mps = 0.0
        self.last_f0_common_hz = self.nominal_f0_hz

    def process_stereo_frame(
        self,
        v_a0: np.ndarray,
        v_a1: np.ndarray,
        fs: float,
        dt_sec: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes simultaneous stereo ADC frame, extracts individual carrier peaks,
        and computes both single-sensor and differential Doppler velocities.
        """
        n = min(len(v_a0), len(v_a1))
        dt = float(dt_sec) if dt_sec is not None else (float(n) / float(fs))
        c_snd = KinematicAnalytics.speed_of_sound(self.temperature_c)

        v0_ac = v_a0[:n] - np.mean(v_a0[:n])
        v1_ac = v_a1[:n] - np.mean(v_a1[:n])

        amp_0 = float(np.sqrt(np.mean(v0_ac ** 2)))
        amp_1 = float(np.sqrt(np.mean(v1_ac ** 2)))

        if max(amp_0, amp_1) < self.noise_gate_v:
            return {
                "velocity_mps": 0.0,
                "velocity_cmps": 0.0,
                "v_mic1_mps": 0.0,
                "v_mic1_cmps": 0.0,
                "v_mic2_mps": 0.0,
                "v_mic2_cmps": 0.0,
                "v_mean_single_mps": 0.0,
                "v_mean_single_cmps": 0.0,
                "acceleration_mps2": 0.0,
                "position_m": self.position_m,
                "f_mic1_hz": np.nan,
                "f_mic2_hz": np.nan,
                "f0_common_hz": self.last_f0_common_hz,
                "delta_f_hz": 0.0,
                "amp_mic1_v": amp_0,
                "amp_mic2_v": amp_1,
                "motion_state": "SILENCE",
                "status": "SILENCE"
            }

        freq_axis = np.fft.rfftfreq(n, d=1.0 / fs)
        scale_lin = 2.0 / float(n)

        f_min = max(20.0, self.last_f0_common_hz - self.tracking_half_bw)
        f_max = self.last_f0_common_hz + self.tracking_half_bw

        # Mic 1 (Left / A0) Pitch
        X0 = np.fft.rfft(v0_ac)
        mag_lin0 = np.abs(X0) * scale_lin
        f1, _ = KinematicAnalytics.track_sub_hertz_pitch(
            freq_axis, mag_lin0, min_freq_hz=f_min, max_freq_hz=f_max, interpolate=True
        )

        # Mic 2 (Right / A1) Pitch
        X1 = np.fft.rfft(v1_ac)
        mag_lin1 = np.abs(X1) * scale_lin
        f2, _ = KinematicAnalytics.track_sub_hertz_pitch(
            freq_axis, mag_lin1, min_freq_hz=f_min, max_freq_hz=f_max, interpolate=True
        )

        # 1. Differential Drift-Canceled Velocity
        v_raw, f0_est = KinematicAnalytics.calculate_differential_doppler_velocity(
            f_mic1=f1, f_mic2=f2, temperature_c=self.temperature_c
        )

        # 2. Individual Single-Microphone Velocities (Relative to nominal f0)
        # Sign convention: Glider moving Right towards Mic 2 => v > 0
        # For Mic 1 (Left): moving away => Red-shifted (f1 < f0) => v_m1 = -c * (f1 - f0)/f0
        # For Mic 2 (Right): moving toward => Blue-shifted (f2 > f0) => v_m2 = +c * (f2 - f0)/f0
        f_ref = self.nominal_f0_hz
        v_mic1 = -c_snd * ((f1 - f_ref) / f_ref) if (np.isfinite(f1) and f_ref > 0) else np.nan
        v_mic2 = +c_snd * ((f2 - f_ref) / f_ref) if (np.isfinite(f2) and f_ref > 0) else np.nan
        v_mean_single = 0.5 * (v_mic1 + v_mic2) if (np.isfinite(v_mic1) and np.isfinite(v_mic2)) else np.nan

        v_mps = 0.0 if abs(v_raw) < self.velocity_deadband else v_raw
        accel_mps2 = (v_mps - self.last_velocity_mps) / max(dt, 1e-6)

        self.position_m = float(np.clip(self.position_m + v_mps * dt, 0.0, self.track_length_m))
        self.last_velocity_mps = v_mps
        if np.isfinite(f0_est):
            self.last_f0_common_hz = f0_est

        if abs(v_mps) < self.velocity_deadband:
            motion_state = "STATIONARY"
        elif v_mps > 0:
            motion_state = "TOWARD_MIC2"
        else:
            motion_state = "TOWARD_MIC1"

        return {
            "velocity_mps": float(v_mps),
            "velocity_cmps": float(v_mps * 100.0),
            "v_mic1_mps": float(v_mic1),
            "v_mic1_cmps": float(v_mic1 * 100.0) if np.isfinite(v_mic1) else np.nan,
            "v_mic2_mps": float(v_mic2),
            "v_mic2_cmps": float(v_mic2 * 100.0) if np.isfinite(v_mic2) else np.nan,
            "v_mean_single_mps": float(v_mean_single),
            "v_mean_single_cmps": float(v_mean_single * 100.0) if np.isfinite(v_mean_single) else np.nan,
            "acceleration_mps2": float(accel_mps2),
            "position_m": float(self.position_m),
            "f_mic1_hz": float(f1),
            "f_mic2_hz": float(f2),
            "f0_common_hz": float(f0_est),
            "delta_f_hz": float(f2 - f1) if (np.isfinite(f1) and np.isfinite(f2)) else 0.0,
            "amp_mic1_v": amp_0,
            "amp_mic2_v": amp_1,
            "motion_state": motion_state,
            "status": "ACTIVE_VALID"
        }

    @classmethod
    def analyze_glider_kinematics(
        cls,
        time_sec: np.ndarray,
        velocity_mps: np.ndarray
    ) -> Dict[str, Any]:
        """Analyzes coasting deceleration (viscous drag γ) and bumper collisions (restitution e)."""
        t = np.asarray(time_sec, dtype=np.float64)
        v = np.asarray(velocity_mps, dtype=np.float64)

        valid = np.isfinite(t) & np.isfinite(v)
        t_clean = t[valid]
        v_clean = v[valid]

        if len(t_clean) < 10:
            return {"status": "INSUFFICIENT_DATA"}

        sign_changes = np.where(np.diff(np.sign(v_clean)))[0]
        restitutions = []

        for idx in sign_changes:
            if idx >= 5 and idx + 6 < len(v_clean):
                v_before = abs(v_clean[idx - 2])
                v_after = abs(v_clean[idx + 3])
                if v_before > 0.05 and v_after > 0.05:
                    restitutions.append(float(v_after / v_before))

        mean_restitution = float(np.mean(restitutions)) if restitutions else np.nan

        coast_mask = abs(v_clean) > 0.05
        for idx in sign_changes:
            bad_start = max(0, idx - 3)
            bad_end = min(len(v_clean), idx + 4)
            coast_mask[bad_start:bad_end] = False

        if np.sum(coast_mask) > 15:
            indices = np.where(coast_mask)[0]
            longest_seg = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
            seg = max(longest_seg, key=len)

            if len(seg) > 10:
                t_seg = t_clean[seg] - t_clean[seg[0]]
                log_v_seg = np.log(abs(v_clean[seg]))
                slope, intercept = np.polyfit(t_seg, log_v_seg, 1)
                gamma_drag = float(-slope)
            else:
                gamma_drag = 0.0
        else:
            gamma_drag = 0.0

        return {
            "max_forward_velocity_mps": float(np.max(v_clean)),
            "max_reverse_velocity_mps": float(np.min(v_clean)),
            "total_collisions_detected": len(restitutions),
            "mean_coefficient_of_restitution": mean_restitution,
            "viscous_drag_gamma": max(0.0, gamma_drag),
            "status": "ANALYSIS_COMPLETE"
        }

    def reset(self, initial_position_m: Optional[float] = None):
        """Resets the tracker's internal kinematic state."""
        self.position_m = float(initial_position_m) if initial_position_m is not None else (self.track_length_m / 2.0)
        self.last_velocity_mps = 0.0
        self.last_f0_common_hz = self.nominal_f0_hz


# =============================================================================
# 15. Time of Arrival (ToA) & Exact 2D Multilateration Estimator
# =============================================================================

class TimeOfArrivalEstimator:
    """
    Acoustic Time of Arrival (ToA) and Time Difference of Arrival (TDOA) Solver.
    Inverts pulse arrival times into calibrated absolute metric distances (r1, r2),
    exact near-field 2D Cartesian coordinates (x, y), range r, and bearing angle θ.
    Supports independent dual-channel standoff offsets (t_offset1, t_offset2).
    """

    def __init__(
        self,
        nominal_f0_hz: float = KinematicAnalytics.DEFAULT_F0_HZ,
        profile: Optional[Union[AcousticProfile, str, Path]] = None,
        mic_distance_m: float = 0.05,
        temperature_c: float = 20.0,
        calibrated_offset_ms: Optional[float] = None,
        calibrated_offset_m1_ms: Optional[float] = None,
        calibrated_offset_m2_ms: Optional[float] = None,
        threshold_ratio: float = 0.50,
        noise_gate_v: float = 0.010
    ):
        self.mic_distance_m = float(mic_distance_m)
        self.temperature_c = float(temperature_c)
        self.threshold_ratio = float(threshold_ratio)
        self.noise_gate_v = float(noise_gate_v)

        default_off = KinematicAnalytics.DEFAULT_TOA_OFFSET_MS

        if profile is not None:
            if isinstance(profile, (str, Path)):
                p_path = Path(profile).resolve()
                with open(p_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.f0_hz = float(data.get("f_res_hz", KinematicAnalytics.DEFAULT_F0_HZ))
                p_off1 = float(data.get("calibrated_toa_offset_m1_ms", data.get("calibrated_toa_offset_ms", default_off)))
                p_off2 = float(data.get("calibrated_toa_offset_m2_ms", p_off1))
                self.k_direct1 = float(data.get("k_values", [0.0196])[0])
                self.k_direct2 = float(data.get("k_values_m2", [self.k_direct1])[0])
            elif isinstance(profile, AcousticProfile):
                self.f0_hz = float(profile.frequencies[0]) if len(profile.frequencies) > 0 else KinematicAnalytics.DEFAULT_F0_HZ
                p_off1 = profile.calibrated_toa_offset_m1_ms
                p_off2 = profile.calibrated_toa_offset_m2_ms
                self.k_direct1 = float(profile.k_values[0])
                self.k_direct2 = float(profile.k_values_m2[0])
            else:
                self.f0_hz = float(nominal_f0_hz)
                p_off1 = default_off
                p_off2 = default_off
                self.k_direct1 = 0.0196
                self.k_direct2 = 0.0196
        else:
            self.f0_hz = float(nominal_f0_hz)
            p_off1 = default_off
            p_off2 = default_off
            self.k_direct1 = 0.0196
            self.k_direct2 = 0.0196

        # Assign calibrated offsets with specific overrides taking precedence
        if calibrated_offset_m1_ms is not None:
            self.offset_m1_ms = float(calibrated_offset_m1_ms)
        elif calibrated_offset_ms is not None:
            self.offset_m1_ms = float(calibrated_offset_ms)
        else:
            self.offset_m1_ms = p_off1

        if calibrated_offset_m2_ms is not None:
            self.offset_m2_ms = float(calibrated_offset_m2_ms)
        elif calibrated_offset_ms is not None:
            self.offset_m2_ms = float(calibrated_offset_ms)
        else:
            self.offset_m2_ms = p_off2

        # Backward-compatible scalar alias
        self.offset_ms = self.offset_m1_ms
        self.offset_m1_sec = self.offset_m1_ms / 1000.0
        self.offset_m2_sec = self.offset_m2_ms / 1000.0

    def estimate_distance_and_tdoa(
        self,
        v_a0: np.ndarray,
        v_a1: np.ndarray,
        fs: float,
        t_emission_sec: float = 0.0
    ) -> Dict[str, Any]:
        """
        Consolidated and unshadowed estimator: inverts pulse arrival times
        into radial distances (r1, r2), exact near-field (x, y) coordinates,
        center range r, path difference Δr, and bearing angles.
        """
        c = KinematicAnalytics.speed_of_sound(self.temperature_c)

        arr0 = KinematicAnalytics.detect_pulse_arrival_time(
            signal_v=v_a0,
            fs=fs,
            target_freq_hz=self.f0_hz,
            threshold_ratio=self.threshold_ratio
        )
        arr1 = KinematicAnalytics.detect_pulse_arrival_time(
            signal_v=v_a1,
            fs=fs,
            target_freq_hz=self.f0_hz,
            threshold_ratio=self.threshold_ratio
        )

        if not (arr0["is_detected"] and arr1["is_detected"]) or (
            arr0["amp_peak_v"] < self.noise_gate_v or arr1["amp_peak_v"] < self.noise_gate_v
        ):
            return {
                "distance_m": np.nan,
                "distance_cm": np.nan,
                "r1_m": np.nan,
                "r1_cm": np.nan,
                "r2_m": np.nan,
                "r2_cm": np.nan,
                "x_m": np.nan,
                "x_cm": np.nan,
                "y_m": np.nan,
                "y_cm": np.nan,
                "range_cm": np.nan,
                "theta_deg": np.nan,
                "theta_tdoa_deg": np.nan,
                "delta_t_ms": np.nan,
                "delta_t12_sec": np.nan,
                "delta_r_cm": np.nan,
                "t_flight_sec": np.nan,
                "amp_a0_v": arr0.get("amp_peak_v", 0.0),
                "amp_a1_v": arr1.get("amp_peak_v", 0.0),
                "status": "SILENCE"
            }

        t_arr0 = arr0["t_arrival_sec"]
        t_arr1 = arr1["t_arrival_sec"]

        # Dual calibrated flight times
        t_flight1 = max(0.0, (t_arr0 - float(t_emission_sec)) - self.offset_m1_sec)
        t_flight2 = max(0.0, (t_arr1 - float(t_emission_sec)) - self.offset_m2_sec)

        r1_m = float(c * t_flight1)
        r2_m = float(c * t_flight2)

        delta_t_sec = float(t_arr0 - t_arr1)
        delta_r_m = float(c * delta_t_sec)

        # Exact near-field 2D Cartesian multilateration with modulo-lambda parity
        loc2d = KinematicAnalytics.solve_2d_multilateration(
            r1_m=r1_m,
            r2_m=r2_m,
            d_m=self.mic_distance_m,
            f0=self.f0_hz,
            c_sound=c,
            wrap_modulo_lambda=True,
            delta_r_m=delta_r_m
        )

        return {
            "distance_m": loc2d["range_m"],
            "distance_cm": loc2d["range_m"] * 100.0 if np.isfinite(loc2d["range_m"]) else np.nan,
            "r1_m": r1_m,
            "r1_cm": r1_m * 100.0,
            "r2_m": r2_m,
            "r2_cm": r2_m * 100.0,
            "x_m": loc2d["x_m"],
            "x_cm": loc2d["x_m"] * 100.0 if np.isfinite(loc2d["x_m"]) else np.nan,
            "y_m": loc2d["y_m"],
            "y_cm": loc2d["y_m"] * 100.0 if np.isfinite(loc2d["y_m"]) else np.nan,
            "range_cm": loc2d["range_m"] * 100.0 if np.isfinite(loc2d["range_m"]) else np.nan,
            "theta_deg": loc2d["theta_deg"],
            "theta_tdoa_deg": loc2d["theta_far_deg"],
            "delta_t12_sec": delta_t_sec,
            "delta_t_ms": delta_t_sec * 1000.0,
            "delta_r_cm": delta_r_m * 100.0,
            "t_flight_sec": t_flight1,
            "t_arrival_a0_sec": float(t_arr0),
            "t_arrival_a1_sec": float(t_arr1),
            "amp_a0_v": arr0["amp_peak_v"],
            "amp_a1_v": arr1["amp_peak_v"],
            "status": loc2d["status"]
        }

    def process_frame(
        self,
        frame_dict: Dict[str, Any],
        fs: float = 50000.0,
        t_emission_sec: float = 0.0
    ) -> Dict[str, Any]:
        """Convenience wrapper for output from capture_spectral_frame() or capture_pulsed_toa_frame()."""
        return self.estimate_distance_and_tdoa(
            v_a0=frame_dict["v_a0"],
            v_a1=frame_dict["v_a1"],
            fs=fs,
            t_emission_sec=t_emission_sec
        )
