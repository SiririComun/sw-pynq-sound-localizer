"""
pynq_localizer.kinematics: High-Precision Acoustic Kinematics & Spectral Physics Analytics.
Provides angular frequency parameter mapping (omega +- Delta omega), physical inverse-distance
acoustic law regression (1/r and 1/r^2), sub-Hertz pitch tracking, and multi-source filter metrics.
"""

from typing import Tuple, Optional, Union, Dict, List
import numpy as np


class KinematicAnalytics:
    """
    High-performance DSP and physical acoustics analytics engine.
    """

    # =========================================================================
    # 1. Physics Models & Temperature Compensation
    # =========================================================================

    @staticmethod
    def speed_of_sound(temperature_c: float = 20.0) -> float:
        """
        Calculates temperature-compensated speed of sound in air:
        c(T) = 331.3 * sqrt(1 + T_c / 273.15) [m/s]
        :param temperature_c: Ambient temperature in Celsius (default: 20.0°C).
        :return: Sound velocity c in m/s (e.g. 343.2 m/s at 20°C).
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
    # 2. Spectral Mask Parameter Mapping: (omega +- Delta omega) -> (k_start, k_stop)
    # =========================================================================

    @staticmethod
    def calculate_filter_bins(
        omega_0: Optional[float] = None,
        delta_omega: Optional[float] = None,
        f_center_hz: Optional[float] = None,
        delta_f_hz: Optional[float] = None,
        fs: float = 50_000.0,
        fft_len: int = 1024
    ) -> Dict[str, Union[int, float]]:
        """
        Translates continuous frequency parameters (in rad/s or Hz) to discrete hardware FFT bins.

        :param omega_0: Center angular frequency in rad/s (optional if f_center_hz is provided).
        :param delta_omega: Half-bandwidth in rad/s (optional if delta_f_hz is provided).
        :param f_center_hz: Center frequency in Hz.
        :param delta_f_hz: Half-bandwidth in Hz.
        :param fs: Sampling rate in Hz (default: 50,000 Hz).
        :param fft_len: FFT transform length N (default: 1024).
        :return: Dictionary containing discrete bins (k_start, k_stop) and physical frequency bounds.
        """
        if omega_0 is not None:
            f0 = float(omega_0) / (2.0 * np.pi)
        elif f_center_hz is not None:
            f0 = float(f_center_hz)
        else:
            raise ValueError("Must provide either omega_0 (rad/s) or f_center_hz (Hz).")

        if delta_omega is not None:
            df_half = float(delta_omega) / (2.0 * np.pi)
        elif delta_f_hz is not None:
            df_half = float(delta_f_hz)
        else:
            df_half = 50.0  # Default 50 Hz half-band

        f_start = max(0.0, f0 - df_half)
        f_stop = min(fs / 2.0, f0 + df_half)

        bin_res = fs / float(fft_len)
        k_start = int(round(f_start / bin_res))
        k_stop = int(round(f_stop / bin_res))

        k_max = fft_len // 2
        k_start = max(0, min(k_max, k_start))
        k_stop = max(0, min(k_max, k_stop))

        if k_start == k_stop:
            k_stop = min(k_max, k_start + 1)

        return {
            "k_start": k_start,
            "k_stop": k_stop,
            "f_center_hz": f0,
            "delta_f_hz": df_half,
            "omega_0_rad_s": f0 * 2.0 * np.pi,
            "delta_omega_rad_s": df_half * 2.0 * np.pi,
            "f_start_hz": k_start * bin_res,
            "f_stop_hz": k_stop * bin_res,
            "bin_resolution_hz": bin_res,
            "fft_len": fft_len,
            "fs_hz": fs
        }

    # =========================================================================
    # 3. Acoustic Distance Law Regression (1/r and 1/r^2)
    # =========================================================================

    @classmethod
    def fit_inverse_distance_law(
        cls,
        distances_m: Union[List[float], np.ndarray],
        voltages_rms: Union[List[float], np.ndarray]
    ) -> Dict[str, float]:
        """
        Validates the physical spherical spreading law:
          • Acoustic Pressure Voltage: V_RMS(r) = A * r^(-n)  (Ideal n = 1.0)
          • Acoustic Intensity:        I(r) ~ V_RMS^2 ~ r^(-2n) (Ideal 2n = 2.0)

        Performs log-log linear regression: ln(V_RMS) = -n * ln(r) + ln(A)

        :param distances_m: Array of calibrated measurement distances in meters (r > 0).
        :param voltages_rms: Measured RMS voltages at each distance.
        :return: Dictionary containing exponent n, power exponent 2n, R^2 fit, and percentage error.
        """
        r = np.asarray(distances_m, dtype=np.float64)
        v = np.asarray(voltages_rms, dtype=np.float64)

        valid = (r > 0) & (v > 0) & np.isfinite(r) & np.isfinite(v)
        r_clean = r[valid]
        v_clean = v[valid]

        if len(r_clean) < 3:
            raise ValueError("At least 3 valid distance points are required to fit the inverse-distance law.")

        ln_r = np.log(r_clean)
        ln_v = np.log(v_clean)

        # 1st-degree polynomial: ln(v) = slope * ln(r) + intercept
        slope, intercept = np.polyfit(ln_r, ln_v, 1)

        ln_v_pred = slope * ln_r + intercept
        ss_res = np.sum((ln_v - ln_v_pred) ** 2)
        ss_tot = np.sum((ln_v - np.mean(ln_v)) ** 2)
        r_squared = 1.0 - (ss_res / (ss_tot + 1e-12))

        # In physics: V ~ r^(-n), so slope = -n => n = -slope
        measured_n = -slope
        scale_a = float(np.exp(intercept))

        return {
            "measured_exponent_n": float(measured_n),
            "intensity_exponent_2n": float(measured_n * 2.0),
            "amplitude_coefficient_A": scale_a,
            "r_squared": float(r_squared),
            "ideal_pressure_exponent": 1.0,
            "ideal_intensity_exponent": 2.0,
            "error_pct_from_ideal_1_over_r": float(abs(measured_n - 1.0) * 100.0)
        }

    # =========================================================================
    # 4. Multi-Source Isolation & Filter Rejection Metrics
    # =========================================================================

    @classmethod
    def calculate_filter_isolation_metrics(
        cls,
        raw_signal: np.ndarray,
        filtered_signal: np.ndarray,
        fs: float,
        target_band_hz: Tuple[float, float],
        interferer_band_hz: Tuple[float, float]
    ) -> Dict[str, float]:
        """
        Quantifies the isolation and stopband rejection of the hardware filter when
        two distinct sources emit simultaneously.

        :param raw_signal: 1D time-domain raw audio array before filtering.
        :param filtered_signal: 1D time-domain audio array after hardware spectral mask + IFFT.
        :param fs: Sampling rate in Hz.
        :param target_band_hz: (f_min, f_max) tuple for the desired source (e.g. (950, 1050)).
        :param interferer_band_hz: (f_min, f_max) tuple for the interferer (e.g. (2400, 2600)).
        :return: Rejection ratio, passband insertion loss, and Signal-to-Interference Ratio (SIR).
        """
        # Compute FFT power spectrum of raw and filtered signals
        n = min(len(raw_signal), len(filtered_signal))
        w = np.hanning(n)

        fft_raw = np.abs(np.fft.rfft((raw_signal[:n] - np.mean(raw_signal[:n])) * w)) ** 2
        fft_filt = np.abs(np.fft.rfft((filtered_signal[:n] - np.mean(filtered_signal[:n])) * w)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        mask_target = (freqs >= target_band_hz[0]) & (freqs <= target_band_hz[1])
        mask_interferer = (freqs >= interferer_band_hz[0]) & (freqs <= interferer_band_hz[1])

        # Energies
        p_target_raw = np.sum(fft_raw[mask_target])
        p_target_filt = np.sum(fft_filt[mask_target])
        p_interf_raw = np.sum(fft_raw[mask_interferer])
        p_interf_filt = np.sum(fft_filt[mask_interferer])

        # Stopband rejection of interferer
        stopband_rejection_db = 10.0 * np.log10(max(p_interf_raw, 1e-12) / max(p_interf_filt, 1e-12))

        # Passband transmission fidelity (Target = 0.0 dB)
        insertion_loss_db = 10.0 * np.log10(max(p_target_filt, 1e-12) / max(p_target_raw, 1e-12))

        # Signal to Interference Ratio (SIR) after filtering
        sir_after_db = 10.0 * np.log10(max(p_target_filt, 1e-12) / max(p_interf_filt, 1e-12))

        return {
            "stopband_rejection_db": float(stopband_rejection_db),
            "insertion_loss_db": float(insertion_loss_db),
            "sir_after_filtering_db": float(sir_after_db),
            "amplitude_preservation_ratio": float(np.sqrt(p_target_filt / max(p_target_raw, 1e-12)))
        }

    # =========================================================================
    # 5. Amplitude Envelope & STFT Tracking
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
        """Computes analytic signal z[n] = x[n] + j*H{x[n]} using NumPy FFT."""
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

    @staticmethod
    def track_sub_hertz_pitch(
        freqs: np.ndarray,
        mags: np.ndarray,
        min_freq_hz: float = 20.0,
        max_freq_hz: float = 20000.0,
        interpolate: bool = True
    ) -> Tuple[float, float]:
        """Extracts dominant fundamental frequency f0 with sub-Hertz accuracy."""
        valid_mask = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            k = int(np.argmax(mags))
            return float(freqs[k]), float(mags[k])

        k = int(valid_indices[np.argmax(mags[valid_indices])])

        if not interpolate or k <= 0 or k >= len(mags) - 1:
            return float(freqs[k]), float(mags[k])

        alpha = float(mags[k - 1])
        beta  = float(mags[k])
        gamma = float(mags[k + 1])

        denom = alpha - 2.0 * beta + gamma
        if abs(denom) < 1e-12:
            return float(freqs[k]), float(beta)

        delta = 0.5 * (alpha - gamma) / denom
        delta = max(-0.5, min(0.5, delta))

        delta_f = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        interp_freq = float(freqs[k] + delta * delta_f)
        interp_mag = float(beta - 0.25 * (alpha - gamma) * delta)

        return interp_freq, interp_mag

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
        """Extracts synchronized Amplitude A(t) and Pitch f0(t) trajectories."""
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
                mag_db = 20.0 * np.log10(np.maximum(linear_v, 1e-6))

                f0, _ = cls.track_sub_hertz_pitch(
                    freq_axis,
                    mag_db,
                    min_freq_hz=min_freq_hz,
                    max_freq_hz=max_freq_hz,
                    interpolate=True
                )
                freq_traj[i] = f0

        return times_sec, amp_traj, freq_traj