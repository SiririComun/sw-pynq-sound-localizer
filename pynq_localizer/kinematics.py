"""
pynq_localizer.kinematics: High-Precision Acoustic Kinematics & Frequency Ridge Tracking Engine.
Provides sub-Hertz pitch tracking (20 Hz - 20 kHz), spectral quadruple extraction (f, A, φ, t),
multi-source tracking, single-channel distance estimation r = k(f)/A, acoustic profile modeling,
phase velocity verification, smoothed RMS envelope extraction, sliding STFT frequency trajectories,
temperature-compensated sound speed, Doppler velocity, and gravity metrics.
"""

import json
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, List, Any, Callable
import numpy as np
from scipy.interpolate import interp1d

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
        :param temperature_c: Ambient air temperature in Celsius (default: 20.0°C).
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
        Positive v => Source approaching observer.
        Negative v => Source receding from observer.

        :param f_observed: Measured dominant frequency in Hz (scalar or numpy array).
        :param f_source: Rest / emitted fundamental frequency in Hz.
        :param temperature_c: Ambient temperature in Celsius.
        :return: Radial velocity in meters per second (m/s).
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

        :param time_sec: 1D time axis array in seconds during the free-fall interval.
        :param f_observed: 1D measured frequency array in Hz during free-fall.
        :param f_source: Rest fundamental frequency of the dropping speaker in Hz.
        :param temperature_c: Ambient temperature in Celsius.
        :return: Dictionary containing experimental g, slope (df/dt), R^2 coefficient, and sound speed.
        """
        t = np.asarray(time_sec, dtype=np.float64)
        f = np.asarray(f_observed, dtype=np.float64)

        # Filter out NaN/invalid values
        valid_mask = np.isfinite(t) & np.isfinite(f)
        t_clean = t[valid_mask]
        f_clean = f[valid_mask]

        if len(t_clean) < 5:
            raise ValueError("Insufficient valid data points to perform linear regression for gravity measurement.")

        # 1st-degree polynomial linear regression: f(t) = slope * t + intercept
        slope, intercept = np.polyfit(t_clean, f_clean, 1)

        # Compute R^2 goodness-of-fit
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
    # 2. Quadruple Extraction (f, A, φ, t) & Spectral Telemetry Engine
    # =========================================================================

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

        :param freq_axis: 1D positive half-spectrum frequency axis in Hz.
        :param magnitude: 1D linear magnitude spectrum |X(f)|.
        :param phase_rad: 1D phase spectrum ∠X(f) in radians [-π, +π].
        :param f_min: Lower search frequency bound in Hz.
        :param f_max: Upper search frequency bound in Hz.
        :param timer_cycles: Hardware AXI Timer clock cycles at frame capture.
        :param clock_freq_hz: AXI system clock frequency (default: 100 MHz).
        :param enbw: Equivalent Noise Bandwidth window correction factor.
        :param raw_scale_factor: ADC linear voltage scaling factor.
        :return: Structured telemetry dictionary containing:
                 - 'frequency_hz': Exact analytical sinc peak frequency (sub-Hertz).
                 - 'amplitude_v': Parseval in-band RMS voltage envelope.
                 - 'phase_rad': Dominant tone phase in radians [-π, +π].
                 - 'phase_deg': Dominant tone phase in degrees [-180°, +180°].
                 - 'timestamp_sec': Sub-microsecond hardware timestamp.
                 - 'peak_bin': Discrete integer bin index of spectral peak.
                 - 'band_energy': Integrated spectral energy in band.
                 - 'is_valid': True if tone meets validity bounds.
        """
        freqs = np.asarray(freq_axis, dtype=np.float64)
        mags = np.asarray(magnitude, dtype=np.float64)
        phases = np.asarray(phase_rad, dtype=np.float64)

        # 1. Continuous-to-Discrete Band Gating
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

        # 2. Sub-Hertz Analytical Sinc Pitch Interpolation
        f0, delta = cls.track_sub_hertz_pitch(
            freqs, mags, min_freq_hz=f_min, max_freq_hz=f_max, interpolate=True, return_delta=True
        )

        # 3. In-Band Parseval RMS Energy Integration
        n_points = len(freqs) * 2  # Full FFT size N
        band_energy = float(np.sum(band_mags ** 2))
        v_rms = (np.sqrt(2.0 * band_energy) / (n_points * enbw)) * raw_scale_factor

        # 4. Fractional-bin Phase Alignment Correction (Subtract window center offset)
        raw_phi = float(phases[k0]) if (0 <= k0 < len(phases)) else 0.0
        phi_corrected = raw_phi - (np.pi * delta * (n_points - 1.0) / n_points)
        phi_wrapped = float((phi_corrected + np.pi) % (2.0 * np.pi) - np.pi)

        # 5. Sub-Microsecond Hardware Timestamp
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
        """
        Calculates instantaneous phase frequency trajectory f_phase(t) = (1 / 2π) * (dΦ / dt)
        from unwrapped phase and computes the Signal Quality Index (SQI) relative to expected frequency.

        :param phase_history: 1D array/list of wrapped phase angles in radians.
        :param time_history: 1D array/list of timestamps in seconds.
        :param f_expected: Target fundamental frequency for SQI cross-check.
        :param f_tol: Frequency deviation tolerance for SQI scaling in Hz.
        :return: (unwrapped_phase_rad, instantaneous_frequency_hz, mean_sqi).
        """
        phi = np.asarray(phase_history, dtype=np.float64)
        t = np.asarray(time_history, dtype=np.float64)

        if len(phi) < 2 or len(t) < 2:
            return np.array([]), np.array([]), 0.0

        # Phase Unwrapping across frame boundaries
        unwrapped_phi = np.unwrap(phi)
        dt = np.diff(t)
        dphi = np.diff(unwrapped_phi)

        # Avoid divide-by-zero on identical timestamps
        valid_dt = dt > 1e-7
        f_inst = np.zeros(len(dt), dtype=np.float64)
        f_inst[valid_dt] = (dphi[valid_dt] / dt[valid_dt]) / (2.0 * np.pi)

        # Calculate Signal Quality Index (SQI)
        if f_expected is not None and len(f_inst) > 0:
            err = np.abs(f_inst - float(f_expected))
            sqi_array = np.clip(1.0 - (err / float(f_tol)), 0.0, 1.0)
            mean_sqi = float(np.mean(sqi_array))
        else:
            mean_sqi = 1.0

        return unwrapped_phi, f_inst, mean_sqi

    # =========================================================================
    # 3. Sub-Hertz Parabolic Pitch Tracking & STFT Trajectories
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

        :param freqs: 1D frequency axis array in Hz.
        :param mags: 1D magnitude array.
        :param min_freq_hz: Lower search cutoff (default: 20.0 Hz).
        :param max_freq_hz: Upper search cutoff (default: 20,000.0 Hz).
        :param interpolate: Enable analytical interpolation on discrete peak.
        :param return_delta: If True, returns (f0, delta) where delta is the fractional bin offset.
        :return: (peak_freq_hz, peak_magnitude) or (peak_freq_hz, delta).
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
        """
        Processes a single polar FFT frame, extracts raw quadruples, cancels harmonic cross-talk,
        and computes SIR isolation metrics for all sources.
        """
        t_sec = float(timer_cycles) / self.clock_freq_hz if self.clock_freq_hz > 0 else 0.0
        raw_quads = {}

        # 1. Extract raw quadruples
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

        # 2. Harmonic Cross-Talk Cancellation & SIR
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
                        # 2nd harmonic
                        if abs(f_j - 2.0 * f_i) < (self.source_bands[j_name][1] - self.source_bands[j_name][0]) * 0.5:
                            leak = p_i * self.h2_coeff
                            total_leakage += leak
                        # 3rd harmonic
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
# 5. Acoustic Calibration Profile & Injectable Distance Inversion Engine
# =========================================================================

class AcousticProfile:
    """
    Data model and interpolator for physical acoustic calibration functions k(f).
    Supports discrete calibration grids, continuous splines, custom callables, and JSON export/import.
    """

    def __init__(
        self,
        frequencies_hz: Optional[Union[List[float], np.ndarray]] = None,
        k_values: Optional[Union[List[float], np.ndarray]] = None,
        r_squared: Optional[Union[List[float], np.ndarray]] = None,
        k_uncertainty: Optional[Union[List[float], np.ndarray]] = None,
        name: str = "DefaultProfile",
        description: str = "Acoustic calibration curve k(f)"
    ):
        self.name = name
        self.description = description
        self._callable_model: Optional[Callable[[float], float]] = None

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
                self._interp_k = interp1d(
                    self.frequencies,
                    self.k_values,
                    kind="linear" if len(self.frequencies) < 4 else "cubic",
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
                self._interp_k = lambda f: self.k_values[0]
                self._interp_err = lambda f: self.k_uncertainty[0]
        else:
            self.frequencies = np.array([1000.0], dtype=np.float64)
            self.k_values = np.array([0.05], dtype=np.float64)
            self.r_squared = np.array([1.0], dtype=np.float64)
            self.k_uncertainty = np.array([0.002], dtype=np.float64)
            self._interp_k = lambda f: 0.05
            self._interp_err = lambda f: 0.002

    def evaluate(self, frequency_hz: float) -> Tuple[float, float]:
        """
        Evaluates k(f) and its uncertainty delta_k at a specific frequency.
        :param frequency_hz: Target acoustic pitch in Hz.
        :return: (k_value in V*m, delta_k in V*m).
        """
        f = float(frequency_hz)
        if not np.isfinite(f) or f <= 0:
            return float(self.k_values[0]), float(self.k_uncertainty[0])

        if self._callable_model is not None:
            k_val = float(self._callable_model(f))
            return k_val, 0.03 * k_val

        k_val = float(self._interp_k(f))
        k_err = float(self._interp_err(f))
        return k_val, k_err

    @classmethod
    def from_constant(cls, k_value: float, relative_error: float = 0.03, name: str = "ConstantProfile") -> "AcousticProfile":
        """Factory creating an AcousticProfile with a flat, constant k across all frequencies."""
        profile = cls(
            frequencies_hz=[100.0, 10000.0],
            k_values=[float(k_value), float(k_value)],
            r_squared=[1.0, 1.0],
            k_uncertainty=[float(k_value) * relative_error, float(k_value) * relative_error],
            name=name,
            description=f"Static flat profile with k={k_value:.4f} V*m"
        )
        return profile

    @classmethod
    def from_callable(cls, func: Callable[[float], float], name: str = "CallableProfile") -> "AcousticProfile":
        """Factory creating an AcousticProfile evaluated directly from a mathematical function k(f)."""
        profile = cls.from_constant(0.05, name=name)
        profile._callable_model = func
        return profile

    def to_json(self, filepath: Union[str, Path]):
        """Serializes calibration profile to a portable JSON file."""
        out_path = Path(filepath).resolve()
        data = {
            "name": self.name,
            "description": self.description,
            "frequencies_hz": self.frequencies.tolist(),
            "k_values": self.k_values.tolist(),
            "r_squared": self.r_squared.tolist(),
            "k_uncertainty": self.k_uncertainty.tolist()
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "AcousticProfile":
        """Loads a calibration profile from a JSON file."""
        in_path = Path(filepath).resolve()
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            frequencies_hz=data["frequencies_hz"],
            k_values=data["k_values"],
            r_squared=data.get("r_squared"),
            k_uncertainty=data.get("k_uncertainty"),
            name=data.get("name", in_path.stem),
            description=data.get("description", "")
        )


class DistanceEstimator:
    """
    Runtime Distance Inversion Engine.
    Computes real-time physical distance r(t) = k(f0) / A(t) with dynamic error propagation.
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
        """
        :param profile: Injected AcousticProfile instance containing k(f) calibration data.
        :param k_constant: Optional static constant k override in V*m (e.g. 0.05 V*m).
        :param k_func: Optional callable k(f) function.
        :param noise_gate_v: Minimum RMS amplitude in Volts to compute distance (squelches silence to NaN).
        :param voltage_uncertainty_v: Inherent ADC voltage noise uncertainty delta_A.
        :param min_distance_m: Minimum physical clipping boundary in meters (prevents near-field division spikes).
        :param max_distance_m: Maximum physical clipping boundary in meters.
        """
        if profile is not None:
            self.profile = profile
        elif k_func is not None:
            self.profile = AcousticProfile.from_callable(k_func)
        elif k_constant is not None:
            self.profile = AcousticProfile.from_constant(k_constant)
        else:
            # Default baseline profile (k = 0.05 V*m)
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
    ) -> Tuple[float, float]:
        """
        Calculates physical distance r(t) and uncertainty delta_r(t) from in-band amplitude and frequency.
        r(t) = k(f0) / A(t)
        delta_r(t) = r * sqrt( (delta_k / k)^2 + (delta_A / A)^2 )

        :param amplitude_v: Measured in-band RMS voltage envelope A(t).
        :param frequency_hz: Measured fundamental pitch f0(t).
        :param delta_a: Optional explicit measurement voltage uncertainty.
        :return: (distance_meters, uncertainty_meters). Returns (NaN, NaN) if below noise gate.
        """
        amp = float(amplitude_v)
        f0 = float(frequency_hz)

        # Squelch silence/noise floor
        if not np.isfinite(amp) or amp < self.noise_gate_v or not np.isfinite(f0) or f0 <= 0:
            return np.nan, np.nan

        # Evaluate k(f0) and delta_k
        k_val, delta_k = self.profile.evaluate(f0)
        da = float(delta_a) if delta_a is not None else self.voltage_uncertainty_v

        # Invert spherical decay: r = k / A
        r_calc = k_val / max(amp, 1e-6)
        r_clamped = float(np.clip(r_calc, self.min_dist, self.max_dist))

        # First-order error propagation
        rel_k_err = delta_k / max(k_val, 1e-6)
        rel_a_err = da / max(amp, 1e-6)
        delta_r = float(r_clamped * np.sqrt(rel_k_err ** 2 + rel_a_err ** 2))

        return r_clamped, delta_r

    def process_quadruple(self, quadruple: Dict[str, Any]) -> Dict[str, Any]:
        """
        Augments an incoming quadruple dict with real-time distance metrics.
        Adds 'distance_m', 'distance_err_m', and 'k_evaluated'.
        """
        res = quadruple.copy()
        amp = res.get("amplitude_v", 0.0)
        f0 = res.get("frequency_hz", np.nan)

        r_m, r_err = self.estimate_distance(amp, f0)
        k_val, delta_k = self.profile.evaluate(f0 if np.isfinite(f0) else 1000.0)

        res["distance_m"] = r_m
        res["distance_err_m"] = r_err
        res["k_evaluated"] = k_val
        res["k_uncertainty"] = delta_k
        return res