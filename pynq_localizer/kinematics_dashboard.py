"""
pynq_localizer.kinematics_dashboard: Multi-Tab Real-Time Kinematics & Distance Dashboard.
Features 3-row telemetry display (In-Band Amplitude [mV], Pitch [Hz], and Metric Distance [cm]),
2048-point STFT resolution, moving median DSP smoothing, BFP-immune coherent demodulation,
clean non-NaN CSV exports, and direct Python data handoff for curve fitting.
"""

import time
import threading
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, Union, Any
from IPython.display import display
import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from pynq import allocate
except (ImportError, ModuleNotFoundError):
    allocate = None

from pynq_localizer.kinematics import (
    KinematicAnalytics,
    AcousticProfile,
    DistanceEstimator,
)

try:
    from pynq_localizer.hw_trigger import HardwareTrigger
except (ImportError, ModuleNotFoundError):
    HardwareTrigger = None


class KinematicsDashboard:
    """
    Multi-tab real-time 10-second rolling acoustic kinematics dashboard.
    Decouples high-speed dual-channel DSP from smart active-tab Plotly rendering.
    Displays flagship 3-row telemetry curves:
      - Row 1: Physical In-Band Amplitude Envelope A_true(t) [mV]
      - Row 2: Sub-Hertz Dominant Pitch Trajectory f0(t) [Hz]
      - Row 3: Real-Time Inverted Metric Distance r(t) [cm] with ±δr confidence bands
    """

    def __init__(
        self,
        overlay=None,
        window_duration_sec: float = 10.0,
        hop_ms: float = 10.0,
        fs_per_ch: float = 50_000.0,
        profile: Optional[Union[AcousticProfile, str, Path]] = None,
        k_constant: Optional[float] = None,
        estimator: Optional[DistanceEstimator] = None,
    ):
        self.overlay = overlay
        self.window_duration_sec = float(window_duration_sec)
        self.hop_ms = float(hop_ms)
        self.fs_per_ch = float(fs_per_ch)

        # Buffer sizing: 100 points/sec -> 1000 points for 10.0s
        self.pts_per_sec = int(1000.0 / self.hop_ms)
        self.buffer_len = int(self.window_duration_sec * self.pts_per_sec)

        # Thread-safe rolling ring buffers for BOTH channels (Amplitudes stored in Volts)
        self.t_axis = np.linspace(-self.window_duration_sec, 0.0, self.buffer_len)
        self.buf_amp_a0 = np.zeros(self.buffer_len, dtype=np.float64)
        self.buf_freq_a0 = np.full(self.buffer_len, np.nan, dtype=np.float64)
        self.buf_dist_a0 = np.full(self.buffer_len, np.nan, dtype=np.float64)       # in cm
        self.buf_disterr_a0 = np.full(self.buffer_len, np.nan, dtype=np.float64)    # in cm

        self.buf_amp_a1 = np.zeros(self.buffer_len, dtype=np.float64)
        self.buf_freq_a1 = np.full(self.buffer_len, np.nan, dtype=np.float64)
        self.buf_dist_a1 = np.full(self.buffer_len, np.nan, dtype=np.float64)       # in cm
        self.buf_disterr_a1 = np.full(self.buffer_len, np.nan, dtype=np.float64)    # in cm

        self._buf_lock = threading.Lock()

        # Moving median history deques for reflection noise rejection
        self._hist_f0_a0 = deque(maxlen=5)
        self._hist_f0_a1 = deque(maxlen=5)

        # High-resolution STFT window (40.96 ms = 2048 samples at 50 kSPS, Δf = 24.41 Hz)
        self.win_len = 2048
        n = np.arange(self.win_len)
        self.stft_win = (
            0.35875
            - 0.48829 * np.cos(2.0 * np.pi * n / (self.win_len - 1))
            + 0.14128 * np.cos(4.0 * np.pi * n / (self.win_len - 1))
            - 0.01168 * np.cos(6.0 * np.pi * n / (self.win_len - 1))
        )
        self.coherent_gain = np.sum(self.stft_win) / self.win_len
        self.freq_axis = np.fft.rfftfreq(self.win_len, d=1.0 / self.fs_per_ch)

        # Distance Estimator Initialization
        if estimator is not None:
            self.estimator = estimator
        elif profile is not None:
            if isinstance(profile, (str, Path)):
                self.estimator = DistanceEstimator(profile=AcousticProfile.from_json(profile))
            elif isinstance(profile, AcousticProfile):
                self.estimator = DistanceEstimator(profile=profile)
            else:
                self.estimator = DistanceEstimator(profile=profile)
        elif k_constant is not None:
            self.estimator = DistanceEstimator(k_constant=float(k_constant))
        elif Path("calibrated_room_profile.json").exists():
            try:
                self.estimator = DistanceEstimator(
                    profile=AcousticProfile.from_json("calibrated_room_profile.json")
                )
            except Exception:
                self.estimator = DistanceEstimator(k_constant=0.050, noise_gate_v=0.003)
        else:
            self.estimator = DistanceEstimator(k_constant=0.050, noise_gate_v=0.003)

        # Threading state
        self._is_running = False
        self._dsp_thread: Optional[threading.Thread] = None
        self._render_thread: Optional[threading.Thread] = None

        # Telemetry readouts cache
        self._cur_amp_a0 = 0.0
        self._cur_f0_a0 = np.nan
        self._cur_dist_a0 = np.nan
        self._cur_disterr_a0 = np.nan

        self._cur_amp_a1 = 0.0
        self._cur_f0_a1 = np.nan
        self._cur_dist_a1 = np.nan
        self._cur_disterr_a1 = np.nan

        if self.overlay and hasattr(self.overlay, "trigger"):
            self.trigger = self.overlay.trigger
        elif self.overlay and HardwareTrigger:
            self.trigger = HardwareTrigger(self.overlay)
        else:
            self.trigger = None

        self._build_ui()
        self._build_plots()
        self._setup_callbacks()

    def _build_ui(self):
        # 1. Action Row
        self.start_btn = widgets.Button(
            description="Start Stream", button_style="success", icon="play", layout=widgets.Layout(width="120px")
        )
        self.stop_btn = widgets.Button(
            description="Stop", button_style="danger", icon="stop", layout=widgets.Layout(width="90px")
        )
        self.clear_btn = widgets.Button(
            description="Reset View", button_style="warning", icon="refresh", layout=widgets.Layout(width="105px")
        )
        self.export_btn = widgets.Button(
            description="Export CSV", button_style="info", icon="download", layout=widgets.Layout(width="115px")
        )
        self.clean_csv_chk = widgets.Checkbox(
            value=True, description="Clean NaNs", indent=False, layout=widgets.Layout(width="110px")
        )

        # 2. Live Dual-Channel Readouts
        self.readout_metrics = widgets.HTML(
            "<span style='color:#00FFCC; font-family:monospace; font-size:12px; font-weight:bold;'>"
            "A0: 0.0mV (--- → ---) | A1: 0.0mV (--- → ---) | Status: IDLE"
            "</span>"
        )

        # 3. Tuning & DSP Controls
        self.squelch_slider = widgets.FloatSlider(
            value=0.015, min=0.001, max=0.150, step=0.002,
            description="Noise Gate (V):", readout_format=".3f",
            layout=widgets.Layout(width="240px")
        )

        self.smooth_dd = widgets.Dropdown(
            options=[("Off (Raw)", 1), ("Medium (3x / 30ms)", 3), ("Strong (5x / 50ms)", 5)],
            value=3,
            description="Smooth:",
            layout=widgets.Layout(width="190px")
        )

        self.f_min_input = widgets.BoundedIntText(
            value=100, min=20, max=24000, step=50,
            description="f_min (Hz):", layout=widgets.Layout(width="160px")
        )

        self.f_max_input = widgets.BoundedIntText(
            value=10000, min=50, max=25000, step=100,
            description="f_max (Hz):", layout=widgets.Layout(width="160px")
        )

        # Profile indicator badge
        prof_name = getattr(self.estimator.profile, "name", "Default")
        self.profile_badge = widgets.HTML(
            f"<span style='color:#FFA500; font-family:monospace; font-size:11px; line-height:28px;'>"
            f"🎯 Profile: <b>{prof_name}</b>"
            f"</span>"
        )

    def _build_plots(self):
        # ---------------------------------------------------------------------
        # Tab 1: Microphone 1 (A0)
        # ---------------------------------------------------------------------
        self.fig_mic1 = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=(
                "<b>Channel 1: A0 In-Band Amplitude Envelope A_true(t) [mV]</b>",
                "<b>Channel 1: A0 Dominant Pitch Trajectory f0(t) [Hz]</b>",
                "<b>Channel 1: A0 Inverted Metric Distance r(t) [cm]</b>"
            )
        )
        self.fig_mic1 = go.FigureWidget(self.fig_mic1)
        # Row 1: Amplitude
        self.fig_mic1.add_scatter(x=self.t_axis, y=self.buf_amp_a0 * 1000.0, mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0 RMS (mV)", row=1, col=1)
        self.fig_mic1.add_scatter(x=[-self.window_duration_sec, 0.0], y=[15.0, 15.0], mode="lines", line=dict(color="#FFA500", width=1.4, dash="dash"), name="Noise Gate", row=1, col=1)
        # Row 2: Pitch
        self.fig_mic1.add_scatter(x=self.t_axis, y=self.buf_freq_a0, mode="markers", marker=dict(size=4.0, color="#00FFCC", opacity=0.85), name="A0 Pitch (Hz)", row=2, col=1)
        # Row 3: Distance + Confidence Band
        self.fig_mic1.add_scatter(x=self.t_axis, y=self.buf_dist_a0, mode="lines", line=dict(width=0.5, color="rgba(0, 255, 204, 0.3)", dash="dot"), showlegend=False, name="A0 +δr", row=3, col=1)
        self.fig_mic1.add_scatter(x=self.t_axis, y=self.buf_dist_a0, mode="lines", line=dict(width=0.5, color="rgba(0, 255, 204, 0.3)", dash="dot"), fill="tonexty", fillcolor="rgba(0, 255, 204, 0.18)", name="A0 ±δr Band", row=3, col=1)
        self.fig_mic1.add_scatter(x=self.t_axis, y=self.buf_dist_a0, mode="lines", line=dict(color="#00FFCC", width=2.0), name="A0 Distance (cm)", row=3, col=1)

        self.fig_mic1.update_layout(template="plotly_dark", height=700, margin=dict(l=55, r=25, t=40, b=30), uirevision="mic1")
        self.fig_mic1.update_yaxes(range=[0, 500], title="A0 (mV)", row=1, col=1)
        self.fig_mic1.update_yaxes(range=[self.f_min_input.value, self.f_max_input.value], title="Pitch (Hz)", row=2, col=1)
        self.fig_mic1.update_yaxes(range=[0, 150], title="Distance (cm)", row=3, col=1)
        self.fig_mic1.update_xaxes(range=[-self.window_duration_sec, 0.0], title="Time Window (Seconds)", row=3, col=1)

        # ---------------------------------------------------------------------
        # Tab 2: Microphone 2 (A1)
        # ---------------------------------------------------------------------
        self.fig_mic2 = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=(
                "<b>Channel 2: A1 In-Band Amplitude Envelope A_true(t) [mV]</b>",
                "<b>Channel 2: A1 Dominant Pitch Trajectory f0(t) [Hz]</b>",
                "<b>Channel 2: A1 Inverted Metric Distance r(t) [cm]</b>"
            )
        )
        self.fig_mic2 = go.FigureWidget(self.fig_mic2)
        # Row 1: Amplitude
        self.fig_mic2.add_scatter(x=self.t_axis, y=self.buf_amp_a1 * 1000.0, mode="lines", line=dict(color="#FF007F", width=1.8), name="A1 RMS (mV)", row=1, col=1)
        self.fig_mic2.add_scatter(x=[-self.window_duration_sec, 0.0], y=[15.0, 15.0], mode="lines", line=dict(color="#FFA500", width=1.4, dash="dash"), name="Noise Gate", row=1, col=1)
        # Row 2: Pitch
        self.fig_mic2.add_scatter(x=self.t_axis, y=self.buf_freq_a1, mode="markers", marker=dict(size=4.0, color="#FF007F", opacity=0.85), name="A1 Pitch (Hz)", row=2, col=1)
        # Row 3: Distance + Confidence Band
        self.fig_mic2.add_scatter(x=self.t_axis, y=self.buf_dist_a1, mode="lines", line=dict(width=0.5, color="rgba(255, 0, 127, 0.3)", dash="dot"), showlegend=False, name="A1 +δr", row=3, col=1)
        self.fig_mic2.add_scatter(x=self.t_axis, y=self.buf_dist_a1, mode="lines", line=dict(width=0.5, color="rgba(255, 0, 127, 0.3)", dash="dot"), fill="tonexty", fillcolor="rgba(255, 0, 127, 0.18)", name="A1 ±δr Band", row=3, col=1)
        self.fig_mic2.add_scatter(x=self.t_axis, y=self.buf_dist_a1, mode="lines", line=dict(color="#FF007F", width=2.0), name="A1 Distance (cm)", row=3, col=1)

        self.fig_mic2.update_layout(template="plotly_dark", height=700, margin=dict(l=55, r=25, t=40, b=30), uirevision="mic2")
        self.fig_mic2.update_yaxes(range=[0, 500], title="A1 (mV)", row=1, col=1)
        self.fig_mic2.update_yaxes(range=[self.f_min_input.value, self.f_max_input.value], title="Pitch (Hz)", row=2, col=1)
        self.fig_mic2.update_yaxes(range=[0, 150], title="Distance (cm)", row=3, col=1)
        self.fig_mic2.update_xaxes(range=[-self.window_duration_sec, 0.0], title="Time Window (Seconds)", row=3, col=1)

        # ---------------------------------------------------------------------
        # Tab 3: Dual Comparison (Overlaid A0 & A1)
        # ---------------------------------------------------------------------
        self.fig_dual = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=(
                "<b>Dual In-Band Amplitude Comparison (A0 Cyan vs A1 Magenta) [mV]</b>",
                "<b>Dual Pitch Scatter Overlay (A0 Cyan dots vs A1 Magenta dots) [Hz]</b>",
                "<b>Dual Inverted Distance Comparison (A0 Cyan vs A1 Magenta) [cm]</b>"
            )
        )
        self.fig_dual = go.FigureWidget(self.fig_dual)
        # Row 1: Amplitudes
        self.fig_dual.add_scatter(x=self.t_axis, y=self.buf_amp_a0 * 1000.0, mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0 (mV)", row=1, col=1)
        self.fig_dual.add_scatter(x=self.t_axis, y=self.buf_amp_a1 * 1000.0, mode="lines", line=dict(color="#FF007F", width=1.8), name="A1 (mV)", row=1, col=1)
        self.fig_dual.add_scatter(x=[-self.window_duration_sec, 0.0], y=[15.0, 15.0], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Noise Gate", row=1, col=1)
        # Row 2: Frequencies
        self.fig_dual.add_scatter(x=self.t_axis, y=self.buf_freq_a0, mode="markers", marker=dict(size=4.0, color="#00FFCC", opacity=0.85), name="A0 Pitch", row=2, col=1)
        self.fig_dual.add_scatter(x=self.t_axis, y=self.buf_freq_a1, mode="markers", marker=dict(size=4.0, color="#FF007F", opacity=0.85), name="A1 Pitch", row=2, col=1)
        # Row 3: Distances
        self.fig_dual.add_scatter(x=self.t_axis, y=self.buf_dist_a0, mode="lines", line=dict(width=0.5, color="rgba(0, 255, 204, 0.3)", dash="dot"), showlegend=False, name="A0 +δr", row=3, col=1)
        self.fig_dual.add_scatter(x=self.t_axis, y=self.buf_dist_a0, mode="lines", line=dict(width=0.5, color="rgba(0, 255, 204, 0.3)", dash="dot"), fill="tonexty", fillcolor="rgba(0, 255, 204, 0.15)", name="A0 ±δr", row=3, col=1)
        self.fig_dual.add_scatter(x=self.t_axis, y=self.buf_dist_a0, mode="lines", line=dict(color="#00FFCC", width=2.0), name="A0 Dist (cm)", row=3, col=1)
        self.fig_dual.add_scatter(x=self.t_axis, y=self.buf_dist_a1, mode="lines", line=dict(width=0.5, color="rgba(255, 0, 127, 0.3)", dash="dot"), showlegend=False, name="A1 +δr", row=3, col=1)
        self.fig_dual.add_scatter(x=self.t_axis, y=self.buf_dist_a1, mode="lines", line=dict(width=0.5, color="rgba(255, 0, 127, 0.3)", dash="dot"), fill="tonexty", fillcolor="rgba(255, 0, 127, 0.15)", name="A1 ±δr", row=3, col=1)
        self.fig_dual.add_scatter(x=self.t_axis, y=self.buf_dist_a1, mode="lines", line=dict(color="#FF007F", width=2.0), name="A1 Dist (cm)", row=3, col=1)

        self.fig_dual.update_layout(template="plotly_dark", height=700, margin=dict(l=55, r=25, t=40, b=30), uirevision="dual")
        self.fig_dual.update_yaxes(range=[0, 500], title="Amplitude (mV)", row=1, col=1)
        self.fig_dual.update_yaxes(range=[self.f_min_input.value, self.f_max_input.value], title="Pitch (Hz)", row=2, col=1)
        self.fig_dual.update_yaxes(range=[0, 150], title="Distance (cm)", row=3, col=1)
        self.fig_dual.update_xaxes(range=[-self.window_duration_sec, 0.0], title="Time Window (Seconds)", row=3, col=1)

        # Assemble Tabs Container
        self.tabs = widgets.Tab(children=[self.fig_mic1, self.fig_mic2, self.fig_dual])
        self.tabs.set_title(0, "🎙 Mic 1 (A0)")
        self.tabs.set_title(1, "🎙 Mic 2 (A1)")
        self.tabs.set_title(2, "🔀 Dual Overlay")

    def _setup_callbacks(self):
        self.start_btn.on_click(lambda _: self.start())
        self.stop_btn.on_click(lambda _: self.stop())
        self.clear_btn.on_click(lambda _: self.reset_buffers())
        self.export_btn.on_click(lambda _: self.export_csv(clean_silence=self.clean_csv_chk.value))
        self.f_min_input.observe(self._on_freq_bounds_change, names="value")
        self.f_max_input.observe(self._on_freq_bounds_change, names="value")

    def _on_freq_bounds_change(self, _):
        f_min = float(self.f_min_input.value)
        f_max = float(self.f_max_input.value)
        if f_max > f_min:
            self.fig_mic1.update_yaxes(range=[f_min, f_max], row=2, col=1)
            self.fig_mic2.update_yaxes(range=[f_min, f_max], row=2, col=1)
            self.fig_dual.update_yaxes(range=[f_min, f_max], row=2, col=1)

    def get_clean_data(
        self,
        channel: int = 1,
        return_distance: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns clean, sanitized, non-NaN arrays for direct post-processing in Jupyter cells.
        :param channel: 1 for Mic 1 (A0), 2 for Mic 2 (A1).
        :param return_distance: If True, returns (t, amp_v, freq_hz, dist_cm, dist_err_cm).
                                If False, returns (t, amp_v, freq_hz) for backwards compatibility.
        :return: Clean arrays without silent NaN frames.
        """
        with self._buf_lock:
            t = np.copy(self.t_axis)
            amp = np.copy(self.buf_amp_a0 if channel == 1 else self.buf_amp_a1)
            freq = np.copy(self.buf_freq_a0 if channel == 1 else self.buf_freq_a1)
            dist = np.copy(self.buf_dist_a0 if channel == 1 else self.buf_dist_a1)
            disterr = np.copy(self.buf_disterr_a0 if channel == 1 else self.buf_disterr_a1)

        valid_mask = np.isfinite(freq) & (amp >= float(self.squelch_slider.value))
        if return_distance:
            return t[valid_mask], amp[valid_mask], freq[valid_mask], dist[valid_mask], disterr[valid_mask]
        return t[valid_mask], amp[valid_mask], freq[valid_mask]

    def get_distance_data(self, channel: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience method to retrieve clean (time_sec, dist_cm, dist_err_cm) arrays.
        """
        t, _, _, dist, disterr = self.get_clean_data(channel=channel, return_distance=True)
        valid = np.isfinite(dist)
        return t[valid], dist[valid], disterr[valid]

    def export_csv(self, filename: Optional[str] = None, clean_silence: bool = True) -> str:
        """
        Exports the synchronized 10-second rolling buffer to a CSV file.
        Includes amplitudes, frequencies, and inverted metric distances for both channels.
        """
        if filename is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"kinematics_telemetry_{ts}.csv"

        out_path = Path(filename).resolve()

        with self._buf_lock:
            t = np.copy(self.t_axis)
            a0_amp = np.copy(self.buf_amp_a0)
            a0_freq = np.copy(self.buf_freq_a0)
            a0_dist = np.copy(self.buf_dist_a0)
            a0_disterr = np.copy(self.buf_disterr_a0)

            a1_amp = np.copy(self.buf_amp_a1)
            a1_freq = np.copy(self.buf_freq_a1)
            a1_dist = np.copy(self.buf_dist_a1)
            a1_disterr = np.copy(self.buf_disterr_a1)

        if clean_silence:
            # Keep rows where at least one channel detected active pitch
            valid_mask = np.isfinite(a0_freq) | np.isfinite(a1_freq)
            if np.sum(valid_mask) == 0:
                valid_mask = (a0_amp >= float(self.squelch_slider.value)) | (a1_amp >= float(self.squelch_slider.value))

            t = t[valid_mask]
            a0_amp = a0_amp[valid_mask]
            a0_freq = a0_freq[valid_mask]
            a0_dist = a0_dist[valid_mask]
            a0_disterr = a0_disterr[valid_mask]

            a1_amp = a1_amp[valid_mask]
            a1_freq = a1_freq[valid_mask]
            a1_dist = a1_dist[valid_mask]
            a1_disterr = a1_disterr[valid_mask]

        data_matrix = np.column_stack([
            t,
            a0_amp, a0_freq, a0_dist, a0_disterr,
            a1_amp, a1_freq, a1_dist, a1_disterr
        ])
        header = "time_sec,a0_amp_v,a0_freq_hz,a0_dist_cm,a0_dist_err_cm,a1_amp_v,a1_freq_hz,a1_dist_cm,a1_dist_err_cm"

        np.savetxt(
            out_path,
            data_matrix,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.4f,%.4f,%.2f,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f"
        )

        tag = "Clean (No NaNs)" if clean_silence else "Full Rolling Buffer"
        self.readout_metrics.value = (
            f"<span style='color:#00FFCC; font-family:monospace; font-size:12px; font-weight:bold;'>"
            f"✅ Saved CSV ({tag}): {out_path.name} ({len(t)} points)"
            f"</span>"
        )
        print(f"[KinematicsDashboard] Exported telemetry to: {out_path}")
        return str(out_path)

    def reset_buffers(self):
        with self._buf_lock:
            self.buf_amp_a0.fill(0.0)
            self.buf_freq_a0.fill(np.nan)
            self.buf_dist_a0.fill(np.nan)
            self.buf_disterr_a0.fill(np.nan)

            self.buf_amp_a1.fill(0.0)
            self.buf_freq_a1.fill(np.nan)
            self.buf_dist_a1.fill(np.nan)
            self.buf_disterr_a1.fill(np.nan)

            self._hist_f0_a0.clear()
            self._hist_f0_a1.clear()

        active_tab = self.tabs.selected_index
        if active_tab == 0:
            with self.fig_mic1.batch_update():
                self.fig_mic1.data[0].y = self.buf_amp_a0
                self.fig_mic1.data[2].y = self.buf_freq_a0
                self.fig_mic1.data[3].y = self.buf_dist_a0
                self.fig_mic1.data[4].y = self.buf_dist_a0
                self.fig_mic1.data[5].y = self.buf_dist_a0
        elif active_tab == 1:
            with self.fig_mic2.batch_update():
                self.fig_mic2.data[0].y = self.buf_amp_a1
                self.fig_mic2.data[2].y = self.buf_freq_a1
                self.fig_mic2.data[3].y = self.buf_dist_a1
                self.fig_mic2.data[4].y = self.buf_dist_a1
                self.fig_mic2.data[5].y = self.buf_dist_a1
        elif active_tab == 2:
            with self.fig_dual.batch_update():
                self.fig_dual.data[0].y = self.buf_amp_a0
                self.fig_dual.data[1].y = self.buf_amp_a1
                self.fig_dual.data[3].y = self.buf_freq_a0
                self.fig_dual.data[4].y = self.buf_freq_a1
                self.fig_dual.data[5].y = self.buf_dist_a0
                self.fig_dual.data[6].y = self.buf_dist_a0
                self.fig_dual.data[7].y = self.buf_dist_a0
                self.fig_dual.data[8].y = self.buf_dist_a1
                self.fig_dual.data[9].y = self.buf_dist_a1
                self.fig_dual.data[10].y = self.buf_dist_a1

    # =========================================================================
    # Thread 1: Fast Producer (Drains Hardware DMA & Computes DSP at 100 Hz)
    # =========================================================================
    def _dsp_worker(self):
        dma = self.overlay.axi_dma_0
        trig = self.trigger

        if hasattr(self.overlay, "xadc_wiz_0"):
            self.overlay.xadc_wiz_0.mmio.write(0x304, 0x2000)
            self.overlay.xadc_wiz_0.mmio.write(0x320, 0x0000)
            self.overlay.xadc_wiz_0.mmio.write(0x324, 0x0202)

        chunk_pts = 1024
        if trig:
            trig.set_decimation(10)
            trig.set_packet_size(chunk_pts)
            trig.set_mode("Auto")

        dma.mmio.write(0x30, 0x04)
        time.sleep(0.002)
        dma.recvchannel.start()

        cma_buf = allocate(shape=(chunk_pts,), dtype="u2")
        dma_armed = False

        try:
            while self._is_running:
                if not dma_armed:
                    dma.recvchannel.transfer(cma_buf)
                    if trig:
                        trig.arm()
                    dma_armed = True

                if dma.recvchannel.idle:
                    dma_armed = False
                    raw = np.array(cma_buf)

                    raw_a0 = raw[0::2]
                    raw_a1 = raw[1::2]

                    v_a0 = (raw_a0 >> 4) * (3.3 / 4095.0)
                    v_a1 = (raw_a1 >> 4) * (3.3 / 4095.0)

                    x_ac_a0 = v_a0 - np.mean(v_a0)
                    x_ac_a1 = v_a1 - np.mean(v_a1)

                    amp_a0 = float(np.sqrt(np.mean(x_ac_a0 ** 2)))
                    amp_a1 = float(np.sqrt(np.mean(x_ac_a1 ** 2)))
                    squelch = float(self.squelch_slider.value)
                    smooth_taps = int(self.smooth_dd.value)

                    f_min = float(self.f_min_input.value)
                    f_max = float(self.f_max_input.value)

                    # 1. Channel 1 (A0) Pitch Tracking (Linear 2048-point STFT)
                    if amp_a0 >= squelch and len(x_ac_a0) >= 128:
                        pad_len = self.win_len - len(x_ac_a0)
                        p0 = np.pad(x_ac_a0, (0, pad_len), mode="constant") if pad_len > 0 else x_ac_a0[:self.win_len]
                        
                        # Correct: Extract linear magnitude for exact sinc ratio peak estimator
                        mag_linear0 = (np.abs(np.fft.rfft(p0 * self.stft_win)) / (self.win_len / 2.0)) / max(self.coherent_gain, 1e-4)
                        
                        raw_f0_a0, _ = KinematicAnalytics.track_sub_hertz_pitch(
                            self.freq_axis, mag_linear0, min_freq_hz=f_min, max_freq_hz=f_max, interpolate=True
                        )

                        # Moving Median Filter
                        self._hist_f0_a0.append(raw_f0_a0)
                        recent_pts = list(self._hist_f0_a0)[-smooth_taps:]
                        f0_a0 = float(np.median(recent_pts))

                        # Coherent in-band physical amplitude (BFP-immune)
                        a_true_a0 = KinematicAnalytics.compute_coherent_inband_amplitude(
                            signal_v=v_a0,
                            fs=self.fs_per_ch,
                            target_freq_hz=f0_a0,
                            remove_dc=True
                        )
                        # Metric distance inversion: r = k(f0) / A_true
                        r_m_a0, r_err_m_a0, st_a0 = self.estimator.estimate_distance(
                            amplitude_v=a_true_a0,
                            frequency_hz=f0_a0
                        )
                        if np.isfinite(r_m_a0):
                            r_cm_a0 = r_m_a0 * 100.0
                            r_err_cm_a0 = r_err_m_a0 * 100.0
                        else:
                            r_cm_a0 = np.nan
                            r_err_cm_a0 = np.nan
                    else:
                        f0_a0 = np.nan
                        a_true_a0 = amp_a0
                        r_cm_a0 = np.nan
                        r_err_cm_a0 = np.nan
                        st_a0 = "SILENCE"
                        self._hist_f0_a0.clear()

                    # 2. Channel 2 (A1) Pitch Tracking (Linear 2048-point STFT)
                    if amp_a1 >= squelch and len(x_ac_a1) >= 128:
                        pad_len = self.win_len - len(x_ac_a1)
                        p1 = np.pad(x_ac_a1, (0, pad_len), mode="constant") if pad_len > 0 else x_ac_a1[:self.win_len]
                        
                        # Correct: Extract linear magnitude for exact sinc ratio peak estimator
                        mag_linear1 = (np.abs(np.fft.rfft(p1 * self.stft_win)) / (self.win_len / 2.0)) / max(self.coherent_gain, 1e-4)
                        
                        raw_f0_a1, _ = KinematicAnalytics.track_sub_hertz_pitch(
                            self.freq_axis, mag_linear1, min_freq_hz=f_min, max_freq_hz=f_max, interpolate=True
                        )

                        # Moving Median Filter
                        self._hist_f0_a1.append(raw_f0_a1)
                        recent_pts = list(self._hist_f0_a1)[-smooth_taps:]
                        f0_a1 = float(np.median(recent_pts))

                        # Coherent in-band physical amplitude (BFP-immune)
                        a_true_a1 = KinematicAnalytics.compute_coherent_inband_amplitude(
                            signal_v=v_a1,
                            fs=self.fs_per_ch,
                            target_freq_hz=f0_a1,
                            remove_dc=True
                        )
                        # Metric distance inversion: r = k(f0) / A_true
                        r_m_a1, r_err_m_a1, st_a1 = self.estimator.estimate_distance(
                            amplitude_v=a_true_a1,
                            frequency_hz=f0_a1
                        )
                        if np.isfinite(r_m_a1):
                            r_cm_a1 = r_m_a1 * 100.0
                            r_err_cm_a1 = r_err_m_a1 * 100.0
                        else:
                            r_cm_a1 = np.nan
                            r_err_cm_a1 = np.nan
                    else:
                        f0_a1 = np.nan
                        a_true_a1 = amp_a1
                        r_cm_a1 = np.nan
                        r_err_cm_a1 = np.nan
                        st_a1 = "SILENCE"
                        self._hist_f0_a1.clear()

                    self._cur_amp_a0 = a_true_a0
                    self._cur_f0_a0 = f0_a0
                    self._cur_dist_a0 = r_cm_a0
                    self._cur_disterr_a0 = r_err_cm_a0

                    self._cur_amp_a1 = a_true_a1
                    self._cur_f0_a1 = f0_a1
                    self._cur_dist_a1 = r_cm_a1
                    self._cur_disterr_a1 = r_err_cm_a1

                    with self._buf_lock:
                        self.buf_amp_a0[:-1] = self.buf_amp_a0[1:]
                        self.buf_amp_a0[-1] = a_true_a0

                        self.buf_freq_a0[:-1] = self.buf_freq_a0[1:]
                        self.buf_freq_a0[-1] = f0_a0

                        self.buf_dist_a0[:-1] = self.buf_dist_a0[1:]
                        self.buf_dist_a0[-1] = r_cm_a0

                        self.buf_disterr_a0[:-1] = self.buf_disterr_a0[1:]
                        self.buf_disterr_a0[-1] = r_err_cm_a0

                        self.buf_amp_a1[:-1] = self.buf_amp_a1[1:]
                        self.buf_amp_a1[-1] = a_true_a1

                        self.buf_freq_a1[:-1] = self.buf_freq_a1[1:]
                        self.buf_freq_a1[-1] = f0_a1

                        self.buf_dist_a1[:-1] = self.buf_dist_a1[1:]
                        self.buf_dist_a1[-1] = r_cm_a1

                        self.buf_disterr_a1[:-1] = self.buf_disterr_a1[1:]
                        self.buf_disterr_a1[-1] = r_err_cm_a1
                else:
                    time.sleep(0.001)

        finally:
            if cma_buf is not None:
                cma_buf.close()
            dma.mmio.write(0x30, 0x04)
            if trig:
                trig.disarm()
                
    # =========================================================================
    # Thread 2: UI Consumer (Smart Active-Tab 30 FPS Render)
    # =========================================================================
    def _render_worker(self):
        while self._is_running:
            time.sleep(0.033)
            if not self._is_running:
                break

            with self._buf_lock:
                amp_a0_mv = np.copy(self.buf_amp_a0) * 1000.0
                freq_a0 = np.copy(self.buf_freq_a0)
                dist_a0 = np.copy(self.buf_dist_a0)
                disterr_a0 = np.copy(self.buf_disterr_a0)

                amp_a1_mv = np.copy(self.buf_amp_a1) * 1000.0
                freq_a1 = np.copy(self.buf_freq_a1)
                dist_a1 = np.copy(self.buf_dist_a1)
                disterr_a1 = np.copy(self.buf_disterr_a1)

            squelch_mv = float(self.squelch_slider.value) * 1000.0
            active_tab = self.tabs.selected_index

            dist_upper_a0 = dist_a0 + disterr_a0
            dist_lower_a0 = np.maximum(0.0, dist_a0 - disterr_a0)
            dist_upper_a1 = dist_a1 + disterr_a1
            dist_lower_a1 = np.maximum(0.0, dist_a1 - disterr_a1)

            if active_tab == 0:
                with self.fig_mic1.batch_update():
                    self.fig_mic1.data[0].y = amp_a0_mv
                    self.fig_mic1.data[1].y = [squelch_mv, squelch_mv]
                    self.fig_mic1.data[2].y = freq_a0
                    self.fig_mic1.data[3].y = dist_upper_a0
                    self.fig_mic1.data[4].y = dist_lower_a0
                    self.fig_mic1.data[5].y = dist_a0
            elif active_tab == 1:
                with self.fig_mic2.batch_update():
                    self.fig_mic2.data[0].y = amp_a1_mv
                    self.fig_mic2.data[1].y = [squelch_mv, squelch_mv]
                    self.fig_mic2.data[2].y = freq_a1
                    self.fig_mic2.data[3].y = dist_upper_a1
                    self.fig_mic2.data[4].y = dist_lower_a1
                    self.fig_mic2.data[5].y = dist_a1
            elif active_tab == 2:
                with self.fig_dual.batch_update():
                    self.fig_dual.data[0].y = amp_a0_mv
                    self.fig_dual.data[1].y = amp_a1_mv
                    self.fig_dual.data[2].y = [squelch_mv, squelch_mv]
                    self.fig_dual.data[3].y = freq_a0
                    self.fig_dual.data[4].y = freq_a1
                    self.fig_dual.data[5].y = dist_upper_a0
                    self.fig_dual.data[6].y = dist_lower_a0
                    self.fig_dual.data[7].y = dist_a0
                    self.fig_dual.data[8].y = dist_upper_a1
                    self.fig_dual.data[9].y = dist_lower_a1
                    self.fig_dual.data[10].y = dist_a1

            f0_str_a0 = f"{self._cur_f0_a0:.1f}Hz" if np.isfinite(self._cur_f0_a0) else "---"
            f0_str_a1 = f"{self._cur_f0_a1:.1f}Hz" if np.isfinite(self._cur_f0_a1) else "---"

            dist_str_a0 = f"{self._cur_dist_a0:.1f}cm" if np.isfinite(self._cur_dist_a0) else "---"
            dist_str_a1 = f"{self._cur_dist_a1:.1f}cm" if np.isfinite(self._cur_dist_a1) else "---"

            self.readout_metrics.value = (
                f"<span style='color:#00FFCC; font-family:monospace; font-size:12px; font-weight:bold;'>"
                f"A0: {self._cur_amp_a0*1000.0:.1f}mV ({f0_str_a0} → <b>{dist_str_a0}</b>) | "
                f"A1: {self._cur_amp_a1*1000.0:.1f}mV ({f0_str_a1} → <b>{dist_str_a1}</b>) | "
                f"Live: 30 FPS"
                f"</span>"
            )

    def start(self):
        if not self._is_running:
            self._is_running = True
            self._dsp_thread = threading.Thread(target=self._dsp_worker, daemon=True)
            self._render_thread = threading.Thread(target=self._render_worker, daemon=True)
            self._dsp_thread.start()
            self._render_thread.start()
            print("[KinematicsDashboard] Multi-Tab Live Stream Started.")

    def stop(self):
        if self._is_running:
            self._is_running = False
            if self.overlay and hasattr(self.overlay, "axi_dma_0"):
                self.overlay.axi_dma_0.mmio.write(0x30, 0x04)
            if self.trigger:
                self.trigger.disarm()

            if self._dsp_thread and self._dsp_thread.is_alive():
                self._dsp_thread.join(timeout=0.3)
            if self._render_thread and self._render_thread.is_alive():
                self._render_thread.join(timeout=0.3)

            self.readout_metrics.value = (
                "<span style='color:#FFA500; font-family:monospace; font-size:12px; font-weight:bold;'>"
                "A0: 0.0mV (--- → ---) | A1: 0.0mV (--- → ---) | Status: STOPPED"
                "</span>"
            )
            print("[KinematicsDashboard] Stream Stopped Cleanly.")

    def display(self):
        r1 = widgets.HBox(
            [self.start_btn, self.stop_btn, self.clear_btn, self.export_btn, self.clean_csv_chk, self.readout_metrics],
            layout=widgets.Layout(gap="10px", margin="0 0 8px 0")
        )
        r2 = widgets.HBox(
            [self.squelch_slider, self.smooth_dd, self.f_min_input, self.f_max_input, self.profile_badge],
            layout=widgets.Layout(gap="10px")
        )
        control_panel = widgets.VBox([r1, r2], layout=widgets.Layout(margin="0 0 10px 0"))
        display(widgets.VBox([control_panel, self.tabs]))