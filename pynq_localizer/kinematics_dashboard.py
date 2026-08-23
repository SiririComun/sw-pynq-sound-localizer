"""
pynq_localizer.kinematics_dashboard: Real-Time 10-Second Rolling Telemetry Dashboard.
Displays synchronized Amplitude vs. Time (Row 1) and Dominant Frequency vs. Time (Row 2)
using high-efficiency circular ring buffers and decoupled 30 FPS Plotly rendering.
"""

import time
import threading
from typing import Optional
from IPython.display import clear_output, display
import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pynq import allocate

from pynq_localizer.kinematics import KinematicAnalytics
from pynq_localizer.hw_trigger import HardwareTrigger


class KinematicsDashboard:
    """
    Real-time 10-second rolling acoustic kinematics dashboard.
    Visualizes instantaneous smoothed amplitude A(t) and dominant pitch f0(t).
    """

    def __init__(
        self,
        overlay=None,
        window_duration_sec: float = 10.0,
        hop_ms: float = 10.0,
        fs_per_ch: float = 50_000.0,
        channel: int = 1
    ):
        self.overlay = overlay
        self.window_duration_sec = float(window_duration_sec)
        self.hop_ms = float(hop_ms)
        self.fs_per_ch = float(fs_per_ch)
        self.channel = int(channel)  # 1 for A0 (Mic 1), 2 for A1 (Mic 2)

        # Buffer sizing: 100 points per second for 10.0 ms hop
        self.pts_per_sec = int(1000.0 / self.hop_ms)
        self.buffer_len = int(self.window_duration_sec * self.pts_per_sec)  # 1000 points

        # Rolling circular ring buffers
        self.t_axis = np.linspace(-self.window_duration_sec, 0.0, self.buffer_len)
        self.buf_amp = np.zeros(self.buffer_len, dtype=np.float64)
        self.buf_freq = np.full(self.buffer_len, np.nan, dtype=np.float64)

        # STFT window pre-calculation (20 ms window = 1024 samples at 50 kSPS)
        self.win_len = 1024
        n = np.arange(self.win_len)
        self.stft_win = (0.35875 - 0.48829 * np.cos(2.0 * np.pi * n / (self.win_len - 1)) +
                         0.14128 * np.cos(4.0 * np.pi * n / (self.win_len - 1)) -
                         0.01168 * np.cos(6.0 * np.pi * n / (self.win_len - 1)))
        self.coherent_gain = np.sum(self.stft_win) / self.win_len
        self.freq_axis = np.fft.rfftfreq(self.win_len, d=1.0 / self.fs_per_ch)

        self._is_running = False
        self._thread: Optional[threading.Thread] = None

        if self.overlay and hasattr(self.overlay, "trigger"):
            self.trigger = self.overlay.trigger
        elif self.overlay:
            self.trigger = HardwareTrigger(self.overlay)
        else:
            self.trigger = None

        self._build_ui()
        self._build_plots()
        self._setup_callbacks()

    def _build_ui(self):
        # 1. Action Row
        self.start_btn = widgets.Button(description="Start Stream", button_style="success", icon="play", layout=widgets.Layout(width="125px"))
        self.stop_btn = widgets.Button(description="Stop", button_style="danger", icon="stop", layout=widgets.Layout(width="95px"))
        self.clear_btn = widgets.Button(description="Reset View", button_style="warning", icon="refresh", layout=widgets.Layout(width="110px"))

        # 2. Live Readouts
        self.readout_metrics = widgets.HTML(
            "<span style='color:#00FFCC; font-family:monospace; font-size:14px; font-weight:bold;'>"
            "A(t): 0.000 V | Dominant f0: 0.0 Hz | Squelch: ACTIVE"
            "</span>"
        )

        # 3. Interactive Controls Row
        self.channel_dd = widgets.Dropdown(
            options=[("Mic 1 (Header A0)", 1), ("Mic 2 (Header A1)", 2)],
            value=self.channel,
            description="Source:",
            layout=widgets.Layout(width="200px")
        )

        self.squelch_slider = widgets.FloatSlider(
            value=0.015,
            min=0.001,
            max=0.150,
            step=0.002,
            description="Noise Gate:",
            readout_format=".3f",
            layout=widgets.Layout(width="260px")
        )

        self.f_min_input = widgets.BoundedIntText(
            value=100, min=20, max=20000, step=50,
            description="f_min (Hz):",
            layout=widgets.Layout(width="170px")
        )

        self.f_max_input = widgets.BoundedIntText(
            value=10000, min=50, max=25000, step=100,
            description="f_max (Hz):",
            layout=widgets.Layout(width="170px")
        )

    def _build_plots(self):
        self.fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
            subplot_titles=(
                "<b>1. Instantaneous Acoustic Amplitude Envelope A(t) [Physical Volts]</b>",
                "<b>2. Real-Time Dominant Frequency Trajectory f0(t) [20 Hz - 20 kHz]</b>"
            )
        )
        self.fig = go.FigureWidget(self.fig)

        # Row 1: Amplitude Curve & Squelch Line
        self.fig.add_scatter(
            x=self.t_axis, y=self.buf_amp, mode="lines",
            line=dict(color="#00FFCC", width=2.0), name="A(t) RMS (V)", row=1, col=1
        )
        self.fig.add_scatter(
            x=[-self.window_duration_sec, 0.0], y=[0.015, 0.015], mode="lines",
            line=dict(color="#FFA500", width=1.4, dash="dash"), name="Noise Gate", row=1, col=1
        )

        # Row 2: Dominant Frequency Curve
        self.fig.add_scatter(
            x=self.t_axis, y=self.buf_freq, mode="lines+markers",
            marker=dict(size=3, color="#FF007F"),
            line=dict(color="#FF007F", width=2.0), name="Pitch f0(t) (Hz)", row=2, col=1
        )

        self.fig.update_layout(
            template="plotly_dark", height=540,
            margin=dict(l=55, r=30, t=45, b=35), uirevision="kinematics_telemetry"
        )
        self.fig.update_yaxes(range=[0, 1.65], title="Amplitude (V)", row=1, col=1)
        self.fig.update_yaxes(range=[0, 6000], title="Frequency (Hz)", row=2, col=1)
        self.fig.update_xaxes(range=[-self.window_duration_sec, 0.0], title="Time Window (Seconds relative to now)", row=2, col=1)

    def _setup_callbacks(self):
        self.start_btn.on_click(lambda _: self.start())
        self.stop_btn.on_click(lambda _: self.stop())
        self.clear_btn.on_click(lambda _: self.reset_buffers())
        self.channel_dd.observe(lambda c: setattr(self, "channel", c["new"]), names="value")
        self.f_max_input.observe(self._on_f_max_change, names="value")

    def _on_f_max_change(self, change):
        self.fig.update_yaxes(range=[0, change["new"] * 1.1], row=2, col=1)

    def reset_buffers(self):
        self.buf_amp.fill(0.0)
        self.buf_freq.fill(np.nan)
        with self.fig.batch_update():
            self.fig.data[0].y = self.buf_amp
            self.fig.data[2].y = self.buf_freq

    def _update_loop(self):
        dma = self.overlay.axi_dma_0
        trig = self.trigger

        # Initialize XADC simultaneous dual conversion (0.00 µs skew)
        if hasattr(self.overlay, "xadc_wiz_0"):
            self.overlay.xadc_wiz_0.mmio.write(0x304, 0x2000)
            self.overlay.xadc_wiz_0.mmio.write(0x320, 0x0000)
            self.overlay.xadc_wiz_0.mmio.write(0x324, 0x0202)

        # Set packet limit to 1024 interleaved words (512 per channel = 10.24 ms at 50 kSPS)
        chunk_pts = 1024
        if trig:
            trig.set_decimation(10)  # M=10 -> 50 kSPS
            trig.set_packet_size(chunk_pts)
            trig.set_mode("Auto")

        dma.mmio.write(0x30, 0x04)
        time.sleep(0.002)
        dma.recvchannel.start()

        cma_buf = allocate(shape=(chunk_pts,), dtype="u2")
        dma_armed = False
        last_render_time = time.time()

        # Rolling circular pointers
        write_idx = 0
        cur_amp = 0.0
        cur_f0 = np.nan

        print(f"[KinematicsDashboard] Live Streaming Active (10s Window | 100 Hz DSP Telemetry Rate)")

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

                    # De-interleave channel
                    if self.channel == 1:
                        v_raw = (raw[0::2] >> 4) * (3.3 / 4095.0)  # Mic 1 (A0)
                    else:
                        v_raw = (raw[1::2] >> 4) * (3.3 / 4095.0)  # Mic 2 (A1)

                    x_ac = v_raw - np.mean(v_raw)
                    cur_amp = float(np.sqrt(np.mean(x_ac ** 2)))
                    squelch = float(self.squelch_slider.value)

                    # Frequency extraction if above squelch threshold
                    if cur_amp >= squelch and len(x_ac) >= 128:
                        # Zero-pad or tile to STFT window length (1024)
                        pad_len = self.win_len - len(x_ac)
                        chunk_padded = np.pad(x_ac, (0, pad_len), mode="constant") if pad_len > 0 else x_ac[:self.win_len]
                        windowed = chunk_padded * self.stft_win

                        fft_mag = np.abs(np.fft.rfft(windowed)) / (self.win_len / 2.0)
                        linear_v = fft_mag / max(self.coherent_gain, 1e-4)
                        mag_db = 20.0 * np.log10(np.maximum(linear_v, 1e-6))

                        cur_f0, _ = KinematicAnalytics.track_sub_hertz_pitch(
                            self.freq_axis, mag_db,
                            min_freq_hz=float(self.f_min_input.value),
                            max_freq_hz=float(self.f_max_input.value),
                            interpolate=True
                        )
                    else:
                        cur_f0 = np.nan

                    # Push into rolling ring buffers
                    self.buf_amp[:-1] = self.buf_amp[1:]
                    self.buf_amp[-1] = cur_amp

                    self.buf_freq[:-1] = self.buf_freq[1:]
                    self.buf_freq[-1] = cur_f0

                    # 30 FPS Render Throttling (Every ~33 ms)
                    now = time.time()
                    if now - last_render_time >= 0.033:
                        last_render_time = now
                        with self.fig.batch_update():
                            self.fig.data[0].y = self.buf_amp
                            self.fig.data[1].y = [squelch, squelch]
                            self.fig.data[2].y = self.buf_freq

                        status_tag = "ACTIVE (Tracking)" if not np.isnan(cur_f0) else "SQUELCHED (Silent)"
                        freq_str = f"{cur_f0:.1f} Hz" if not np.isnan(cur_f0) else "--- Hz"
                        self.readout_metrics.value = (
                            f"<span style='color:#00FFCC; font-family:monospace; font-size:14px; font-weight:bold;'>"
                            f"A(t): {cur_amp:.3f} V | Dominant f0: {freq_str} | Status: {status_tag}"
                            f"</span>"
                        )
                else:
                    time.sleep(0.001)

        finally:
            self._is_running = False
            cma_buf.close()
            print("[KinematicsDashboard] Stream stopped cleanly.")

    def start(self):
        if not self._is_running:
            self._is_running = True
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._is_running = False

    def display(self):
        r1 = widgets.HBox([self.start_btn, self.stop_btn, self.clear_btn, self.readout_metrics], layout=widgets.Layout(gap="10px", margin="0 0 8px 0"))
        r2 = widgets.HBox([self.channel_dd, self.squelch_slider, self.f_min_input, self.f_max_input])
        control_panel = widgets.VBox([r1, r2], layout=widgets.Layout(margin="0 0 10px 0"))
        display(widgets.VBox([control_panel, self.fig]))