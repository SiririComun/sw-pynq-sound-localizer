"""
tests/test_dashboard.py: Verification Suite for KinematicsDashboard 3-Row Telemetry View.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from pynq_localizer.kinematics import AcousticProfile, DistanceEstimator
from pynq_localizer.kinematics_dashboard import KinematicsDashboard

class TestKinematicsDashboard:

    def test_dashboard_initialization_and_buffer_geometry(self):
        """Verify dashboard buffers and distance estimator initialization."""
        dash = KinematicsDashboard(
            overlay=None,
            window_duration_sec=10.0,
            hop_ms=10.0,
            k_constant=0.050
        )

        assert dash.buffer_len == 1000
        assert len(dash.buf_amp_a0) == 1000
        assert len(dash.buf_freq_a0) == 1000
        assert len(dash.buf_dist_a0) == 1000
        assert len(dash.buf_disterr_a0) == 1000

        assert len(dash.buf_amp_a1) == 1000
        assert len(dash.buf_freq_a1) == 1000
        assert len(dash.buf_dist_a1) == 1000
        assert len(dash.buf_disterr_a1) == 1000

        # Estimator was properly configured
        k_eval, _ = dash.estimator.profile.evaluate(1000.0)
        assert abs(k_eval - 0.050) < 1e-6

    def test_figure_trace_counts_and_subplot_geometry(self):
        """Verify 3-row layout and exact trace counts for all tabs."""
        dash = KinematicsDashboard(overlay=None, k_constant=0.040)

        # Tab 1: Mic 1 (6 traces: Amp, Gate, Pitch, Dist+err, Dist-err, Dist)
        assert len(dash.fig_mic1.data) == 6
        # Tab 2: Mic 2 (6 traces)
        assert len(dash.fig_mic2.data) == 6
        # Tab 3: Dual Overlay (11 traces: A0/A1 amp, gate, A0/A1 pitch, A0 upper/lower/dist, A1 upper/lower/dist)
        assert len(dash.fig_dual.data) == 11

    def test_get_clean_data_and_distance_handoff(self):
        """Verify clean telemetry extraction with and without distance arrays."""
        dash = KinematicsDashboard(overlay=None, k_constant=0.050)

        # Inject simulated motion data into rolling buffers
        with dash._buf_lock:
            dash.buf_amp_a0[500:600] = 0.050      # 50 mV -> r = 0.050 / 0.050 = 1.0 m = 100 cm
            dash.buf_freq_a0[500:600] = 1200.0
            dash.buf_dist_a0[500:600] = 100.0
            dash.buf_disterr_a0[500:600] = 4.0

        # Backwards compatible 3-tuple
        t_clean, amp_clean, freq_clean = dash.get_clean_data(channel=1, return_distance=False)
        assert len(t_clean) == 100
        assert np.allclose(amp_clean, 0.050)
        assert np.allclose(freq_clean, 1200.0)

        # Extended 5-tuple with distance metrics
        t_clean, amp_clean, freq_clean, dist_clean, disterr_clean = dash.get_clean_data(
            channel=1, return_distance=True
        )
        assert len(dist_clean) == 100
        assert np.allclose(dist_clean, 100.0)
        assert np.allclose(disterr_clean, 4.0)

        # Convenience helper
        t_dist, r_cm, r_err_cm = dash.get_distance_data(channel=1)
        assert len(t_dist) == 100
        assert np.allclose(r_cm, 100.0)
        assert np.allclose(r_err_cm, 4.0)

    def test_export_csv_with_distance_columns(self):
        """Verify CSV export includes distance and uncertainty headers and data."""
        dash = KinematicsDashboard(overlay=None, k_constant=0.050)

        with dash._buf_lock:
            dash.buf_amp_a0[10:20] = 0.080
            dash.buf_freq_a0[10:20] = 1500.0
            dash.buf_dist_a0[10:20] = 62.5
            dash.buf_disterr_a0[10:20] = 2.5

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_csv = Path(tmp.name)

        try:
            saved_path = dash.export_csv(filename=str(tmp_csv), clean_silence=True)
            with open(saved_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            assert len(lines) == 11  # 1 header + 10 data rows
            header = lines[0].strip()
            assert header == "time_sec,a0_amp_v,a0_freq_hz,a0_dist_cm,a0_dist_err_cm,a1_amp_v,a1_freq_hz,a1_dist_cm,a1_dist_err_cm"
        finally:
            if tmp_csv.exists():
                tmp_csv.unlink()