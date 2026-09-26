"""
pynq_localizer.notebooks: Automated Installer for Interactive Jupyter Lab Notebooks.
Deploys only official pynq_sound_localizer notebooks into the Jupyter workspace.
"""

import os
import shutil
from pathlib import Path

# Strict whitelist of notebooks belonging ONLY to pynq_sound_localizer
LOCALIZER_NOTEBOOKS = [
    "01_realtime_kinematics_telemetry.ipynb",
    "02_acoustic_calibration_lab.ipynb",
    "03_phase_angle_of_arrival_lab.ipynb",
    "04_air_track_differential_doppler.ipynb",
    "05_pulse_time_of_arrival_lab.ipynb",
]


def install_localizer_notebooks(target_dir: str = None):
    """
    Copies official pynq_sound_localizer example notebooks into the Jupyter root folder
    under /home/xilinx/jupyter_notebooks/pynq_sound_localizer/.
    Automatically purges any collided/foreign notebooks from other packages.
    """
    package_dir = Path(__file__).resolve().parent.parent
    src_notebooks = package_dir / "notebooks"

    if not src_notebooks.exists():
        src_notebooks = Path(__file__).resolve().parent / "notebooks_data"

    if target_dir is None:
        pynq_jupyter_root = Path("/home/xilinx/jupyter_notebooks")
        if pynq_jupyter_root.exists():
            dest_dir = pynq_jupyter_root / "pynq_sound_localizer"
        else:
            dest_dir = Path.cwd() / "pynq_sound_localizer_notebooks"
    else:
        dest_dir = Path(target_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Purge any foreign/stale notebooks from other packages in this folder
    for existing_file in dest_dir.glob("*.ipynb"):
        if existing_file.name not in LOCALIZER_NOTEBOOKS:
            try:
                existing_file.unlink()
            except Exception:
                pass

    # 2. Copy ONLY official localizer notebooks
    deployed_files = []
    for nb_name in LOCALIZER_NOTEBOOKS:
        src_file = src_notebooks / nb_name
        if src_file.exists():
            dest_file = dest_dir / nb_name
            shutil.copy2(src_file, dest_file)
            deployed_files.append(nb_name)

    print(f"[NotebookInstaller] Successfully deployed {len(deployed_files)} localizer notebooks to:")
    print(f"                   {dest_dir.resolve()}")
    for f in deployed_files:
        print(f"  • {f}")


# Backward-compatibility alias
copy_notebooks = install_localizer_notebooks

if __name__ == "__main__":
    install_localizer_notebooks()