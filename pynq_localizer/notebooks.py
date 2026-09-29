"""
pynq_localizer.notebooks: Automated Installer for Interactive Jupyter Labs & Experiment Harvesters.
Deploys official pynq_sound_localizer demo notebooks (01-06) and dedicated experiment harvesters
(experiments/exp01-exp03) with protected data output storage into the Jupyter workspace.
"""

import os
import shutil
from pathlib import Path

# Whitelist of official top-level interactive lab notebooks
LOCALIZER_NOTEBOOKS = [
    "01_realtime_kinematics_telemetry.ipynb",
    "02_acoustic_calibration_lab.ipynb",
    "03_phase_angle_of_arrival_lab.ipynb",
    "04_air_track_differential_doppler.ipynb",
    "05_pulse_time_of_arrival_lab.ipynb",
    "06_planar_tdoa_sound_localizer.ipynb",
]

# Whitelist of dedicated statistical experiment harvester notebooks
EXPERIMENT_NOTEBOOKS = [
    "exp01_distance_energy_characterization.ipynb",
    "exp02_angle_3method_characterization.ipynb",
    "exp03_air_track_kinematics_harvesting.ipynb",
]


def install_localizer_notebooks(target_dir: str = None):
    """
    Copies official demo notebooks and dedicated experiment harvesters into
    the Jupyter root folder under /home/xilinx/jupyter_notebooks/pynq_sound_localizer/.
    Preserves and protects user-generated data files in experiments/data/.
    """
    package_dir = Path(__file__).resolve().parent.parent
    src_notebooks = package_dir / "notebooks"

    if not src_notebooks.exists():
        src_notebooks = Path(__file__).resolve().parent / "notebooks_data"

    src_experiments = src_notebooks / "experiments"

    if target_dir is None:
        pynq_jupyter_root = Path("/home/xilinx/jupyter_notebooks")
        if pynq_jupyter_root.exists():
            dest_dir = pynq_jupyter_root / "pynq_sound_localizer"
        else:
            dest_dir = Path.cwd() / "pynq_sound_localizer_notebooks"
    else:
        dest_dir = Path(target_dir).resolve()

    dest_experiments = dest_dir / "experiments"
    dest_data = dest_experiments / "data"

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_experiments.mkdir(parents=True, exist_ok=True)
    dest_data.mkdir(parents=True, exist_ok=True)

    # 1. Purge foreign/stale notebooks from root folder (leaves subfolders intact)
    for existing_file in dest_dir.glob("*.ipynb"):
        if existing_file.name not in LOCALIZER_NOTEBOOKS:
            try:
                existing_file.unlink()
            except Exception:
                pass

    # 2. Purge stale notebooks from experiments folder (leaves data/ directory intact)
    for existing_file in dest_experiments.glob("*.ipynb"):
        if existing_file.name not in EXPERIMENT_NOTEBOOKS:
            try:
                existing_file.unlink()
            except Exception:
                pass

    # 3. Deploy top-level interactive lab notebooks
    deployed_labs = []
    for nb_name in LOCALIZER_NOTEBOOKS:
        src_file = src_notebooks / nb_name
        if src_file.exists():
            dest_file = dest_dir / nb_name
            shutil.copy2(src_file, dest_file)
            deployed_labs.append(nb_name)

    # 4. Deploy dedicated experiment harvester notebooks
    deployed_experiments = []
    if src_experiments.exists():
        for nb_name in EXPERIMENT_NOTEBOOKS:
            src_file = src_experiments / nb_name
            if src_file.exists():
                dest_file = dest_experiments / nb_name
                shutil.copy2(src_file, dest_file)
                deployed_experiments.append(nb_name)

    print("=" * 80)
    print(f"🚀 [NotebookInstaller] Notebooks deployed to: {dest_dir}")
    print("=" * 80)
    print(f"📖 Interactive Labs ({len(deployed_labs)} deployed):")
    for f in deployed_labs:
        print(f"   • {f}")

    print(f"\n🔬 Dedicated Experiment Harvesters ({len(deployed_experiments)} deployed):")
    if deployed_experiments:
        for f in deployed_experiments:
            print(f"   • experiments/{f}")
    else:
        print("   (Harvester notebooks will appear once created in Phase 5)")

    print(f"\n📁 Persistent Data Output Directory: {dest_data}")
    print("   (All .xlsx and .csv exports will be saved here)")
    print("=" * 80)


# Backward-compatibility alias
copy_notebooks = install_localizer_notebooks

if __name__ == "__main__":
    install_localizer_notebooks()