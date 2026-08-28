import os
import shutil
from pathlib import Path


def copy_notebooks(target_dir: str = None):
    """
    Copies pynq_sound_localizer example notebooks into the Jupyter root folder.
    """
    # 1. First check inside the installed package namespace
    pkg_dir = Path(__file__).resolve().parent
    src_notebooks = pkg_dir / "notebooks"

    # 2. Fallback to repo root if running in editable / source mode
    if not src_notebooks.exists() or not any(src_notebooks.glob("*.ipynb")):
        src_notebooks = pkg_dir.parent / "notebooks"

    if target_dir is None:
        pynq_jupyter_root = Path("/home/xilinx/jupyter_notebooks")
        dest_dir = pynq_jupyter_root / "pynq_sound_localizer" if pynq_jupyter_root.exists() else Path.cwd() / "pynq_sound_localizer_notebooks"
    else:
        dest_dir = Path(target_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    if src_notebooks.exists():
        for item in src_notebooks.glob("*.ipynb"):
            dest_file = dest_dir / item.name
            shutil.copy2(item, dest_file)
            copied_files.append(item.name)

    print(f"[NotebookInstaller] Successfully copied {len(copied_files)} notebooks to:\n                   {dest_dir.resolve()}")
    for f in copied_files:
        print(f"  • {f}")


if __name__ == "__main__":
    copy_notebooks()
