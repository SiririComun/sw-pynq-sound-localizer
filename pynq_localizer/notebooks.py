import os
import shutil
from pathlib import Path


def copy_notebooks(target_dir: str = None):
    """
    Copies pynq_sound_localizer example notebooks into the Jupyter root folder.
    Strictly isolated to pynq_localizer.notebooks subpackage.
    """
    pkg_notebooks_dir = Path(__file__).resolve().parent / "notebooks"

    if target_dir is None:
        pynq_jupyter_root = Path("/home/xilinx/jupyter_notebooks")
        dest_dir = (
            pynq_jupyter_root / "pynq_sound_localizer"
            if pynq_jupyter_root.exists()
            else Path.cwd() / "pynq_sound_localizer_notebooks"
        )
    else:
        dest_dir = Path(target_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    if pkg_notebooks_dir.exists():
        for item in pkg_notebooks_dir.glob("*.ipynb"):
            dest_file = dest_dir / item.name
            shutil.copy2(item, dest_file)
            copied_files.append(item.name)

    print(f"[NotebookInstaller] Successfully copied {len(copied_files)} notebooks to:\n                   {dest_dir.resolve()}")
    for f in copied_files:
        print(f"  • {f}")


if __name__ == "__main__":
    copy_notebooks()