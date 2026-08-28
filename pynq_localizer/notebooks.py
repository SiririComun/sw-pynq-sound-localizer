import os
import shutil
from pathlib import Path


def copy_notebooks(target_dir: str = None):
    """
    Copies pynq_sound_localizer example notebooks into the Jupyter root folder.
    """
    pkg_dir = Path(__file__).resolve().parent

    # Candidate sources in priority order:
    candidates = [
        pkg_dir / "notebooks_data",
        pkg_dir.parent / "notebooks",
    ]

    src_notebooks = None
    for cand in candidates:
        if cand.exists() and any(cand.glob("*.ipynb")):
            src_notebooks = cand
            break

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
    if src_notebooks is not None:
        for item in src_notebooks.glob("*.ipynb"):
            dest_file = dest_dir / item.name
            shutil.copy2(item, dest_file)
            copied_files.append(item.name)

    print(f"[NotebookInstaller] Successfully copied {len(copied_files)} notebooks to:\n                   {dest_dir.resolve()}")
    for f in copied_files:
        print(f"  • {f}")


if __name__ == "__main__":
    copy_notebooks()