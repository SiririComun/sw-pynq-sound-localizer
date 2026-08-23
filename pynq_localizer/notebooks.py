import os
import shutil
from pathlib import Path


def copy_notebooks(target_dir: str = None):
    """
    Copies pynq_sound_localizer example notebooks into the Jupyter root folder
    under a dedicated 'pynq_sound_localizer/' subfolder.
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

    copied_files = []
    if src_notebooks.exists():
        for item in src_notebooks.glob("*.ipynb"):
            dest_file = dest_dir / item.name
            shutil.copy2(item, dest_file)
            copied_files.append(item.name)

    print(f"[NotebookInstaller] Successfully copied {len(copied_files)} notebooks to:")
    print(f"                   {dest_dir.resolve()}")
    for f in copied_files:
        print(f"  • {f}")


if __name__ == "__main__":
    copy_notebooks()
