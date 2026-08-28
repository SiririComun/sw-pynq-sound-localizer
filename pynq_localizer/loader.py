"""
pynq_localizer.loader: Smart Hardware Overlay Downloader & Binary Manager.
Pulls target bitstream and handoff metadata from GitHub Releases based on hardware.json.
"""

import os
import json
import urllib.request
from pathlib import Path
try:
    from pynq import Overlay
except ImportError:
    Overlay = object

class HardwareLoader:
    """
    Smart Overlay Loader that detects the target board and pulls matching
    .bit and .hwh release binaries from GitHub Releases API based on hardware.json.
    """

    @staticmethod
    def get_project_root() -> Path:
        """Find the root directory of the package where hardware.json lives."""
        return Path(__file__).resolve().parent.parent

    @classmethod
    def get_hardware_config(cls) -> dict:
        """Load hardware pinning configuration from hardware.json."""
        config_path = cls.get_project_root() / "hardware.json"
        if not config_path.exists():
            return {
                "repo": "SiririComun/hw-xadc-dma-overlays",
                "version": "v1.5.0"
            }
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def get_board_name() -> str:
        """
        Detect active board running PYNQ across all PYNQ OS versions.
        Normalizes 'PYNQ-Z2' -> 'pynq_z2', 'ZedBoard' -> 'zedboard', etc.
        """
        board_env = os.environ.get("BOARD")
        if board_env:
            return board_env.lower().replace("-", "_")

        try:
            if os.path.exists("/etc/board.name"):
                with open("/etc/board.name", "r", encoding="utf-8") as f:
                    return f.read().strip().lower().replace("-", "_")
        except Exception:
            pass

        try:
            from pynq import Device
            if Device.active_device and Device.active_device.name:
                return Device.active_device.name.lower().replace("-", "_")
        except Exception:
            pass

        return "pynq_z2"

    @classmethod
    def get_overlay_path(cls, version: str = None, download_dir: str = None) -> Path:
        """
        Detects host board, downloads matching .bit and .hwh if missing,
        and returns the local Path to the .bit file.
        """
        config = cls.get_hardware_config()
        repo = config.get("repo", "SiririComun/hw-xadc-dma-overlays")
        target_version = version or config.get("version", "v1.6.0")

        board_name = cls.get_board_name()
        bit_filename = f"{board_name}.bit"
        hwh_filename = f"{board_name}.hwh"

        if download_dir is None:
            download_dir = Path.home() / ".cache" / "pynq_sound_localizer" / target_version
        else:
            download_dir = Path(download_dir)

        download_dir.mkdir(parents=True, exist_ok=True)

        local_bit = download_dir / bit_filename
        local_hwh = download_dir / hwh_filename

        base_url = f"https://github.com/{repo}/releases/download/{target_version}/"
        url_bit = f"{base_url}{bit_filename}"
        url_hwh = f"{base_url}{hwh_filename}"

        if not local_bit.exists() or not local_hwh.exists():
            print(f"[HardwareLoader] Target board: '{board_name}'")
            print(f"[HardwareLoader] Fetching overlay '{target_version}' from {repo}...")
            try:
                urllib.request.urlretrieve(url_bit, local_bit)
                urllib.request.urlretrieve(url_hwh, local_hwh)
                print("[HardwareLoader] Bitstream and handoff metadata downloaded.")
            except Exception as e:
                raise RuntimeError(
                    f"Could not download {bit_filename} from {base_url}. Check internet connection."
                ) from e

        return local_bit

    @classmethod
    def get_hardware_config(cls) -> dict:
        """Load hardware pinning configuration from hardware.json."""
        config_path = cls.get_project_root() / "hardware.json"
        if not config_path.exists():
            return {
                "repo": "SiririComun/hw-xadc-dma-overlays",
                "version": "v1.6.0",
                "overlay_name": "pynq_z2"
            }
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)