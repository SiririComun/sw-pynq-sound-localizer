"""
pynq_sound_localizer: FPGA-Accelerated Acoustic Kinematics, Doppler Tracking & Sound Localization.
"""

from pynq_localizer.loader import HardwareLoader
from pynq_localizer.hw_trigger import HardwareTrigger
from pynq_localizer.array import MicrophoneArrayOverlay
from pynq_localizer.notebooks import copy_notebooks

__version__ = "1.0.0"
__all__ = [
    "HardwareLoader",
    "HardwareTrigger",
    "MicrophoneArrayOverlay",
    "copy_notebooks",
]