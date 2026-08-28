"""
pynq_sound_localizer: FPGA-Accelerated Acoustic Kinematics, Doppler Tracking & Sound Localization.
"""

from pynq_localizer.loader import HardwareLoader
from pynq_localizer.hw_trigger import HardwareTrigger
from pynq_localizer.spectral_mask import SpectralMaskDriver
from pynq_localizer.array import MicrophoneArrayOverlay
from pynq_localizer.kinematics import KinematicAnalytics
from pynq_localizer.kinematics_dashboard import KinematicsDashboard

__version__ = "1.0.0"
__all__ = [
    "HardwareLoader",
    "HardwareTrigger",
    "SpectralMaskDriver",
    "MicrophoneArrayOverlay",
    "KinematicAnalytics",
    "KinematicsDashboard",
]