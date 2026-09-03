"""
pynq_sound_localizer: FPGA-Accelerated Acoustic Kinematics, Doppler Tracking & Sound Localization.
"""

from pynq_localizer.kinematics import (
    KinematicAnalytics,
    MultiSourceTracker,
    AcousticProfile,
    DistanceEstimator,
    AcousticCalibrationProtocol,
)
from pynq_localizer.notebooks import copy_notebooks

try:
    from pynq_localizer.loader import HardwareLoader
    from pynq_localizer.hw_trigger import HardwareTrigger
    from pynq_localizer.array import MicrophoneArrayOverlay
    from pynq_localizer.kinematics_dashboard import KinematicsDashboard
    _HAS_PYNQ = True
except (ImportError, ModuleNotFoundError):
    HardwareLoader = None
    HardwareTrigger = None
    MicrophoneArrayOverlay = None
    KinematicsDashboard = None
    _HAS_PYNQ = False

__version__ = "1.0.0"
__all__ = [
    "HardwareLoader",
    "HardwareTrigger",
    "MicrophoneArrayOverlay",
    "KinematicAnalytics",
    "MultiSourceTracker",
    "AcousticProfile",
    "DistanceEstimator",
    "AcousticCalibrationProtocol",
    "KinematicsDashboard",
    "copy_notebooks",
]