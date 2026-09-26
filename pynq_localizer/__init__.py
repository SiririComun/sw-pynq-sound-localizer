"""
pynq_sound_localizer: FPGA-Accelerated Acoustic Kinematics, Doppler Tracking & Sound Localization.
"""

from pynq_localizer.kinematics import (
    KinematicAnalytics,
    MultiSourceTracker,
    AcousticProfile,
    DistanceEstimator,
    AcousticCalibrationProtocol,
    MultipathCalibrationProtocol,
    AngleOfArrivalEstimator,
    DifferentialDopplerTracker,
    TimeOfArrivalEstimator,
)
from pynq_localizer.notebooks import install_localizer_notebooks, copy_notebooks

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

__version__ = "1.3.0"
__all__ = [
    "HardwareLoader",
    "HardwareTrigger",
    "MicrophoneArrayOverlay",
    "KinematicAnalytics",
    "MultiSourceTracker",
    "AcousticProfile",
    "DistanceEstimator",
    "AcousticCalibrationProtocol",
    "MultipathCalibrationProtocol",
    "AngleOfArrivalEstimator",
    "DifferentialDopplerTracker",
    "TimeOfArrivalEstimator",
    "KinematicsDashboard",
    "install_localizer_notebooks",
    "copy_notebooks",
]