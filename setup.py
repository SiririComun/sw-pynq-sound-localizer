import os
from setuptools import setup, find_packages

long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="pynq-sound-localizer",
    version="1.0.0",
    author="Juan Pablo Sánchez (SiririComun)",
    description="FPGA-Accelerated Acoustic Kinematics, Doppler Tracking & Direction-of-Arrival Sound Localizer for PYNQ-Z2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SiririComun/sw-pynq-sound-localizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "plotly>=5.10.0,<6.0.0",
        "ipywidgets>=8.0.0",
    ],
    package_data={
        "pynq_localizer": ["notebooks_data/*.ipynb", "hardware.json", "../hardware.json"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "pynq-localizer-get-notebooks=pynq_localizer.notebooks:copy_notebooks",
        ],
    },
)
