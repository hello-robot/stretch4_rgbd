from setuptools import setup, find_packages

setup(
    name="stretch4_emulated_rgbd",
    version="0.1.0",
    description="Package for emulated RGB-D imagery alignment on Stretch 4.",
    packages=find_packages(),
    install_requires=[
        "numpy>=2",
        "matplotlib>=3.8",
        "opencv-contrib-python",
        "cma",
        "scipy",
        "rerun-sdk>=0.31.4",
        "pyzmq",
    ],
    python_requires=">=3.12",
)
