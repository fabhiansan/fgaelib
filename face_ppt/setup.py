from setuptools import find_packages, setup

setup(
    name="fgaelib",
    packages=find_packages(include=["fgaelib"]),
    version="0.1.0",
    description="Face Gender Age Emotion Detection Library for Proyek Penelitian Terapan Tasks",
    author="Fabhianto Maoludyo",
    license="MIT",
    install_requires=["opencv-python", "numpy", "matplotlib"],
    setup_requires=["pytest-runner"],
    test_require=["pytest"],
    test_suite="test"
)