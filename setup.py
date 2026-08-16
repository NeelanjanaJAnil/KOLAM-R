from setuptools import setup, find_packages

setup(
    name="kolam_r",
    version="0.1.0",
    description="KOLAM-R: Inverse mathematical reconstruction of structured dotted Kolam patterns",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "Pillow>=10.0",
        "matplotlib>=3.7",
        "pydantic>=2.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)
