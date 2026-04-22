from setuptools import setup

setup(
    name="core-lip",
    version="0.1.0",
    description="COformational Representation Ensemble for LIP prediction",
    packages=["core_lip"],  # The actual folder name
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "h5py>=3.9",
        "matplotlib>=3.7",
        "tqdm>=4.65",
    ],
)
