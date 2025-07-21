#!/usr/bin/env python
"""Setup script for Cancer Alpha package."""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from a file."""
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="cancer-alpha",
    version="1.0.0",
    author="Cancer Alpha Research Team",
    author_email="research@cancer-alpha.org",
    description="Multi-Modal Transformer Architecture for Cancer Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rstil2/cancer-alpha",
    project_urls={
        "Bug Tracker": "https://github.com/rstil2/cancer-alpha/issues",
        "Documentation": "https://github.com/rstil2/cancer-alpha/docs",
        "Source Code": "https://github.com/rstil2/cancer-alpha",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "jupyter>=1.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "cancer-alpha-train=cancer_alpha.scripts.train:main",
            "cancer-alpha-evaluate=cancer_alpha.scripts.evaluate:main",
            "cancer-alpha-predict=cancer_alpha.scripts.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cancer_alpha": [
            "data/configs/*.yaml",
            "data/configs/*.json",
        ],
    },
    keywords="cancer classification machine-learning transformers genomics bioinformatics",
)
