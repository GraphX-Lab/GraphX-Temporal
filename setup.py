#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Read the contents of README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return requirements
        

setup(
    name="gx-temporal",
    version="0.1.0",
    author="Temporal Graph Learning Team",
    author_email="",
    description="Temporal Graph Learning Library",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/TGL-X/GraphX-Temporal",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    keywords="temporal graph learning, graph neural networks, machine learning, deep learning",
    project_urls={
        "Bug Reports": "https://github.com/TGL-X/GraphX-Temporal/issues",
        "Source": "https://github.com/TGL-X/GraphX-Temporal",
        "Documentation": "https://tgx.readthedocs.io/",
    },
)