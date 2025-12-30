"""Setup script for essay-ml."""

from setuptools import find_packages, setup

setup(
    name="essay-ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "typer[all]>=0.12.0",
        "rich>=13.7.0",
        "scikit-learn>=1.4.0",
        "nltk>=3.8.1",
        "spacy>=3.7.2",
        "pandas>=2.2.0",
        "numpy>=1.26.3",
        "joblib>=1.3.2",
        "pydantic>=2.5.3",
    ],
    entry_points={
        "console_scripts": [
            "essay-ml=src.cli:app",
        ],
    },
    python_requires=">=3.11",
)
