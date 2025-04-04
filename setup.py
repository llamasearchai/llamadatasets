#!/usr/bin/env python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llamadatasets-llamasearch",
    version="0.1.0",
    author="LlamaSearch AI",
    author_email="nikjois@llamasearch.ai",
    description="Dataset management and processing library for LlamaSearch.ai applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://llamasearch.ai",
    project_urls={
        "Bug Tracker": "https://github.com/llamasearch/llamadatasets/issues",
        "Documentation": "https://docs.llamasearch.ai/llamadatasets",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "pyarrow>=5.0.0",
        "tqdm>=4.61.0",
        "pydantic>=1.8.0",
        "sqlalchemy>=1.4.0",
        "scikit-learn>=1.0.0",
        "fsspec>=2021.10.0",
        "smart-open>=5.0.0",
        "typing-extensions>=3.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.2",
            "mypy>=0.812",
            "isort>=5.9.1",
            "tox>=3.24.0",
        ],
        "nlp": [
            "nltk>=3.6.0",
            "spacy>=3.0.0",
            "transformers>=4.5.0",
        ],
        "images": [
            "pillow>=8.2.0",
            "torchvision>=0.10.0",
        ],
        "docs": [
            "sphinx>=4.0.2",
            "sphinx-rtd-theme>=0.5.2",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) # Version bump for first release
# Version bump for first release
