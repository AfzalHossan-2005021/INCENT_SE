# INCENT-SE: Spatial Transcriptomics Alignment with SE(2) Transformation Recovery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview
**INCENT-SE** is a robust Python framework for the alignment of Spatial Transcriptomics (ST) data. It integrates biological similarity, spatial contiguity, and rigid-body transformation recovery to solve the complex task of aligning tissue slices. 

The framework is uniquely designed to handle **partial spatial overlaps**, **asymmetric slice coverage**, and **unmatched cellular masses**, ensuring that cross-portion mixing is prevented during alignment. It supports both same-timepoint alignments (e.g., adjacent serial slices) and cross-timepoint spatiotemporal alignments (e.g., developmental stages).

## Key Innovations

- **Robust to Asymmetric Coverage**: Engineered specifically for bilateral slices where portions can be asymmetric in coverage (e.g., one side full, the other side partial / 80%).
- **Partial Overlap & Unmatched Mass**: Alignment models do NOT assume equal size, equal mass, or complete overlap across portions. Optimal transport (OT) constraints carefully support partial overlap without forcing artificial cross-portion mixing.
- **Explicit SE(2) Recovery ((2)$-OT)**: Jointly recovers explicit optimal rotation and translation while establishing soft cell-to-cell correspondences.
- **Cross-Timepoint Latent Embeddings (cVAE)**: Uses a Conditional Variational Autoencoder (cVAE) to factor out temporal/developmental variation, enabling biologically meaningful alignments across different timepoints.
- **Topology & Contiguity**: Employs spatial affinity graphs, cell neighborhood structures, and structural fingerprints to ensure anatomically consistent mappings.

## Package Architecture

The package is modular and organized into the \incent\ directory:

| Module | Description |
|--------|-------------|
| \core_se.py\ | Main public API containing the high-level alignment entry points (\pairwise_align_se\, \pairwise_align_spatiotemporal\). |
| \seot.py\ | Core mathematical engine for SE(2)-OT transformation recovery and Expectation-Maximization (EM) alignment. |
| \cvae.py\ | PyTorch-based Conditional Variational Autoencoder for generating cross-timepoint latent cost matrices. |
| \pose.py\ | Pose estimation algorithms, robust Fourier-Mellin initialization, and rigid transformation refinement. |
| \	opology.py\ | Computations for topological fingerprints and structural/subgraph matching. |
| \contiguity.py\ | Functions to capture spatial affinities and contiguity gradients across tissue slices. |
| \utils.py & _seot_support.py\ | Mathematical utilities, Gromov-Wasserstein support tools, and objective initializers. |
| \_gpu.py\ | GPU acceleration utilities for heavy matrix operations. |

## Installation

### Prerequisites
INCENT-SE relies on numeric and machine learning libraries:
* \
umpy\
* \scipy\
* \nndata\
* \pot\ (Python Optimal Transport)
* \	orch\ (PyTorch)

### Setup
Ensure you have the required dependencies and install the package locally:

`ash
# 1. Clone the repository
git clone https://github.com/your-username/INCENT_SE.git
cd INCENT_SE

# 2. Install dependencies
pip install numpy scipy anndata pot torch

# 3. Install the package in editable mode
pip install -e .
`
*(Note: If \setup.py\ or \pyproject.toml\ is absent, ensure the repository root is in your \PYTHONPATH\ or simply run scripts from the root directory).*

## Usage Guide

The primary interface involves two functions: \pairwise_align_se\ and \pairwise_align_spatiotemporal\. Both expect spatial data encapsulated in \AnnData\ objects.

`python
import anndata as ad
from incent import pairwise_align_se, pairwise_align_spatiotemporal

# 1. Load your Spatial Transcriptomics slices
sliceA = ad.read_h5ad("path/to/sliceA.h5ad")
sliceB = ad.read_h5ad("path/to/sliceB.h5ad")

# 2. Perform Same-Timepoint Alignment (adjacent developmental slices)
pi = pairwise_align_se(
    sliceA,
    sliceB,
    alpha=0.5,       # Weight for transcriptomic similarity
    beta=0.5,        # Weight for spatial/structural penalty
    gamma=0.1,       # Entropic regularization / mass relaxation
    radius=50.0,     # Spatial neighborhood radius
    filePath="./results/se_alignment",
    estimate_rotation=True
)

# 3. Perform Cross-Timepoint / Spatiotemporal Alignment
result = pairwise_align_spatiotemporal(
    sliceA,
    sliceB,
    alpha=0.4,
    beta=0.4,
    gamma=0.2,
    radius=50.0,
    filePath="./results/spatiotemporal",
    estimate_rotation=True,
    use_rapa=True     # Enable robust partial alignment features
)
`

### Understanding Outputs
* **\pi\ (Correspondence Matrix)**: A highly-sparse, soft mapping matrix \(N x M)\ indicating the transportation plan of cells from \sliceA\ to \sliceB\.
* Output directories (specified by \ilePath\) will contain logged matrices, transformed coordinates, and alignment scores (objective metrics identifying alignment quality).

## License

This project is licensed under the terms described in the \LICENSE\ file found at the root of the repository.
