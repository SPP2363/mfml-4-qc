# Multifidelity Machine Learning for Quantum Chemistry (MFML-4-QC)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

MFML-4-QC is an open-source library that enables multifidelity machine learning for quantum chemical systems. While the multifidelity methods are model-architecture agnostic, this library provides a lightweight, ultra-fast Numba-compiled Kernel Ridge Regression (KRR) setup as the primary architecture. Users can seamlessly integrate their own custom ML models (e.g., from scikit-learn) and directly interface with quantum chemistry engines like ORCA and PySCF for automated data generation and active learning.

## Key Features
* **Ultra-Fast Kernels:** Compute Matern, Gaussian, Laplacian, and Wasserstein kernels efficiently using JIT-compiled C-loops via Numba.
* **In-Memory Representations:** Generate flattened Coulomb Matrices directly from .xyz trajectories without slow disk I/O.
* **Flexible ML Architectures:** Use the built-in KRR or drop in any scikit-learn compatible estimator (e.g., RandomForestRegressor, MLPRegressor).
* **Quantum Chemistry Oracles:** Abstract interfaces to automatically generate inputs, execute runs, and parse outputs from engines like ORCA and PySCF.
* **Automatic Differentiation:** Differentiable PyTorch wrappers for trained KRR and MFML models, enabling automatic force prediction via backpropagation.
* **Vector-Valued Learning:** Vector KRR (isotropic and coupled) and Vector MFML for directly learning vector properties such as dipole moments.
* **RPR Frame Canonicalisation:** Rotate-Predict-Rotate canonicalisation for improved equivariance of vector property predictions.

## Installation (v1.5.0)
**Prerequisites:** Python 3.10 or higher. It is best to install within a fresh `conda` environment to avoid dependency clashes.

### Installing from PyPI
Users can directly install the package from PyPI using the `pip` call.
```bash
# Create and activate a fresh conda environment
conda create -n mfmlenv python=3.10 -y
conda activate mfmlenv

# Install the package
pip install mfml-4-qc
```

### Install Directly From Source
You can also directly download the GitHub repo and install the package from there.
```bash
# Clone the repository
git clone https://github.com/vivinvinod/mfml-qc.git

# Create and activate a fresh conda environment
conda create -n mfmlenv python=3.10 -y
conda activate mfmlenv

# Install the package
pip install .
```


### Additional Dependencies
If you plan to use the built-in `PySCFEngine` oracle, you can install the package with the optional PySCF dependency. (Note: PySCF can be a heavy dependency, which is why it is kept optional).

```bash
pip install .[pyscf]
```

If you plan to use the automatic differentiation features, install with the optional PyTorch dependency:

```bash
pip install .[torch]
```

To run the `ORCAEngine` you will need to install ORCA. See the [official ORCA manual](https://www.faccts.de/docs/orca/6.1/manual/contents/quickstartguide/installation.html) for details on how to do so.

### Developer Installation
If you are beta testing, modifying the source code, or want to build the local documentation, install the package in "editable" mode (`-e`) with the `[dev]` flag. This installs testing tools (`pytest`, `black` etc) and the Sphinx documentation stack:

```bash
pip install -e .[dev]
```

## Documentation and Examples

MFML-4-QC comes with a comprehensive documentation site that includes detailed API references and tutorials. An inbuilt 15 picosecond MD-trajectory of benzene is also provided as an inbuilt dataset for preliminary exploration of the package. 

You can view the documentation at https://vivinvinod.github.io/mfml-4-qc/.
To view the documentation locally, go to `docs/build/html/` and open `index.html` in a browser of your choice. 
You can also browse the raw tutorial scripts directly in the `examples/` directory of the code repository.
