# Changelog

All changes to the MFML-4-QC package will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v1.5.0

### New: Automatic Differentiation for Forces

- **`src/autodiff.py`** — Differentiable PyTorch wrappers for trained KRR and MFML models. Enables energy prediction with automatic force derivation via backpropagation (F = -nabla E). Includes `DifferentiableKRR`, `DifferentiableMFML`, and a differentiable `generate_coulomb_matrix()`. Descriptor functions are pluggable callables. Supports Gaussian, Matern, Laplacian, and Wasserstein kernels. Factory methods `from_krr_model()` / `from_mfml_model()` for wrapping trained models.

### New: Vector-Valued Kernel Ridge Regression

- **`src/vector_krr.py`** — Vector-valued KRR for directly learning vector targets (e.g., dipole moments):
  - `VectorKRR`: isotropic kernel K(x,x') = k(x,x') I_d — equivalent to d independent scalar KRRs with a single Cholesky factorisation.
  - `CoupledVectorKRR`: learnable d x d PSD output covariance T, optimised via Nelder-Mead on validation MSE (6 free parameters for d=3).
  - `DifferentiableVectorKRR`: frozen PyTorch evaluator for inference pipelines.

### New: Vector-Valued Multi-Fidelity Machine Learning

- **`src/vector_mfml.py`** — `VectorMFML` class for multi-fidelity learning of vector-valued properties using SGCT on d-dimensional vectors. Wraps any duck-typed vector estimator. Includes `DifferentiableVectorMFML` for differentiable inference.

### New: Rotate-Predict-Rotate (RPR) Frame Canonicalisation

- **`src/rpr.py`** — Per-geometry canonical frame from inertia tensor principal axes with cubic-moment sign disambiguation. `RPRVectorKRR` is a composable wrapper compatible with any `.train()/.predict()` model. Handles degeneracies (linear molecules, symmetric tops). Pure NumPy, no PyTorch dependency.

### New: Dataset — QeMFi Nitrophenol

- **`src/datasets.py`** — `load_nitrophenol_data()` loader for the QeMFi nitrophenol dataset (15,000 geometries, 5 SCF fidelities).

### New Examples

| # | File | Description |
|---|------|-------------|
| 09 | `autodiff_forces.py` | Energy to force prediction via backpropagation on QeMFi nitrophenol |
| 10 | `vector_krr_dipole.py` | Vector KRR for dipole moments: isotropic vs coupled T |
| 11 | `rpr_vector_krr.py` | 4-variant comparison: {Isotropic, Coupled} x {with/without RPR} |
| 12 | `vector_mfml_dipole.py` | 5-fidelity vector MFML vs 3 independent scalar MFML models |
| 13 | `learning_curve_ablation.py` | Full 8-variant ablation (RPR x T x MFML) with learning curves |

### Design Notes

- **No breaking changes** — all existing modules (`krr.py`, `mfml.py`, `representations.py`, `utils.py`) are unchanged.
- **PyTorch is optional** — only `autodiff.py` and `Differentiable*` classes require it. VectorKRR, VectorMFML, and RPR work with pure NumPy/SciPy.
- **Duck-type compatibility** — all new estimators expose `.train(X, Y)` / `.predict(X)` for use as `ModelMFML` base estimators.
