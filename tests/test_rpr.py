"""
Unit tests for RPR canonicalization and covariance.

Tests:
  (a) Sign-convention determinism on repeated calls
  (b) Canonical coordinates identical after rotation
  (c) Graceful handling of linear molecules
  (d) End-to-end rotational covariance of RPR predictions
"""

import numpy as np
import warnings
from scipy.spatial.transform import Rotation

from mfml_qc.representations import compute_flat_coulomb
from mfml_qc.rpr import canonicalize, RPRVectorKRR
from mfml_qc.vector_krr import VectorKRR


def test_sign_convention_determinism():
    rng = np.random.RandomState(123)
    coords = rng.randn(5, 3)
    masses = np.array([12.0, 1.0, 1.0, 16.0, 14.0])

    U1, com1 = canonicalize(coords, masses)
    U2, com2 = canonicalize(coords, masses)

    assert np.allclose(U1, U2, atol=1e-14)
    assert np.allclose(com1, com2, atol=1e-14)


def test_sign_convention_under_rotation():
    rng = np.random.RandomState(456)
    coords = rng.randn(6, 3) * 2
    masses = np.array([12.0, 16.0, 1.0, 1.0, 14.0, 1.0])

    R = Rotation.random(random_state=42).as_matrix()
    coords_rot = coords @ R.T

    U_orig, com_orig = canonicalize(coords, masses)
    U_rot, com_rot = canonicalize(coords_rot, masses)

    r_canon_orig = (coords - com_orig) @ U_orig
    r_canon_rot = (coords_rot - com_rot) @ U_rot

    assert np.allclose(r_canon_orig, r_canon_rot, atol=1e-10)
    assert np.allclose(U_rot, R @ U_orig, atol=1e-10)


def test_linear_molecule():
    coords = np.array([[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]])
    masses = np.array([16.0, 12.0, 16.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        U, com = canonicalize(coords, masses)
        assert len(w) >= 1 and "linear" in str(w[0].message).lower()

    assert np.isclose(np.linalg.det(U), 1.0, atol=1e-10)


def test_covariance_end_to_end():
    try:
        from mfml_qc.datasets import load_nitrophenol_data

        try:
            dataset = load_nitrophenol_data()
        except FileNotFoundError:
            import os

            alt = os.path.expanduser("~/MFML/data/nitrophenol")
            dataset = load_nitrophenol_data(data_dir=alt)
    except (ImportError, FileNotFoundError):
        import pytest

        pytest.skip("QeMFi dataset not available")
        return

    R_all = dataset["R"]
    Z = dataset["Z"]
    DPe = dataset["DPe"][:, -1, :]

    N_TRAIN, N_TEST = 200, 5
    R_train, R_test = R_all[:N_TRAIN], R_all[N_TRAIN : N_TRAIN + N_TEST]
    Y_train = DPe[:N_TRAIN]
    Y_mean = np.mean(Y_train, axis=0)

    inner = VectorKRR(kernel_type="gaussian", sigma=250.0, reg=1e-10)
    model = RPRVectorKRR(inner, Z, compute_flat_coulomb)
    model.train(R_train.reshape(N_TRAIN, -1), Y_train - Y_mean)

    preds_orig = model.predict(R_test.reshape(N_TEST, -1))

    rot = Rotation.random(random_state=99).as_matrix()
    R_test_rot = np.array([r @ rot.T for r in R_test])
    preds_rot = model.predict(R_test_rot.reshape(N_TEST, -1))

    max_diff = max(
        np.max(np.abs(preds_rot[i] - rot @ preds_orig[i])) for i in range(N_TEST)
    )
    assert max_diff < 1e-9, f"Covariance violated! Max diff = {max_diff}"


if __name__ == "__main__":
    test_sign_convention_determinism()
    test_sign_convention_under_rotation()
    test_linear_molecule()
    test_covariance_end_to_end()
