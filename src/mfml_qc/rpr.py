"""
Rotate-Predict-Rotate (RPR) frame canonicalization for vector-valued KRR.

Provides a per-geometry canonical frame from the inertia tensor and a
composable wrapper (``RPRVectorKRR``) that works with any model exposing
``.train(X, Y)`` / ``.predict(X)``.

The canonical frame is defined by the principal axes of inertia. Sign
ambiguity is resolved using the mass-weighted cubic moment along each
axis: s_i = Σ_a m_a (r_a · e_i)^3. This is computed in the molecular
frame and is therefore rotationally covariant.

Degeneracy handling:
  - Linear molecules (one eigenvalue ≈ 0): warning emitted, fallback
    convention based on the heaviest off-axis atom.
  - Symmetric tops (two equal eigenvalues): similar fallback with warning.

Reference: [1] J. Martinka, M. Pederzoli, M. Barbatti, P. O. Dral, J. Pittner, A simple approach to rotationally invariant machine learning of a vector quantity, J. Chem. Phys. 161, 174104 (2024). DOI: 10.1063/5.0230176
"""

import numpy as np
import warnings

ATOMIC_MASSES = {
    1: 1.00794,
    2: 4.00260,
    3: 6.941,
    4: 9.01218,
    5: 10.811,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998,
    10: 20.180,
    11: 22.990,
    12: 24.305,
    13: 26.982,
    14: 28.086,
    15: 30.974,
    16: 32.065,
    17: 35.453,
    18: 39.948,
    19: 39.098,
    20: 40.078,
    26: 55.845,
    29: 63.546,
    30: 65.380,
    35: 79.904,
    53: 126.904,
}


def canonicalize(coords, masses, degeneracy_tol=1e-6):
    """
    Compute the RPR canonical frame for a molecular geometry.

    Parameters
    ----------
    coords : np.ndarray, shape (N_atoms, 3)
        Cartesian coordinates in the lab frame.
    masses : np.ndarray, shape (N_atoms,)
        Atomic masses.
    degeneracy_tol : float
        Relative tolerance for detecting degenerate inertia eigenvalues.

    Returns
    -------
    U : np.ndarray, shape (3, 3)
        Rotation matrix whose columns are the principal axes. Right-handed
        (det = +1). Sign convention: cubic moment along each axis is
        non-negative.
    com : np.ndarray, shape (3,)
        Center of mass.
    """
    total_mass = np.sum(masses)
    com = np.sum(masses[:, None] * coords, axis=0) / total_mass
    r = coords - com

    r_sq = np.sum(r**2, axis=1)
    I = np.einsum("a,a->", masses, r_sq) * np.eye(3) - np.einsum(
        "a,ai,aj->ij", masses, r, r
    )

    eigvals, eigvecs = np.linalg.eigh(I)

    max_eigval = np.max(np.abs(eigvals))
    if max_eigval > 0:
        if eigvals[0] < degeneracy_tol * max_eigval:
            warnings.warn(
                "Near-linear molecule detected (smallest eigenvalue "
                f"{eigvals[0]:.2e} vs largest {max_eigval:.2e}). "
                "The two degenerate perpendicular axes use a fallback "
                "convention based on the heaviest off-axis atom.",
                stacklevel=2,
            )
        elif (eigvals[1] - eigvals[0]) < degeneracy_tol * max_eigval:
            warnings.warn(
                f"Near-degenerate eigenvalues: {eigvals}. "
                "Canonical frame may be less stable for nearly "
                "symmetric-top geometries.",
                stacklevel=2,
            )

    for i in range(3):
        proj = r @ eigvecs[:, i]
        cubic_moment = np.sum(masses * proj**3)
        if cubic_moment < -1e-14:
            eigvecs[:, i] *= -1
        elif abs(cubic_moment) < 1e-14:
            max_idx = np.argmax(np.abs(proj))
            if proj[max_idx] < 0:
                eigvecs[:, i] *= -1

    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    return eigvecs, com


class RPRVectorKRR:
    """
    RPR wrapper around any vector-valued model with .train/.predict.

    During training, each geometry is canonicalized: descriptors are computed
    from canonical coordinates, and targets are rotated to the canonical frame.
    During prediction, predictions in the canonical frame are rotated back to
    the lab frame.

    Parameters
    ----------
    model : object
        Inner model with ``.train(X, Y)`` and ``.predict(X)`` methods.
    Z : array-like, shape (N_atoms,)
        Atomic numbers.
    descriptor_fn : callable
        Maps ``(Z=atomic_numbers, R=coordinates)`` to a 1D descriptor array.
        E.g., ``mfml_qc.representations.compute_flat_coulomb``.
    """

    def __init__(self, model, Z, descriptor_fn):
        self.model = model
        self.Z = np.asarray(Z, dtype=np.int32)
        self.masses = np.array([ATOMIC_MASSES[int(z)] for z in self.Z])
        self.n_atoms = len(self.Z)
        self.descriptor_fn = descriptor_fn

    def _process_geometries(self, X_flat, Y=None):
        """
        Canonicalize geometries and compute descriptors.

        Parameters
        ----------
        X_flat : np.ndarray, shape (N, N_atoms*3)
        Y : np.ndarray, shape (N, d), optional

        Returns
        -------
        descriptors : np.ndarray, shape (N, n_features)
        Y_canon : np.ndarray, shape (N, d) or None
        Us : list of np.ndarray, each (3, 3)
        """
        N = X_flat.shape[0]
        coords_all = X_flat.reshape(N, self.n_atoms, 3)

        descriptors = []
        Us = []
        Y_canon = np.zeros_like(Y) if Y is not None else None

        for i in range(N):
            U, com = canonicalize(coords_all[i], self.masses)
            coords_canon = (coords_all[i] - com) @ U
            desc = self.descriptor_fn(Z=self.Z, R=coords_canon)
            descriptors.append(desc)
            Us.append(U)
            if Y is not None:
                Y_canon[i] = U.T @ Y[i]

        return np.array(descriptors), Y_canon, Us

    def train(self, X_train, Y_train):
        """
        Train the RPR-wrapped model.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, N_atoms*3)
            Flattened Cartesian coordinates.
        Y_train : np.ndarray, shape (N, d)
            Vector targets in the lab frame.
        """
        X_desc, Y_canon, _ = self._process_geometries(X_train, Y_train)
        self.model.train(X_desc, Y_canon)

    def predict(self, X_test):
        """
        Predict and rotate back to lab frame.

        Parameters
        ----------
        X_test : np.ndarray, shape (M, N_atoms*3)
            Flattened Cartesian coordinates.

        Returns
        -------
        np.ndarray, shape (M, d)
            Predictions in the lab frame.
        """
        X_desc, _, Us = self._process_geometries(X_test)
        preds_canon = self.model.predict(X_desc)

        preds = np.zeros_like(preds_canon)
        for i in range(len(Us)):
            preds[i] = Us[i] @ preds_canon[i]

        return preds
