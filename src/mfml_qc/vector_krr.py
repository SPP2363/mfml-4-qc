"""
Vector-valued Kernel Ridge Regression.

Provides two estimators for learning vector-valued targets (e.g., dipole moments):

- ``VectorKRR``: Isotropic kernel K(x,x') = k(x,x') I_d. Equivalent to d
  independent scalar KRRs with a shared regularisation parameter. Uses a single
  Cholesky factorisation solved against all d right-hand sides.

- ``CoupledVectorKRR``: Separable kernel K(x,x') = k(x,x') T, where T is a
  d×d positive semidefinite output covariance matrix, learnable via Nelder-Mead
  on validation MSE. T is parameterised as L L^T (Cholesky factor, d(d+1)/2
  free parameters). Efficient solve via eigendecomposition of the scalar kernel.

Both classes expose ``.train(X, Y)`` / ``.predict(X)`` for duck-type
compatibility with ``ModelMFML``'s ``base_estimator`` interface.

Also provides ``DifferentiableVectorKRR``, a frozen PyTorch evaluator for
inference in differentiable pipelines.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .krr import (
    gaussian_kernel_symmetric,
    gaussian_kernel_asymmetric,
    matern_kernel_symmetric,
    matern_kernel_asymmetric,
    laplacian_kernel_symmetric,
    laplacian_kernel_asymmetric,
    wasserstein_kernel_symmetric,
    wasserstein_kernel_asymmetric,
)


def _generate_kernel(X_train, X_test, kernel_type, sigma, nu, p, q):
    """Shared kernel dispatch for both VectorKRR and CoupledVectorKRR."""
    if kernel_type == "gaussian":
        if X_test is None:
            return gaussian_kernel_symmetric(X_train, sigma=sigma)
        return gaussian_kernel_asymmetric(X_train, X_test, sigma=sigma)
    elif kernel_type == "matern":
        if X_test is None:
            return matern_kernel_symmetric(X_train, sigma=sigma, nu=nu)
        return matern_kernel_asymmetric(X_train, X_test, sigma=sigma, nu=nu)
    elif kernel_type == "laplacian":
        if X_test is None:
            return laplacian_kernel_symmetric(X_train, sigma=sigma)
        return laplacian_kernel_asymmetric(X_train, X_test, sigma=sigma)
    elif kernel_type == "wasserstein":
        if X_test is None:
            return wasserstein_kernel_symmetric(X_train, sigma=sigma, p=p, q=q)
        return wasserstein_kernel_asymmetric(X_train, X_test, sigma=sigma, p=p, q=q)
    else:
        raise ValueError(f"Unsupported kernel type: '{kernel_type}'")


# ============================================================
# Isotropic VectorKRR
# ============================================================


class VectorKRR:
    """
    Vector-valued KRR with isotropic separable kernel k(x,x') I_d.

    Parameters
    ----------
    kernel_type : str
        One of 'gaussian', 'matern', 'laplacian', 'wasserstein'.
    sigma : float
        Kernel width parameter.
    nu : float
        Matern smoothness parameter (0.5, 1.5, or 2.5).
    p : float
        Wasserstein power parameter.
    q : float
        Wasserstein outer exponent.
    reg : float
        Tikhonov regularisation parameter.
    """

    def __init__(
        self, kernel_type="gaussian", sigma=100.0, nu=1.5, p=1.0, q=1.0, reg=1e-10
    ):
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.nu = nu
        self.p = p
        self.q = q
        self.reg = reg
        self.alphas = None
        self.X_train = None

    def _kernel(self, X_train, X_test=None):
        return _generate_kernel(
            X_train, X_test, self.kernel_type, self.sigma, self.nu, self.p, self.q
        )

    def train(self, X_train, Y_train):
        """
        Solve the vector KRR system.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, D)
            Training feature matrix.
        Y_train : np.ndarray, shape (N, d) or (N,)
            Vector targets. If 1D, treated as d=1.
        """
        self.X_train = np.copy(X_train)
        if Y_train.ndim == 1:
            Y_train = Y_train[:, np.newaxis]

        K = self._kernel(self.X_train)
        K[np.diag_indices_from(K)] += self.reg
        cho = cho_factor(K)
        self.alphas = cho_solve(cho, Y_train)

    def predict(self, X_test):
        """
        Predict vector targets for test features.

        Parameters
        ----------
        X_test : np.ndarray, shape (M, D)

        Returns
        -------
        np.ndarray, shape (M, d)
        """
        if self.alphas is None:
            raise ValueError("Model not trained. Call .train() first.")
        K_test = self._kernel(self.X_train, X_test)
        return K_test @ self.alphas


# ============================================================
# Coupled VectorKRR with learnable T
# ============================================================


def _L_to_T(L_flat, d):
    """Reconstruct T = L L^T from flat lower-triangular parameters."""
    L = np.zeros((d, d))
    L[np.tril_indices(d)] = L_flat
    return L @ L.T


def _solve_coupled(eigvals, U, Y, T, reg):
    """
    Solve (K ⊗ T + reg I) α = y using eigendecomposition of K.

    Parameters
    ----------
    eigvals : np.ndarray, shape (N,)
    U : np.ndarray, shape (N, N)
    Y : np.ndarray, shape (N, d)
    T : np.ndarray, shape (d, d)
    reg : float

    Returns
    -------
    np.ndarray, shape (N, d)
    """
    N, d = Y.shape
    Z = U.T @ Y
    beta = np.zeros_like(Z)
    Id = np.eye(d)
    for i in range(N):
        A = eigvals[i] * T + reg * Id
        beta[i] = np.linalg.solve(A, Z[i])
    return U @ beta


class CoupledVectorKRR:
    """
    Vector KRR with learnable output covariance T = L L^T.

    Training procedure:
      1. Split data into train/val
      2. Eigendecompose training kernel (computed once)
      3. Optimise T via Nelder-Mead on validation MSE
      4. Re-solve on full training data with optimal T

    Parameters
    ----------
    kernel_type : str
        One of 'gaussian', 'matern', 'laplacian', 'wasserstein'.
    sigma : float
        Kernel width parameter.
    nu : float
        Matern smoothness parameter.
    p : float
        Wasserstein power parameter.
    q : float
        Wasserstein outer exponent.
    reg : float
        Tikhonov regularisation.
    val_fraction : float
        Fraction of training data held out for T optimisation.
    max_opt_iter : int
        Maximum Nelder-Mead iterations for T optimisation.
    """

    def __init__(
        self,
        kernel_type="gaussian",
        sigma=100.0,
        nu=1.5,
        p=1.0,
        q=1.0,
        reg=1e-10,
        val_fraction=0.15,
        max_opt_iter=200,
    ):
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.nu = nu
        self.p = p
        self.q = q
        self.reg = reg
        self.val_fraction = val_fraction
        self.max_opt_iter = max_opt_iter

        self.alphas = None
        self.X_train = None
        self.T = None

    def _kernel(self, X_train, X_test=None):
        return _generate_kernel(
            X_train, X_test, self.kernel_type, self.sigma, self.nu, self.p, self.q
        )

    def train(self, X_train, Y_train):
        """
        Train with T optimisation.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, D)
        Y_train : np.ndarray, shape (N, d) or (N,)
        """
        if Y_train.ndim == 1:
            Y_train = Y_train[:, np.newaxis]

        N, d = Y_train.shape

        n_val = max(1, int(N * self.val_fraction))
        n_tr = N - n_val
        perm = np.random.RandomState(42).permutation(N)
        tr_idx, val_idx = perm[:n_tr], perm[n_tr:]

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        Y_tr, Y_val = Y_train[tr_idx], Y_train[val_idx]

        K_tr = self._kernel(X_tr)
        eigvals, U = np.linalg.eigh(K_tr)
        eigvals = np.maximum(eigvals, 0.0)

        K_val = self._kernel(X_tr, X_val)

        def objective(L_flat):
            T = _L_to_T(L_flat, d)
            alphas = _solve_coupled(eigvals, U, Y_tr, T, self.reg)
            preds = K_val @ alphas
            return np.mean((preds - Y_val) ** 2)

        L0 = np.eye(d)[np.tril_indices(d)]
        result = minimize(
            objective,
            L0,
            method="Nelder-Mead",
            options={"maxiter": self.max_opt_iter, "xatol": 1e-6, "fatol": 1e-10},
        )

        T_opt = _L_to_T(result.x, d)
        self.T = T_opt

        self.X_train = np.copy(X_train)
        K_full = self._kernel(self.X_train)
        eigvals_full, U_full = np.linalg.eigh(K_full)
        eigvals_full = np.maximum(eigvals_full, 0.0)
        self.alphas = _solve_coupled(eigvals_full, U_full, Y_train, T_opt, self.reg)

    def predict(self, X_test):
        """
        Predict vector targets.

        Parameters
        ----------
        X_test : np.ndarray, shape (M, D)

        Returns
        -------
        np.ndarray, shape (M, d)
        """
        if self.alphas is None:
            raise ValueError("Model not trained. Call .train() first.")
        K_test = self._kernel(self.X_train, X_test)
        return K_test @ self.alphas


# ============================================================
# Differentiable evaluator (requires PyTorch)
# ============================================================

try:
    import torch

    class DifferentiableVectorKRR(torch.nn.Module):
        """
        Frozen PyTorch evaluator for vector-valued KRR.

        Pre-solved alphas are stored as buffers (not learnable parameters).
        Useful as a building block in differentiable pipelines.

        Parameters
        ----------
        X_train_np : np.ndarray, shape (N, D)
        alphas_np : np.ndarray, shape (N, d)
        kernel_type : str
        sigma : float
        nu : float
        p : float
        q : float
        """

        def __init__(
            self,
            X_train_np,
            alphas_np,
            kernel_type="gaussian",
            sigma=100.0,
            nu=1.5,
            p=1.0,
            q=1.0,
        ):
            super().__init__()
            self.register_buffer(
                "X_train", torch.tensor(X_train_np, dtype=torch.float64)
            )
            self.register_buffer("alphas", torch.tensor(alphas_np, dtype=torch.float64))
            self.kernel_type = kernel_type
            self.sigma = sigma
            self.nu = nu
            self.p = p
            self.q = q

        @classmethod
        def from_vector_krr(cls, model):
            """Construct from a trained VectorKRR or CoupledVectorKRR."""
            return cls(
                X_train_np=model.X_train,
                alphas_np=model.alphas,
                kernel_type=model.kernel_type,
                sigma=model.sigma,
                nu=getattr(model, "nu", 1.5),
                p=getattr(model, "p", 1.0),
                q=getattr(model, "q", 1.0),
            )

        def _compute_kernel(self, X_test):
            if self.kernel_type == "gaussian":
                gamma = 1.0 / (2.0 * self.sigma**2)
                diffs = X_test.unsqueeze(1) - self.X_train.unsqueeze(0)
                return torch.exp(-gamma * torch.sum(diffs**2, dim=-1))
            elif self.kernel_type == "matern":
                diffs = X_test.unsqueeze(1) - self.X_train.unsqueeze(0)
                dists = torch.norm(diffs, dim=-1)
                if self.nu == 0.5:
                    return torch.exp(-dists / self.sigma)
                elif self.nu == 1.5:
                    scaled = np.sqrt(3.0) * dists / self.sigma
                    return torch.exp(-scaled) * (1.0 + scaled)
                elif self.nu == 2.5:
                    scaled = np.sqrt(5.0) * dists / self.sigma
                    return torch.exp(-scaled) * (1.0 + scaled + scaled**2 / 3.0)
                else:
                    raise ValueError(f"Only nu=0.5, 1.5, 2.5 supported, got {self.nu}")
            elif self.kernel_type == "laplacian":
                diffs = X_test.unsqueeze(1) - self.X_train.unsqueeze(0)
                return torch.exp(-torch.sum(torch.abs(diffs), dim=-1) / self.sigma)
            elif self.kernel_type == "wasserstein":
                X_test_s = torch.sort(X_test, dim=-1).values
                X_train_s = torch.sort(self.X_train, dim=-1).values
                diffs = X_test_s.unsqueeze(1) - X_train_s.unsqueeze(0)
                n_feat = X_test.shape[-1]
                w_p = (torch.sum(torch.abs(diffs) ** self.p, dim=-1) / n_feat) ** (
                    1.0 / self.p
                )
                return torch.exp(-(w_p**self.q) / self.sigma)
            else:
                raise ValueError(f"Unsupported kernel type: '{self.kernel_type}'")

        def forward(self, X_test):
            """
            Parameters
            ----------
            X_test : torch.Tensor, shape (1, D)

            Returns
            -------
            torch.Tensor, shape (d,)
            """
            k = self._compute_kernel(X_test)
            return (k @ self.alphas).squeeze(0)

except ImportError:
    pass
