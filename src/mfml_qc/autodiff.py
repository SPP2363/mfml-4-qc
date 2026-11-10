"""
Autodifferentiable KRR and MFML evaluators.

Provides differentiable (PyTorch) wrappers around trained KRR and MFML models,
enabling energy prediction with automatic force derivation via backpropagation
(F = -∇E with respect to atomic coordinates).

Requires PyTorch (optional dependency of mfml_qc).
"""

import numpy as np
import torch


def generate_coulomb_matrix(coords, charges):
    """
    Differentiable Unsorted Coulomb Matrix generator.

    Extracts the diagonal and upper triangle into a 1D flat tensor,
    matching the feature order of ``mfml_qc.representations.compute_flat_coulomb``.

    Parameters
    ----------
    coords : torch.Tensor, shape (N_atoms, 3)
        Atomic coordinates (requires_grad=True for force computation).
    charges : torch.Tensor, shape (N_atoms,)
        Nuclear charges.

    Returns
    -------
    torch.Tensor, shape (N_atoms*(N_atoms+1)//2,)
        Flattened upper-triangular Coulomb matrix.
    """
    N = coords.shape[0]
    row_idx, col_idx = torch.triu_indices(N, N, offset=0, device=coords.device)
    is_diag = row_idx == col_idx
    is_off_diag = ~is_diag

    num_features = N * (N + 1) // 2
    cm_flat = torch.zeros(num_features, dtype=coords.dtype, device=coords.device)

    cm_flat[is_diag] = 0.5 * (charges[row_idx[is_diag]] ** 2.4)

    r_i = coords[row_idx[is_off_diag]]
    r_j = coords[col_idx[is_off_diag]]
    dists = torch.norm(r_i - r_j, dim=-1)

    z_i = charges[row_idx[is_off_diag]]
    z_j = charges[col_idx[is_off_diag]]
    cm_flat[is_off_diag] = (z_i * z_j) / dists

    return cm_flat


class DifferentiableKRR(torch.nn.Module):
    """
    Descriptor-agnostic differentiable KRR evaluator.

    Takes pre-computed descriptor tensors as input and returns scalar
    energy predictions. Supports all kernels from ``mfml_qc.krr``:
    gaussian, matern, laplacian, wasserstein.

    Parameters
    ----------
    X_train_np : np.ndarray, shape (N, D)
        Training descriptors.
    alphas_np : np.ndarray, shape (N,)
        Solved KRR coefficients.
    kernel_type : str
        Kernel type. One of 'gaussian', 'matern', 'laplacian', 'wasserstein'.
    sigma : float
        Kernel width parameter.
    nu : float
        Matern smoothness parameter (0.5, 1.5, or 2.5).
    p : float
        Wasserstein power parameter.
    q : float
        Wasserstein outer exponent.
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
        self.register_buffer("X_train", torch.tensor(X_train_np, dtype=torch.float64))
        self.register_buffer("alphas", torch.tensor(alphas_np, dtype=torch.float64))
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.nu = nu
        self.p = p
        self.q = q

    @classmethod
    def from_krr_model(cls, krr_model):
        """
        Construct from a trained ``mfml_qc.krr.KRR`` instance.

        Parameters
        ----------
        krr_model : mfml_qc.krr.KRR
            A trained KRR model.
        """
        return cls(
            X_train_np=krr_model.X_train,
            alphas_np=krr_model.alphas,
            kernel_type=krr_model.kernel_type,
            sigma=krr_model.sigma,
            nu=getattr(krr_model, "nu", 1.5),
            p=getattr(krr_model, "p", 1.0),
            q=getattr(krr_model, "q", 1.0),
        )

    def _compute_kernel(self, X_test):
        """Compute asymmetric kernel between X_test (1, D) and X_train (N, D)."""
        if self.kernel_type == "gaussian":
            gamma = 1.0 / (2.0 * self.sigma**2)
            diffs = X_test.unsqueeze(1) - self.X_train.unsqueeze(0)
            sq_dists = torch.sum(diffs**2, dim=-1)
            return torch.exp(-gamma * sq_dists)

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
                return torch.exp(-scaled) * (1.0 + scaled + (scaled**2) / 3.0)
            else:
                raise ValueError(f"Only nu=0.5, 1.5, 2.5 supported, got {self.nu}")

        elif self.kernel_type == "laplacian":
            diffs = X_test.unsqueeze(1) - self.X_train.unsqueeze(0)
            dists = torch.sum(torch.abs(diffs), dim=-1)
            return torch.exp(-dists / self.sigma)

        elif self.kernel_type == "wasserstein":
            X_test_sorted = torch.sort(X_test, dim=-1).values
            X_train_sorted = torch.sort(self.X_train, dim=-1).values
            diffs = X_test_sorted.unsqueeze(1) - X_train_sorted.unsqueeze(0)
            n_feat = X_test.shape[-1]
            w_p = (torch.sum(torch.abs(diffs) ** self.p, dim=-1) / n_feat) ** (
                1.0 / self.p
            )
            dists = w_p**self.q
            return torch.exp(-dists / self.sigma)

        else:
            raise ValueError(f"Unsupported kernel type: '{self.kernel_type}'")

    def forward(self, X_test):
        """
        Predict scalar energy from descriptor tensor.

        Parameters
        ----------
        X_test : torch.Tensor, shape (1, D)
            Descriptor for a single geometry.

        Returns
        -------
        torch.Tensor, scalar
            Predicted energy.
        """
        k_test = self._compute_kernel(X_test)
        return torch.matmul(k_test, self.alphas).squeeze()


class DifferentiableMFML(torch.nn.Module):
    """
    Differentiable MFML evaluator wrapping 2N-1 ``DifferentiableKRR`` sub-models.

    Supports default SGCT and sklearn-based o-MFML optimizers.

    Parameters
    ----------
    sub_models : list of DifferentiableKRR
        The 2N-1 trained sub-model evaluators.
    nfids : int
        Number of fidelity levels.
    optimiser : str
        Combination strategy: 'default' (SGCT) or 'learned' (sklearn o-MFML).
    lcc_optimizer : object, optional
        Fitted sklearn estimator for o-MFML combination (OLS, LRR, LASSO, MLPR).
    """

    def __init__(self, sub_models, nfids, optimiser="default", lcc_optimizer=None):
        super().__init__()
        self.sub_models = torch.nn.ModuleList(sub_models)
        self.nfids = nfids
        self.optimiser = optimiser
        self.lcc_optimizer = lcc_optimizer

    @classmethod
    def from_mfml_model(cls, mfml_model):
        """
        Construct from a trained ``mfml_qc.mfml.ModelMFML`` instance.

        Parameters
        ----------
        mfml_model : mfml_qc.mfml.ModelMFML
            A trained MFML model.
        """
        nfids = mfml_model.indexes.shape[0]
        sub_models = [DifferentiableKRR.from_krr_model(m) for m in mfml_model.models]

        optimiser = "default"
        lcc_optimizer = None
        if mfml_model.LCCoptimizer is not None:
            lcc_optimizer = mfml_model.LCCoptimizer
            optimiser = "learned"

        return cls(
            sub_models=sub_models,
            nfids=nfids,
            optimiser=optimiser,
            lcc_optimizer=lcc_optimizer,
        )

    def forward(self, X_test):
        """
        Predict scalar energy from descriptor tensor.

        Parameters
        ----------
        X_test : torch.Tensor, shape (1, D)

        Returns
        -------
        torch.Tensor, scalar
        """
        sub_preds = torch.stack([m(X_test) for m in self.sub_models])

        if self.optimiser == "default":
            return torch.sum(sub_preds[: self.nfids]) - torch.sum(
                sub_preds[self.nfids :]
            )

        elif self.optimiser == "learned":
            preds_np = sub_preds.detach().numpy().reshape(1, -1)
            combo = self.lcc_optimizer.predict(preds_np)
            return torch.tensor(combo, dtype=torch.float64).squeeze()

        else:
            raise ValueError(f"Unknown optimiser '{self.optimiser}'")


def predict_energies_and_forces(
    evaluator, test_coords_np, charges_np, descriptor_fn=None
):
    """
    Predict energies and forces via autodifferentiation.

    For each test geometry, computes the descriptor, predicts the energy,
    and obtains forces as ``F = -∇E`` via backpropagation through the
    descriptor function and kernel evaluation.

    Parameters
    ----------
    evaluator : DifferentiableKRR or DifferentiableMFML
        A trained differentiable evaluator.
    test_coords_np : np.ndarray, shape (N_mol, N_atoms, 3)
        Test geometries.
    charges_np : np.ndarray, shape (N_atoms,)
        Nuclear charges.
    descriptor_fn : callable, optional
        Maps ``(coords_tensor, charges_tensor) -> (n_features,)`` tensor.
        Defaults to :func:`generate_coulomb_matrix`.

    Returns
    -------
    energies : np.ndarray, shape (N_mol,)
        Predicted energies.
    forces : np.ndarray, shape (N_mol, N_atoms, 3)
        Predicted forces (negative energy gradient).
    """
    if descriptor_fn is None:
        descriptor_fn = generate_coulomb_matrix

    charges = torch.tensor(charges_np, dtype=torch.float64)

    predicted_energies = []
    predicted_forces = []

    for i in range(test_coords_np.shape[0]):
        single_coords = torch.tensor(
            test_coords_np[i], dtype=torch.float64, requires_grad=True
        )

        X_test = descriptor_fn(single_coords, charges).unsqueeze(0)
        energy = evaluator(X_test)
        energy.backward()

        predicted_energies.append(energy.detach().numpy())
        predicted_forces.append(-single_coords.grad.detach().numpy())

    return np.array(predicted_energies), np.array(predicted_forces)
