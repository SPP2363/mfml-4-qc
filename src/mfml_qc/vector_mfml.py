"""
MFML for vector-valued properties.

Provides ``VectorMFML``, a multi-fidelity wrapper that trains and predicts
vector targets (e.g., dipole moments) using the Single-Grid Combination
Technique (SGCT). Uses any duck-typed vector estimator (VectorKRR,
CoupledVectorKRR, RPRVectorKRR) as the sub-model.

Also provides ``DifferentiableVectorMFML`` (requires PyTorch) for
differentiable inference pipelines.

Note: ``ModelMFML.predict`` assumes scalar outputs, so ``VectorMFML``
implements its own SGCT prediction over d-dimensional vectors.
"""

import copy
import numpy as np
from .utils import top_down_subsetting


class VectorMFML:
    """
    Multi-Fidelity Machine Learning for vector-valued properties.

    Trains 2N-1 sub-models (N upper + N-1 lower) following the SGCT scheme
    and predicts d-dimensional vectors.

    Parameters
    ----------
    base_estimator : object
        A model instance with ``.train(X, Y)`` and ``.predict(X)`` methods.
        Will be deep-copied for each sub-model. E.g., ``VectorKRR``,
        ``CoupledVectorKRR``, or ``RPRVectorKRR``.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.models = None
        self.nfids = None
        self.sub_Y_mean = None

    def train(self, X_train_parent, y_trains, indexes, n_trains_target=None, seed=42):
        """
        Train the MFML vector model.

        Parameters
        ----------
        X_train_parent : np.ndarray, shape (N_parent, D)
            Feature matrix for the full parent (lowest fidelity) training set.
            For RPR models, these are flattened coordinates (N, N_atoms*3).
            For non-RPR models, these are pre-computed descriptors.
        y_trains : np.ndarray of object, length num_fids
            Each element is (N_i, d) array of mean-centered vector targets
            at fidelity i.
        indexes : np.ndarray of object, length num_fids
            Each element is (N_i, 2) array of [baseline_idx, level_idx] mappings.
        n_trains_target : list of int, optional
            Target number of training samples per fidelity (lowest to highest).
            If None, uses all available samples.
        seed : int
            Random seed for subsetting.
        """
        self.nfids = len(y_trains)
        n_sub = 2 * self.nfids - 1

        if n_trains_target is not None:
            subset_y, subset_idx = top_down_subsetting(
                y_trains, indexes, n_trains_target, seed=seed
            )
        else:
            subset_y, subset_idx = y_trains, indexes

        # Build sub-model training data
        sub_X = []
        sub_Y = []
        self.sub_Y_mean = []

        # Upper sub-models: fidelity i targets on fidelity i subset
        for i in range(self.nfids):
            idx_i = subset_idx[i][:, 0]
            Y_i = (
                subset_y[i][subset_idx[i][:, 1]]
                if subset_y[i].ndim > 1
                else subset_y[i]
            )
            # Re-index: subset_idx maps into the original arrays
            # For direct indexing into X_train_parent we use baseline indices
            sub_X.append(X_train_parent[idx_i])
            m_i = np.mean(Y_i, axis=0) if Y_i.ndim > 1 else np.mean(Y_i)
            sub_Y.append(Y_i - m_i)
            self.sub_Y_mean.append(m_i)

        # Lower sub-models: fidelity i targets on fidelity i+1 subset
        for i in range(self.nfids - 1):
            idx_ip1 = subset_idx[i + 1][:, 0]
            # Need fidelity i targets for the samples that are in fidelity i+1's subset
            # These samples exist in fidelity i's data; find them via baseline index
            fid_i_map = {row[0]: row[1] for row in subset_idx[i]}
            level_indices = [fid_i_map[b] for b in idx_ip1 if b in fid_i_map]
            Y_lower = subset_y[i][level_indices]
            m_i = np.mean(Y_lower, axis=0) if Y_lower.ndim > 1 else np.mean(Y_lower)
            sub_X.append(X_train_parent[idx_ip1])
            sub_Y.append(Y_lower - m_i)
            self.sub_Y_mean.append(m_i)

        # Train all sub-models
        self.models = []
        for s in range(n_sub):
            model_s = copy.deepcopy(self.base_estimator)
            model_s.train(sub_X[s], sub_Y[s])
            self.models.append(model_s)

    def predict(self, X_test):
        """
        Predict vector targets using SGCT combination.

        Parameters
        ----------
        X_test : np.ndarray, shape (M, D)
            Test feature matrix (descriptors or flattened coordinates for RPR).

        Returns
        -------
        np.ndarray, shape (M, d)
            Predicted vectors (un-centered).
        """
        if self.models is None:
            raise ValueError("Model not trained. Call .train() first.")

        # Get output dimension from first sub-model prediction
        pred_0 = self.models[0].predict(X_test)
        M = pred_0.shape[0]
        d = pred_0.shape[1] if pred_0.ndim > 1 else 1

        result = np.zeros((M, d))

        # Upper: sum predictions + means
        for i in range(self.nfids):
            pred = self.models[i].predict(X_test)
            result += pred + self.sub_Y_mean[i]

        # Lower: subtract predictions + means
        for i in range(self.nfids - 1):
            pred = self.models[self.nfids + i].predict(X_test)
            result -= pred + self.sub_Y_mean[self.nfids + i]

        return result


# ============================================================
# Differentiable evaluator (requires PyTorch)
# ============================================================

try:
    import torch
    from .vector_krr import DifferentiableVectorKRR

    class DifferentiableVectorMFML(torch.nn.Module):
        """
        Differentiable MFML evaluator for vector properties.

        Wraps 2N-1 ``DifferentiableVectorKRR`` sub-models and applies SGCT
        arithmetic on d-dimensional prediction vectors.

        Parameters
        ----------
        sub_models : list of DifferentiableVectorKRR
        nfids : int
        """

        def __init__(self, sub_models, nfids):
            super().__init__()
            self.sub_models = torch.nn.ModuleList(sub_models)
            self.nfids = nfids

        @classmethod
        def from_vector_mfml(cls, vector_mfml_model):
            """Construct from a trained ``VectorMFML`` instance."""
            nfids = vector_mfml_model.nfids
            sub_models = [
                DifferentiableVectorKRR.from_vector_krr(m)
                for m in vector_mfml_model.models
            ]
            return cls(sub_models=sub_models, nfids=nfids)

        def forward(self, X_test):
            """
            Parameters
            ----------
            X_test : torch.Tensor, shape (1, D)

            Returns
            -------
            torch.Tensor, shape (d,)
            """
            sub_preds = torch.stack([m(X_test) for m in self.sub_models])
            upper = torch.sum(sub_preds[: self.nfids], dim=0)
            lower = torch.sum(sub_preds[self.nfids :], dim=0)
            return upper - lower

except ImportError:
    pass
