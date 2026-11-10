import os
import numpy as np
from typing import Dict, Any
from .representations import generate_coulomb_matrices, compute_flat_coulomb


def load_benzene_data(data_dir: str = None) -> Dict[str, Any]:
    """
    Helper utility to load and parse the built-in Benzene trajectory dataset.

    This function reads the pre-computed energy and time cost CSV files,
    and either loads the cached Coulomb matrices or generates them
    dynamically from the provided XYZ trajectory file.

    Parameters
    ----------
    data_dir : str, optional
        Path to the benzene data directory. If None, it dynamically
        resolves the path relative to the installed package's directory
        (assuming the standard 'data/benzene' repository layout).
        Default is None.

    Returns
    -------
    dict
        A dictionary containing the parsed dataset components:

        * ``'X_CM'`` (np.ndarray): Flattened Coulomb matrices (shape: 15000, 36).
        * ``'energies'`` (np.ndarray): Raw energies extracted from the CSV data.
        * ``'timecosts'`` (np.ndarray): Time costs extracted from the CSV data.
        * ``'columns'`` (list of str): List of string names corresponding to each column.

    Raises
    ------
    FileNotFoundError
        If the required 'energies.csv' file is not found at the resolved path,
        indicating the data has not been downloaded or placed correctly.
    """
    if data_dir is None:
        # Resolve path relative to the installed package directory
        # package lives in src/mfml_qc/ so we go two levels up to find the root 'data/benzene'
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "benzene")
        )

    xyz_path = os.path.join(data_dir, "traj.xyz")
    csv_path = os.path.join(data_dir, "energies.csv")
    time_path = os.path.join(data_dir, "timecosts.csv")
    cm_cache = os.path.join(data_dir, "X_CM.npy")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Benzene dataset not found at {data_dir}. "
            "Ensure you have placed the files in your project's 'data/benzene' directory.\nIf you have not installed the dataset in the pip installation, please do so."
        )

    # Generate CM or load if already generated and saved
    if os.path.exists(cm_cache):
        X_CM = np.load(cm_cache)
    else:
        X_CM = generate_coulomb_matrices(xyz_path, save_path=cm_cache)

    # Energies and Time costs
    energies = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    timecosts = np.genfromtxt(time_path, delimiter=",", skip_header=1)

    # Define exact columns representing the CSV structure
    columns = [
        "Time",
        "ZINDO",
        "LC-DFTB",
        "STO-3G",
        "3-21G",
        "6-31G",
        "def2-SVP",
        "def2-TZVP",
        "def2-QZVP",
    ]

    return {
        "X_CM": X_CM,
        "energies": energies,
        "timecosts": timecosts,
        "columns": columns,
    }


def load_nitrophenol_data(data_dir: str = None) -> Dict[str, Any]:
    """
    Load and parse the built-in QeMFi nitrophenol dataset.

    The QeMFi dataset contains multifidelity electronic structure properties
    for 15,000 MD geometries of nitrophenol at 5 SCF fidelity levels
    (STO-3G, 3-21G, 6-31G, def2-SVP, def2-TZVP).

    Reference: Vinod et al., J. Chem. Phys. (2024).

    Parameters
    ----------
    data_dir : str, optional
        Path to the nitrophenol data directory. If None, resolves the path
        relative to the installed package's directory (assuming the standard
        'data/nitrophenol' repository layout). Default is None.

    Returns
    -------
    dict
        A dictionary containing the parsed dataset components:

        * ``'R'`` (np.ndarray): Cartesian coordinates, shape (15000, 15, 3).
        * ``'Z'`` (np.ndarray): Atomic numbers, shape (15,).
        * ``'SCF'`` (np.ndarray): SCF energies at 5 fidelities, shape (15000, 5).
        * ``'DPe'`` (np.ndarray): Electronic dipole moments, shape (15000, 5, 3).
        * ``'X_CM'`` (np.ndarray): Flattened Coulomb matrices, shape (15000, 120).
        * ``'fidelities'`` (list of str): Fidelity level names.
        * ``'ID'`` (np.ndarray, optional): WS22 geometry indices, shape (15000,).

    Raises
    ------
    FileNotFoundError
        If the required data file is not found at the resolved path.
    """
    if data_dir is None:
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "nitrophenol")
        )

    npz_path = os.path.join(data_dir, "QeMFi_nitrophenol.npz")
    cm_cache = os.path.join(data_dir, "X_CM.npy")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"QeMFi nitrophenol dataset not found at {data_dir}. "
            "Ensure you have placed QeMFi_nitrophenol.npz in your project's "
            "'data/nitrophenol' directory."
        )

    data = np.load(npz_path, allow_pickle=True)
    R = data["R"]
    Z = data["Z"]

    if os.path.exists(cm_cache):
        X_CM = np.load(cm_cache)
    else:
        n_geom = R.shape[0]
        n_feat = Z.shape[0] * (Z.shape[0] + 1) // 2
        X_CM = np.zeros((n_geom, n_feat))
        for i in range(n_geom):
            X_CM[i] = compute_flat_coulomb(Z=Z, R=R[i])
        np.save(cm_cache, X_CM)

    fidelities = ["STO-3G", "3-21G", "6-31G", "def2-SVP", "def2-TZVP"]

    result = {
        "R": R,
        "Z": Z,
        "SCF": data["SCF"],
        "DPe": data["DPe"],
        "X_CM": X_CM,
        "fidelities": fidelities,
    }

    if "ID" in data.files:
        result["ID"] = data["ID"]

    return result
