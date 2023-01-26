import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles

from molskill.data.descriptors import get_rdkit2D_desc
from molskill.helpers.download import download
from molskill.helpers.logging import get_logger
from molskill.paths import DATA_PATH, DEFAULT_MOMENTS_REMOTE

LOGGER = get_logger(__name__)

CHEMBL_PATH = os.path.join(DATA_PATH, "compounds_al_tautomer.csv")
ASSET_PATH = os.path.join(DATA_PATH, "assets")
MOMENT_PATH = os.path.join(ASSET_PATH, "chembl_population_mean_std.csv")


def get_population_moments(
    moment_path: Union[str, os.PathLike] = MOMENT_PATH,
    desc_list: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Returns dict of population mean and std for given `desc_list`

    Args:
        moment_path (Union[str, os.PathLike]): Path to saved population moments .csv
        desc_list (Optional, List[str]): List of descriptor names to standardize

    returns:
        Dict[str, np.ndarray]: population {mean: np.ndarray, std: np.ndarray}
    """

    if not os.path.exists(moment_path):
        os.makedirs(ASSET_PATH, exist_ok=True)
        LOGGER.info("Standardization moments not found. Downloading from remote...")
        download(DEFAULT_MOMENTS_REMOTE, MOMENT_PATH)

    population_df = pd.read_csv(moment_path, index_col="descriptor")
    full_desc_nms = population_df.index.tolist()
    moments = population_df.to_dict(orient="list")
    moments = {k: np.array(v, dtype=np.float32) for k, v in moments.items()}

    if desc_list is None:
        return moments
    else:
        if len(set(desc_list) - set(full_desc_nms)):
            LOGGER.warning(
                f"{set(desc_list) - set(full_desc_nms)} descriptors cannot be computed"
            )

        desc_idx = [
            ii for ii, desc_nm in enumerate(full_desc_nms) if desc_nm in desc_list
        ]
        return {k: v[desc_idx] for k, v in moments.items()}


def calculate_rdkit2d_desc_moments(
    molrpr: List[str], read_f: Callable = MolFromSmiles
) -> Tuple[Dict["str", np.ndarray], List[str]]:
    """Calculate and returns dict of population meand and std of filtered ChEMBL dataset's rdkit 2D descriptors

    Args:
        molrpr (List[str]): List of molecular representations, e.g., smiles
        read_f (Callable): mol callable function, Defaults to MolFromSmiles

    Returns:
        Dict: keys = ["mean", "std"], calculated population mean and std for full descriptors
        List[str]: list of descriptor names
    """
    moments: Dict[str, np.ndarray] = dict()
    descriptors, desc_nms = get_rdkit2D_desc(molrpr, read_f)
    moments["mean"] = descriptors.mean(axis=0)
    moments["std"] = descriptors.std(axis=0)

    return moments, desc_nms
