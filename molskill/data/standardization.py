from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles

from molskill.data.descriptors import get_rdkit2D_desc
from molskill.helpers.logging import get_logger
from molskill.paths import DATA_PATH

LOGGER = get_logger(__name__)

CHEMBL_PATH = Path(DATA_PATH) / "compounds_al.csv"
MOMENT_PATH = Path(DATA_PATH) / "chembl_population_mean_std.csv"


def get_population_moments(
    moment_path: Path = MOMENT_PATH, desc_list: Optional[List[str]] = None
) -> Dict:
    """returns dict of population mean and std for given desc_list

    Args:
        moment_path (Path): path to saved population moments
        desc_list (Optional, List[str]): List of descriptor names

    returns:
        Dict[str, np.ndarray]: population {mean: np.ndarray, std: np.ndarray}
    """

    if not moment_path.is_file():
        moments, full_desc_nms = calculate_rdkit2d_desc_moments(
            pd.read_csv(CHEMBL_PATH)["smiles"]
        )
    else:
        population_df = pd.read_csv(moment_path, index_col="descriptor")
        full_desc_nms = population_df.index.tolist()
        moments = population_df.to_dict(orient="list")
        moments = {k: np.array(v, dtype=np.float32) for k, v in moments.items()}

    if desc_list is None:
        return moments
    else:
        if len(set(desc_list) - set(full_desc_nms)):
            LOGGER.warning(
                f"{set(desc_list) - set(full_desc_nms)} descriptors are not calculatable"
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
