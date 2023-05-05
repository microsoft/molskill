import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from molskill.helpers.download import download
from molskill.helpers.logging import get_logger
from molskill.paths import DATA_PATH, DEFAULT_MOMENTS_REMOTE

LOGGER = get_logger(__name__)

CHEMBL_PATH = os.path.join(DATA_PATH, "compounds_al_tautomer.csv")
ASSET_PATH = os.path.join(DATA_PATH, "assets")
MOMENT_CSV = os.path.join(ASSET_PATH, "chembl_population_mean_std.csv")


def get_population_moments(
    desc_list: List[str],
    moment_csv: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, np.ndarray]:
    """Returns dict of population mean and std for given `desc_list`

    Args:
        desc_list (Optional, List[str]): List of descriptor names to standardize
        moment_csv (Union[str, os.PathLike]): Path to saved population moments .csv

    returns:
        Dict[str, np.ndarray]: population {mean: np.ndarray, std: np.ndarray}
    """

    if moment_csv is None:
        LOGGER.info("Standardization moments not found. Downloading from remote...")
        moment_csv = MOMENT_CSV
        os.makedirs(ASSET_PATH, exist_ok=True)
        download(DEFAULT_MOMENTS_REMOTE, moment_csv)

    population_df = pd.read_csv(moment_csv, index_col="descriptor")
    assert set(population_df.keys()) == {
        "mean",
        "std",
    }, "Not recognised keys in standardization data"
    for desc in desc_list:
        assert (
            desc in population_df.index
        ), f"{desc} moments not found in precomputed file {moment_csv}"

    moments = {k: v.loc[desc_list].values for k, v in population_df.items()}
    return moments
