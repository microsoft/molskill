import multiprocessing
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from molskill.data.featurizers import get_featurizer
from molskill.models.ranknet import LitRankNet

ASSETS_PATH = Path(__file__).parent / "assets"


@pytest.fixture
def featurizer_name() -> str:
    return "morgan_count_rdkit_2d_norm"


@pytest.fixture
def litranknet_w_uncert(dropout_p, featurizer_name):
    featurizer = get_featurizer(featurizer_name)
    return LitRankNet(
        dropout_p=dropout_p, mc_dropout_samples=25, input_size=featurizer.dim()
    )


@pytest.fixture
def dropout_p() -> float:
    return 0.2


@pytest.fixture
def test_seed() -> int:
    return 1000


@pytest.fixture
def n_epochs() -> int:
    return 3


@pytest.fixture
def random_state(test_seed) -> np.random.RandomState:
    return np.random.RandomState(test_seed)


@pytest.fixture
def num_workers() -> int:
    return multiprocessing.cpu_count() // 2


@pytest.fixture
def smiles_list() -> List[str]:
    return pd.read_csv(ASSETS_PATH / "smiles_list.csv")["smiles"].tolist()


@pytest.fixture
def duplicated_smiles_list(clean_smiles_list) -> List[str]:
    return clean_smiles_list + clean_smiles_list[:20]


@pytest.fixture
def pair_training_data() -> Tuple[List[Tuple[str, str]], List[float]]:
    df_pairs = pd.read_csv(ASSETS_PATH / "pair_list.csv")
    target = df_pairs["label"].tolist()
    molrpr = df_pairs[["smiles_i", "smiles_j"]].to_records(index=False).tolist()
    return molrpr, target



