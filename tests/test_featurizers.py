from typing import List

import numpy as np
import pytest
from rdkit.Chem import MolFromSmiles

from molskill.data.featurizers import (
    AVAILABLE_FEATURIZERS,
    Featurizer,
    FingerprintFeaturizer,
    MultiFeaturizer,
    get_featurizer,
)


@pytest.fixture
def nbits() -> int:
    return 2048


@pytest.mark.parametrize("featurizer_name", AVAILABLE_FEATURIZERS.keys())
def test_individual_featurizer(
    smiles_list: List[str], featurizer_name: str, nbits: int
):
    featurizer = get_featurizer(featurizer_name=featurizer_name)
    feat = featurizer.get_feat(MolFromSmiles(smiles_list[0]))
    assert feat.ndim == 1
    assert len(feat) == featurizer.dim()

    if isinstance(featurizer, FingerprintFeaturizer):
        if not featurizer.count:
            assert np.max(feat) == 1


def test_multi_featurizer(smiles_list: List[str], nbits: int):
    n_feat = 0
    featurizers: List[Featurizer] = []
    for featurizer_name in AVAILABLE_FEATURIZERS.keys():
        featurizer = get_featurizer(featurizer_name)
        n_feat += featurizer.dim()
        featurizers.append(featurizer)

    multifeaturizer = MultiFeaturizer(featurizers)
    feats = multifeaturizer.get_feat(MolFromSmiles(smiles_list[0]))
    assert feats.ndim == 1
    assert len(feats) == n_feat


@pytest.mark.parametrize(
    "featurizer_name",
    [
        name
        for name in AVAILABLE_FEATURIZERS.keys()
        if ("norm" in name) and ("count" not in name)
    ],
)
def test_standardization_featurizer(smiles_list: List[str], featurizer_name: str):
    featurizer = get_featurizer(featurizer_name=featurizer_name)
    feat = featurizer.get_feat(MolFromSmiles(smiles_list[0]))
    feats = np.array([featurizer.get_feat(MolFromSmiles(smi)) for smi in smiles_list])
    assert feat.ndim == 1
    assert len(feat) == featurizer.dim()
    assert np.allclose(feats.mean(axis=0), np.zeros(feats.shape[1]), atol=1)
