import numpy as np
import pytest

from molskill.data.featurizers import get_featurizer
from molskill.models.ranknet import LitRankNet
from molskill.scorer import MolSkillScorer


@pytest.mark.parametrize("mc_dropout_samples", [1, 25])
def test_scorer(
    smiles_list, dropout_p, mc_dropout_samples, featurizer_name, num_workers
):
    """Tests whether the `MolSkillScorer` class returns objects of the required
    shape and type depending on whether uncertainty estimates are required
    or not.
    """
    featurizer = get_featurizer(featurizer_name)

    litranknet = LitRankNet(
        dropout_p=dropout_p,
        input_size=featurizer.dim(),
    )
    scorer = MolSkillScorer(
        model=litranknet,
        featurizer=featurizer,
        num_workers=num_workers,
        mc_dropout_samples=mc_dropout_samples,
    )
    scores_out = scorer.score(molrpr=smiles_list)
    if mc_dropout_samples > 1:
        assert len(scores_out) == 2
        assert len(scores_out[0]) == len(scores_out[1]) == len(smiles_list)
        assert isinstance(scores_out[0], np.ndarray)
        assert isinstance(scores_out[1], np.ndarray)
    else:
        assert isinstance(scores_out, np.ndarray)
        assert scores_out.ndim == 1
        assert len(scores_out) == len(smiles_list)
