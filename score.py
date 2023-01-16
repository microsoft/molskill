import argparse

import pandas as pd

from molskill.data.featurizers import AVAILABLE_FEATURIZERS, get_featurizer
from molskill.helpers.cleaners import ensure_readability_and_remove
from molskill.helpers.logging import get_logger
from molskill.models.ranknet import LitRankNet
from molskill.scorer import MolSkillScorer

LOGGER = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="Scoring module for MolSkill.",
        add_help=True,
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        required=False,
        help="Path to model checkpoint (`.ckpt`) file.",
    )
    parser.add_argument(
        "--featurizer_name",
        choices=list(AVAILABLE_FEATURIZERS.keys()),
        default="morgan_count_rdkit_2d",
        help="Molecular representation to use.",
    )
    parser.add_argument(
        "--compound_csv",
        type=str,
        required=True,
        help="Path to a `.csv` file separated by commas with compounds to be scored.",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="smiles",
        help="Column name containing SMILES strings.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output `.csv` file, which will contain a column of SMILES and another of corresponding scores.",
    )

    args = parser.parse_args()

    cpd_df = pd.read_csv(args.compound_csv)
    molrpr = cpd_df[args.smiles_col].tolist()

    molrpr = ensure_readability_and_remove(molrpr)
    featurizer = get_featurizer(args.featurizer_name)

    model = (
        LitRankNet.load_from_checkpoint(args.model_ckpt, input_size=featurizer.dim())
        if args.model_ckpt is not None
        else None
    )
    scorer = MolSkillScorer(model=model, featurizer=featurizer)
    LOGGER.info("Now predicting...")
    scores = scorer.score(molrpr=molrpr)

    score_df = pd.DataFrame({"smiles": molrpr, "score": scores})
    score_df.to_csv(args.output_csv, index=False)
