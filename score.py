import argparse
import os

import pandas as pd

from molskill.helpers.cleaners import ensure_readability_and_remove
from molskill.paths import MODEL_PATH
from molskill.scorer import MolSkillScorer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="Scoring module for MolSkill.",
        add_help=True,
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=os.path.join(MODEL_PATH, "default", "checkpoints", "last.ckpt"),
        required=False,
        help="Path to model checkpoint file",
    )
    parser.add_argument(
        "--compound_csv",
        type=str,
        required=True,
        help="Path to .csv file with compounds to be scored",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="smiles",
        help="Column name containing SMILES strings",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save .csv file",
    )

    args = parser.parse_args()

    cpd_df = pd.read_csv(args.compound_csv)
    molrpr = cpd_df[args.smiles_col].tolist()

    molrpr = ensure_readability_and_remove(molrpr)

    scorer = MolSkillScorer(model_ckpt=args.model_ckpt)
    scores = scorer.score(molrpr=molrpr)

    score_df = pd.DataFrame({"smiles": molrpr, "score": scores})
    score_df.to_csv(args.output_csv, index=False)
