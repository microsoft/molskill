import argparse
import os
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
from pytorch_lightning.utilities.seed import seed_everything
from rdkit.Chem import MolFromSmiles
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from molskill.data.dataloaders import get_dataloader
from molskill.data.featurizers import AVAILABLE_FEATURIZERS, Featurizer, get_featurizer
from molskill.helpers.cleaners import ensure_readability_and_remove
from molskill.helpers.logging import get_logger
from molskill.models.utils import get_new_model_and_trainer
from molskill.paths import MODEL_PATH

LOGGER = get_logger(__name__)


def train_ranknet(
    molrpr: List[Tuple[str, str]],
    target: List[float],
    save_dir: Optional[Union[str, os.PathLike]] = None,
    featurizer: Optional[Featurizer] = None,
    lr: float = 3e-4,
    regularization_factor: float = 1e-4,
    n_epochs: int = 100,
    log_every: int = 10,
    val_size: float = 0.0,
    seed: Optional[int] = None,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    read_f: Callable = MolFromSmiles,
) -> None:
    """Trains a RankNet model from scratch and saves results

    Args:
        molrpr (List[Tuple[str, str]]): A list of tuples containing molecular representations (e.g., SMILES)
        target (List[float]): Target values to train RankNet on ([0.0-1.0] range)
        save_dir (Optional[Union[str, os.PathLike]]): Directory to save trained model and results
        featurizer (Optional[Featurizer]): Featurizer to use when training the model. Default is count-based\
                                           Morgan Fingerprints and rdkit 2d descriptors.
        lr (float, optional): Initial learning rate. Defaults to 3e-4.
        regularization_factor (float, optional): Regularization factor for the learned scores. It is usually\
                                                 enough to set to a small value to guarantee 0-centering\
                                                 Defaults to 1e-4.
        n_epochs (int, optional): Number of maximum epochs to train. Defaults to 100.
        log_every (int, optional): Logging interval when training. Defaults to 10.
        val_size (float, optional): Random split validation set fraction. Defaults to 0.0.
        seed (Optional[int], optional): Random seed. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (Optional[int]): Number of threads to use when loading data from the dataloaders.\
                                     Defaults to half the available threads.
    """
    molrpr, target = ensure_readability_and_remove(molrpr, target=target, read_f=read_f)

    val_loaders: List[DataLoader] = []

    if val_size > 0:
        train_molrpr, val_molrpr, train_target, val_target = train_test_split(
            molrpr, target, test_size=val_size, random_state=seed
        )
        val_loaders.append(
            get_dataloader(
                val_molrpr,
                val_target,
                batch_size=batch_size,
                shuffle=False,
                featurizer=featurizer,
                num_workers=num_workers,
                read_f=read_f,
            )
        )

    else:
        LOGGER.info(
            "No validation data provided. Performing production training run on entire set."
        )
        train_molrpr, train_target = molrpr, target

    train_loader = get_dataloader(
        train_molrpr,
        train_target,
        batch_size=batch_size,
        shuffle=True,
        featurizer=featurizer,
        num_workers=num_workers,
        read_f=read_f,
    )

    model, trainer = get_new_model_and_trainer(
        save_dir=save_dir,
        lr=lr,
        regularization_factor=regularization_factor,
        n_epochs=n_epochs,
        log_every=log_every,
        input_size=train_loader.dataset.featurizer.dim(),
    )

    # use the last model to resume training if available
    assert save_dir is not None  # shut up pyright
    model_ckpt = os.path.join(save_dir, "checkpoints", "last.ckpt")
    if os.path.exists(model_ckpt):
        LOGGER.info(f"Found checkpoint in {model_ckpt}, resuming training.")
    else:
        model_ckpt = None

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loaders,
        ckpt_path=model_ckpt,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="MolSkill training module for a RankNet model on pair preference data.",
        add_help=True,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(MODEL_PATH, "default"),
        required=False,
        help="Directory path to store model checkpoints",
    )
    parser.add_argument(
        "--compound_csv",
        type=str,
        required=True,
        help="Path to compound `.csv` file with pair ratings.",
    )
    parser.add_argument(
        "--compound_cols",
        type=List[str],
        default=["smiles_i", "smiles_j"],
        help="Column names with SMILES for each comparison.",
    )
    parser.add_argument(
        "--rating_col",
        type=str,
        default="label",
        help="Column name with the target rating label",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.0,
        help="Fraction of compounds to use for validation during training [0.0-1)",
    )
    parser.add_argument(
        "--regularization_factor",
        type=float,
        default=1e-4,
        help="Regularization factor for the learned scores. \
            If > 0.0, it encourages optimization to center them the real origin.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--log_every",
        dest="log_every",
        type=int,
        default=20,
        help="Log metrics every `log_every` steps.",
    )
    parser.add_argument(
        "--n_epochs",
        dest="n_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs for training.",
    )
    parser.add_argument(
        "--featurizer_name",
        choices=list(AVAILABLE_FEATURIZERS.keys()),
        default="morgan_count_rdkit_2d",
        help="Molecular representation to use.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Workers to use (processes) during training. Default is half of the available cores.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    os.makedirs(MODEL_PATH, exist_ok=True)
    assert (
        len(args.compound_cols) == 2
    ), f"Compound columns need to be 2, {len(args.compound_cols)} passed instead"

    seed_everything(args.seed)

    ratings_df = pd.read_csv(args.compound_csv)
    molrpr: List[Tuple[str, str]] = (
        ratings_df[args.compound_cols].to_records(index=False).tolist()
    )
    target = ratings_df[args.rating_col].tolist()
    featurizer = get_featurizer(args.featurizer_name)

    train_ranknet(
        molrpr=molrpr,
        target=target,
        save_dir=args.save_dir,
        featurizer=featurizer,
        lr=args.lr,
        regularization_factor=args.regularization_factor,
        n_epochs=args.n_epochs,
        log_every=args.log_every,
        val_size=args.val_size,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
