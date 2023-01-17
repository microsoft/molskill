import os
import random
import tempfile
from typing import Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from molskill.helpers.logging import get_logger
from molskill.models.ranknet import LitRankNet, RankNet

LOGGER = get_logger(__name__)


def setup_seed(seed: int):
    """Set seed for all torch and cuda dependencies

    Args:
        seed (int): seed for torch models
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_new_model_and_trainer(
    save_dir: Optional[Union[os.PathLike, str]] = None,
    lr: float = 3e-4,
    n_epochs: int = 100,
    log_every: int = 20,
    regularization_factor: float = 0.0,
    dropout_p: float = 0.0,
    mc_dropout_samples: int = 1,
    sigmoid: bool = False,
    input_size: int = 2048,
) -> Tuple[LitRankNet, pl.Trainer]:
    """Initialize LitRankNet model and pl trainer.
    The trainer uses best model path to return best model score.
    When resume_from_saved == True, use the last saved ckpt to resume the training.

    Args:
        save_dir (os.PathLike | str): directory path to save model ckpt
        lr (float, optional): learning rate to initialize LitRankNet model. Defaults to 3e-4.
        n_epochs (int, optional): max_epoch to initialize pl.Trainer. Defaults to 100.
        log_every (int, optional): log_every_n_steps to initialize pl.Trainer. Defaults to 20.
        regularization_factor (float, optional): regularization factor for the (norm) of learned scores.
                                                 If > 0.0, it will encourage them to be centered around 0.
        dropout_p (float, optional): mc dropout probability to initialize LitRankNet model. Defaults to 0.2.
        mc_dropout_samples (int, optional): the number of samples for mc dropout
        sigmoid (bool, optional): whether of not to apply a sigmoid function for paired data
        input_size (int, optional): input size of given featurizer, default to 2048
    Returns:
        Tuple[LitRankNet, pl.Trainer]: Return model and trainer
    """

    # use tempdir if unavailable
    if save_dir is None:
        save_dir = tempfile.mkdtemp()
        LOGGER.info(f"Save directory not provided, using {save_dir} to save model logs")

    os.makedirs(save_dir, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=save_dir,
        name="default",
    )
    net = RankNet(input_size=input_size, dropout_p=dropout_p)
    model = LitRankNet(
        net=net,
        lr=lr,
        regularization_factor=regularization_factor,
        mc_dropout_samples=mc_dropout_samples,
        dropout_p=dropout_p,
        sigmoid=sigmoid,
    )

    ckpt_callback = ModelCheckpoint(
        monitor="train/loss",
        dirpath=os.path.join(save_dir, "checkpoints"),
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=n_epochs,
        logger=logger,
        log_every_n_steps=log_every,
        callbacks=[ckpt_callback],
    )

    return model, trainer
