import multiprocessing
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from rdkit.Chem import MolFromSmiles

from molskill.helpers.logging import get_logger
from molskill.data.dataloaders import get_dataloader
from molskill.data.featurizers import Featurizer, get_featurizer
from molskill.models.ranknet import LitRankNet, RankNet

LOGGER = get_logger(__name__)


class MolSkillScorer:
    def __init__(
        self,
        model_ckpt: Optional[Union[os.PathLike, str]] = None,
        model: Optional[LitRankNet] = None,
        featurizer: Optional[Featurizer] = None,
        num_workers: Optional[int] = None,
        verbose: bool = True,
    ):
        """Base hloop scorer class

        Args:
            model_ckpt (Optional[Union[os.PathLike, str]], optional): Path to trained RankNet
                        model checkpoint. Defaults to None.
            model (Optional[LitRankNet]): Instead of supplying the checkpoint, pass\
                   a `LitRankNet` instance. Defaults to None.
            featurizer (Optional[Featurizer]): featurizer used to train either `model_ckpt` or
                       `model`.
            num_workers (Optional[int]): Number of workers to use in the dataloader. Default is\
                        half the available threads.
            verbose (bool, optional): Controls verbosity of the lightning trainer class.\
                    Defaults to True.
        """
        assert (model is None) != (
            model_ckpt is None
        ), "either model or model_ckpt has to be passed"

        if featurizer is None:
            featurizer = get_featurizer("morgan_count_rdkit_2d")

        self.featurizer = featurizer

        if model_ckpt is not None:
            LOGGER.info(f"Attempting to load model from checkpoint {model_ckpt}")
            self.net = RankNet(input_size=self.featurizer.dim())
            self.model = LitRankNet.load_from_checkpoint(checkpoint_path=model_ckpt, input_size=featurizer.dim())  # type: ignore
        else:
            self.model = model

        if num_workers is None:
            num_workers = multiprocessing.cpu_count() // 2
        self.num_workers = num_workers
        self.trainer = pl.Trainer(
            accelerator="auto", devices=1, max_epochs=-1, logger=verbose
        )

    def score(
        self,
        molrpr: Union[List[str], List[Tuple[str, str]]],
        batch_size: int = 32,
        read_f: Callable = MolFromSmiles,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Scores a list of compounds

        Args:
            cpds_strings (Union[List[str], List[Tuple[str, str]]]): A list of molecular strings, or a list of tuples of molecular strings if using
                                                                    paired data.
            batch_size (int, optional): Batch size for inference. Defaults to 32.
            read_f (Callable, optional): rdkit function to read the molecular strings\
                   in `cpds_strings`. Defaults to MolFromSmiles.

        Returns:
            np.ndarray: Scores for the compounds provided
        """
        loader = get_dataloader(
            molrpr=molrpr,
            batch_size=batch_size,
            read_f=read_f,
            shuffle=False,
            num_workers=self.num_workers,
            featurizer=self.featurizer,
        )
        model_out = self.trainer.predict(self.model, dataloaders=loader)
        if self.model.mc_dropout_samples > 1:
            scores_mean, scores_var = [out[0] for out in model_out], [
                out[1] for out in model_out
            ]
            return torch.cat(scores_mean).numpy(), torch.cat(scores_var).numpy()
        else:
            return torch.cat(model_out).numpy()
