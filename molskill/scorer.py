import multiprocessing
import os
import tarfile
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from rdkit.Chem import MolFromSmiles

from molskill.data.dataloaders import get_dataloader
from molskill.data.featurizers import Featurizer, get_featurizer
from molskill.helpers.download import download
from molskill.helpers.logging import get_logger
from molskill.models.ranknet import LitRankNet
from molskill.paths import DEFAULT_CHECKPOINT_PATH, DEFAULT_CHECKPOINT_REMOTE, ROOT_PATH

LOGGER = get_logger(__name__)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MolSkillScorer:
    def __init__(
        self,
        model: Optional[LitRankNet] = None,
        featurizer: Optional[Featurizer] = None,
        num_workers: Optional[int] = None,
        verbose: bool = True,
        mc_dropout_samples: int = 1,
    ):
        """Base MolSkill scorer class

        Args:
            model (Optional[LitRankNet]): Instead of supplying the checkpoint, pass
                   a `LitRankNet` instance. Defaults to None.
            featurizer (Optional[Featurizer]): featurizer used to train either `model_ckpt` or
                       `model`. Default is count Morgan fingerprints and rdkit 2d descriptors.
            num_workers (Optional[int]): Number of workers to use in the dataloader. Default is
                        half the available threads.
            verbose (bool, optional): Controls verbosity of the lightning trainer class.
                    Defaults to True.
            mc_dropout_samples (int, optional): If >1, the `score` method will return both
                                                predicted scores and uncertainty estimates
                                                using the Monte Carlo dropout method.
        """
        if featurizer is None:
            featurizer = get_featurizer("morgan_count_rdkit_2d")

        self.featurizer = featurizer

        if model is None:
            LOGGER.info(
                f"Model not specified. Using default from {DEFAULT_CHECKPOINT_PATH}."
            )
            if not os.path.exists(DEFAULT_CHECKPOINT_PATH):
                LOGGER.info("Default model not present. Downloading asset...")
                self._download_default_model()
            model = LitRankNet.load_from_checkpoint(
                checkpoint_path=DEFAULT_CHECKPOINT_PATH,
                input_size=featurizer.dim(),
                map_location=DEVICE,
            )  # type: ignore

        self.model = model
        self.model.mc_dropout_samples = mc_dropout_samples

        if num_workers is None:
            num_workers = multiprocessing.cpu_count() // 2
        self.num_workers = num_workers
        self.trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=-1,
            logger=verbose,
            enable_progress_bar=verbose,
        )

    def score(
        self,
        molrpr: Union[List[str], List[Tuple[str, str]]],
        batch_size: int = 32,
        read_f: Callable = MolFromSmiles,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Scores a list of compounds strings.

        Args:
            molrpr (Union[List[str], List[Tuple[str, str]]]): A list of molecular strings,
                                                              or a list of tuples of molecular
                                                              strings if using paired data.
            batch_size (int, optional): Batch size for inference. Defaults to 32.
            read_f (Callable, optional): rdkit function to read the molecular strings
                   in `cpds_strings`. Defaults to MolFromSmiles.

        Returns:
            np.ndarray: Scores for the compounds in `molrpr` and (optionally) uncertainty values.
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

        # shut up pyright
        assert self.model is not None
        assert model_out is not None

        if self.model.mc_dropout_samples > 1:
            scores_mean, scores_var = [out[0] for out in model_out], [
                out[1] for out in model_out
            ]
            return torch.cat(scores_mean).numpy(), torch.cat(scores_var).numpy()
        else:
            return torch.cat(model_out).numpy()

    @staticmethod
    def _download_default_model(remote_url: str = DEFAULT_CHECKPOINT_REMOTE):
        """Downloads default MolSkill model from remote. Was trained with `morgan_count_rdkit_2d`
        featurizer
        """
        targz_path = os.path.join(ROOT_PATH, "models.tar.gz")
        download(remote_url, dest=targz_path)

        with tarfile.open(targz_path) as tar_handle:
            tar_handle.extractall(ROOT_PATH)

        os.remove(targz_path)
