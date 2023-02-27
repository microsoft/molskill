import abc
import multiprocessing
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import rdkit
import torch
from rdkit.Chem import MolFromSmiles
from torch.utils.data import DataLoader, Dataset

from molskill.data.featurizers import Featurizer, get_featurizer
from molskill.helpers.logging import get_logger

LOGGER = get_logger(__name__)


def get_dataloader(
    molrpr: Union[List[str], List[Tuple[str, str]]],
    target: Optional[Union[List, np.ndarray]] = None,
    batch_size: int = 32,
    shuffle: bool = False,
    featurizer: Optional[Featurizer] = None,
    num_workers: Optional[int] = None,
    read_f: Callable = MolFromSmiles,
) -> DataLoader:
    """Generic factory function that returns a Pytorch `DataLoader`
    for MolSkill.

    Args:
        molrpr (Union[List[str], List[Tuple[str, str]]]): List (or list of pairs) of molecular representations.
        target (Union[List, np.ndarray]: n_pairs): Target values, default to None
        batch_size (int, default: 32): batch size for the Dataloader
        shuffle (bool, Optional): whether or not shuffling the batch at every epoch. Default to False
        featurizer (FingerprintFeaturizer, Optional): Default to MorganFingerprint
        num_workers (int, Optional): Number of processes to use during dataloading. Default is half of the
                                     available cores.
        read_f (Callable, optional): rdkit function to read molecules
    """
    if isinstance(molrpr[0], (list, tuple)):
        data = PairDataset(
            molrpr=molrpr,
            target=target,
            featurizer=featurizer,
            read_f=read_f,
        )
    elif isinstance(molrpr[0], str):
        data = SingleDataset(
            molrpr=molrpr,
            target=target,
            featurizer=featurizer,
            read_f=read_f,
        )
    else:
        raise ValueError(
            "Could not recognize `molrpr` data format. Please check function signature"
        )

    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2
    return DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


class BaseDataset(Dataset, abc.ABC):
    def __init__(
        self,
        molrpr: List,
        target: Optional[Union[List, np.ndarray]] = None,
        read_f: Callable = MolFromSmiles,
        featurizer: Optional[Featurizer] = None,
    ) -> None:
        """Base dataset class for MolSkill

        Args:
            molrpr (List): A list of molecular representations (e.g. SMILES) or a list of
                           tuples (length 2) of molecular representations.
            target (Optional[Union[List, np.ndarray]], optional): A list of target values for
                           each molecule present in `molrpr`. Defaults to None.
            read_f (Callable, optional): Function to use to read items in `molrpr`. Defaults
                           to MolFromSmiles.
            featurizer (Optional[Featurizer], optional): Featurizer to use. Defaults to None.
        """
        self.molrpr = molrpr
        self.target = target
        self.read_f = read_f
        super().__init__()

        if featurizer is None:
            featurizer = get_featurizer("morgan_count_rdkit_2d")

        self.featurizer = featurizer

    def __getitem__(self, index: int):
        raise NotImplementedError()

    def __len__(self):
        return len(self.molrpr)

    def get_desc(self, mol: rdkit.Chem.rdchem.Mol):
        raise NotImplementedError()


class PairDataset(BaseDataset):
    def __init__(
        self,
        molrpr: List,
        target: Optional[Union[List, np.ndarray]] = None,
        read_f: Callable = MolFromSmiles,
        featurizer: Optional[Featurizer] = None,
    ) -> None:
        """
        Same as `BaseDataset` but assuming that that `molrpr` is going to contain
        a list of pairs of molecular representations.
        """
        super().__init__(
            molrpr=molrpr,
            target=target,
            read_f=read_f,
            featurizer=featurizer,
        )

    def __getitem__(
        self, index: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        molrpr_index = self.molrpr[index]
        mol_i, mol_j = self.read_f(molrpr_index[0]), self.read_f(molrpr_index[1])
        desc_i, desc_j = self.get_desc(mol_i), self.get_desc(mol_j)
        if self.target is not None:
            target = torch.FloatTensor([self.target[index]])
            return (desc_i, desc_j), target
        else:
            return (desc_i, desc_j)

    def get_desc(self, mol: rdkit.Chem.rdchem.Mol):
        return torch.from_numpy(self.featurizer.get_feat(mol))


class SingleDataset(PairDataset):
    def __init__(
        self,
        molrpr: List,
        target: Optional[Union[List, np.ndarray]] = None,
        read_f: Callable = MolFromSmiles,
        featurizer: Optional[Featurizer] = None,
    ) -> None:
        """
        Same as `BaseDataset` but assuming `molrpr` is going
        to contain a list of molecular representations
        """
        super().__init__(
            molrpr=molrpr,
            target=target,
            read_f=read_f,
            featurizer=featurizer,
        )

    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        molrpr_index = self.molrpr[index]
        mol = self.read_f(molrpr_index)
        desc = self.get_desc(mol)
        if self.target is not None:
            target = torch.FloatTensor([self.target[index]])
            return desc, target
        else:
            return desc
