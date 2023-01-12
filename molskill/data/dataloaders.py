import abc
import multiprocessing
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import rdkit
import torch
from rdkit.Chem import MolFromSmiles
from torch.utils.data import DataLoader, Dataset

from molskill.helpers.cleaners import ensure_readability
from molskill.data.featurizers import Featurizer, get_featurizer


def get_dataloader(
    molrpr: Union[List[str], List[Tuple[str, str]]],
    target: Optional[Union[List, np.ndarray]] = None,
    batch_size: int = 32,
    shuffle: bool = False,
    featurizer: Optional[Featurizer] = None,
    num_workers: Optional[int] = None,
    read_f: Callable = MolFromSmiles,
) -> DataLoader:
    """Returns PyG DataLoader

    Args:
        molrpr (Union[List[str], List[Tuple[str, str]]]): List (or list of pairs) of molecular representations.
        target (Union[List, np.ndarray]: n_pairs): Target values, default to None
        batch_size (int, default: 32): batch size for the Dataloader
        shuffle (bool, Optional): whether or not shuffling the batch at every epoch. Default to False
        featurizer (FingerprintFeaturizer, Optional): Default to MorganFingerprint
        read_f (Callable, optional): rdkit function to read molecules
    """
    if isinstance(molrpr[0], (list, tuple)):
        data = PairData(
            molrpr=molrpr, target=target, featurizer=featurizer, read_f=read_f
        )
    elif isinstance(molrpr[0], str):
        data = SingleData(
            molrpr=molrpr, target=target, featurizer=featurizer, read_f=read_f
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


class DataHloop(Dataset, abc.ABC):
    def __init__(
        self,
        molrpr: List,
        target: Optional[Union[List, np.ndarray]] = None,
        read_f: Callable = MolFromSmiles,
        check_readable: bool = True,
        featurizer: Optional[Featurizer] = None,
    ) -> None:
        self.molrpr = molrpr
        self.target = target
        self.read_f = read_f
        super().__init__()

        self.is_pair_data = True if isinstance(self.molrpr[0], (tuple, list)) else False
        if check_readable:
            self.sanity_read()
        self.featurizer = featurizer
        if featurizer is None:
            self.featurizer = get_featurizer("morgan_count_rdkit_2d")

    def sanity_read(self) -> None:
        if self.is_pair_data:
            valid_idx_i = ensure_readability(
                [rpr[0] for rpr in self.molrpr], read_f=self.read_f
            )
            valid_idx_j = ensure_readability(
                [rpr[1] for rpr in self.molrpr], read_f=self.read_f
            )
            valid_idx_i, valid_idx_j = set(valid_idx_i), set(valid_idx_j)
            valid_idx = valid_idx_i.intersection(valid_idx_j)
        else:
            valid_idx = set(
                ensure_readability([rpr for rpr in self.molrpr], read_f=self.read_f)
            )

        self.molrpr = [rpr for idx, rpr in enumerate(self.molrpr) if idx in valid_idx]

        if self.target is not None:
            self.target = [t for idx, t in enumerate(self.target) if idx in valid_idx]

    def __getitem__(self, index: int):
        raise NotImplementedError()

    def __len__(self):
        return len(self.molrpr)

    def get_desc(self, mol: rdkit.Chem.rdchem.Mol):
        raise NotImplementedError()


class PairData(DataHloop):
    def __init__(
        self,
        molrpr: List,
        target: Optional[Union[List, np.ndarray]] = None,
        read_f: Callable = MolFromSmiles,
        featurizer: Optional[Featurizer] = None,
        check_readable: bool = True,
    ) -> None:
        super().__init__(
            molrpr=molrpr,
            target=target,
            read_f=read_f,
            check_readable=check_readable,
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
        fp_i, fp_j = self.get_desc(mol_i), self.get_desc(mol_j)
        if self.target is not None:
            target = torch.FloatTensor([self.target[index]])
            return (fp_i, fp_j), target
        else:
            return (fp_i, fp_j)

    def get_desc(self, mol: rdkit.Chem.rdchem.Mol):
        return torch.from_numpy(self.featurizer.get_feat(mol))


class SingleData(PairData):
    def __init__(
        self,
        molrpr: List,
        target: Optional[Union[List, np.ndarray]] = None,
        read_f: Callable = MolFromSmiles,
        featurizer: Optional[Featurizer] = None,
        check_readable=True,
    ) -> None:
        super().__init__(
            molrpr=molrpr,
            target=target,
            read_f=read_f,
            check_readable=check_readable,
            featurizer=featurizer,
        )

    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        molrpr_index = self.molrpr[index]
        mol = self.read_f(molrpr_index)
        fp = self.get_desc(mol)
        if self.target is not None:
            target = torch.FloatTensor([self.target[index]])
            return fp, target
        else:
            return fp
