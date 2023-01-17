from typing import Callable, Iterable, List, Optional, Set, Tuple, Union

from rdkit.Chem import MolFromSmiles
from tqdm import tqdm

from molskill.helpers.logging import get_logger

LOGGER = get_logger(__name__)


def ensure_readability(
    strings: Iterable[str],
    read_f: Callable = MolFromSmiles,
    cond_fs: Optional[List[Callable]] = None,
    verbose: bool = True,
) -> Set[int]:
    """Checks readability of the molecular `strings` using `read_f` and the molecule agrees condition functions.

    Args:
        strings (Iterable[str]): A list of molecular strings (e.g., SMILES)
        read_f (Callable): A function to read those strings
        cond_fs (List[Callable]): list of functions that checks conditions of molecule, defauts to None, which is lambda x: True, always returns True
        verbose (bool): Whether to print progress bar

    Returns:
        Set[int]: A list of indexes of those molecules in `strings` that
                   could be successfully read
    """
    if cond_fs is None:
        # when no specific condition was applied, always return true
        cond_fs = [lambda x: True]

    if verbose:
        strings = tqdm(strings)

    valid_idx: List[int] = []
    for idx, string in enumerate(strings):
        mol = read_f(string)
        if mol is not None:
            if all([func(mol) for func in cond_fs]):
                valid_idx.append(idx)
            else:
                LOGGER.warning(f"{string} does not agree specified condition(s)")
        else:
            LOGGER.warning(f"{string} is not readable")
    return set(valid_idx)


def ensure_pair_readability(
    pair_molrpr: List[Tuple[str, str]],
    read_f: Callable = MolFromSmiles,
    cond_fs: Optional[List[Callable]] = None,
    verbose: bool = True,
) -> Set[int]:
    """Checks readability of the pairs of molecular `strings` using `read_f`.
    Args:
        pair_molrpr (List[Tuple[str,str]]): A list of pairs of molecular strings (e.g., SMILES)
        read_f (Callable, optional): A function to read those strings. Defaults to MolFromSmiles.
        cond_fs (List[Callable]): list of functions that checks conditions of molecule, defauts to None, which is lambda x: True, always returns True
        verbose (bool): Whether to print progress bar
    Returns:
        Set[int]: A list of indexes of those pairs that both molecules could be successfully read
    """
    valid_idx: List[int] = []

    pairs_range = range(len(pair_molrpr))
    if verbose:
        pairs_range = tqdm(pairs_range)

    for ii in pairs_range:
        if (
            len(ensure_readability(pair_molrpr[ii], read_f, cond_fs, verbose=False))
            == 2
        ):
            valid_idx.append(ii)
    return set(valid_idx)


def ensure_readability_and_remove(
    molrpr: Union[List[str], List[Tuple[str, str]]],
    target: Optional[List[float]] = None,
    read_f: Callable = MolFromSmiles,
    cond_fs: Optional[List[Callable]] = None,
    verbose: bool = True,
) -> Union[List, Tuple[List, List]]:
    """Checks whether the molecules in `molrpr` are readable and
    returns a new list with those that are in the same order as the
    original list.

    Args:
        molrpr (Union[List[str], List[Tuple[str, str]]]): Either a list with molecular strings or a list with pairs
        of molecular strings
        target (Optional[List[float]], optional): An optional target variable (e.g., ratings). Defaults to None.
        read_f (Callable, optional): A function to read those strings. Defaults to MolFromSmiles.
        cond_fs (List[Callable]): list of functions that checks conditions of molecule, defauts to None, which
                                  is lambda x: True, always returns True
        verbose (bool): Whether to print progress bar

    Returns:
        molrpr (Union[List[str], List[Tuple[str, str]]]): Readable molecular string in the same format as input
        target (Optional[List[float]], optional): An optional target variable, only for readable mols. When target
                                                  is None, not returned
    """
    if isinstance(molrpr[0], str):
        read_fun = ensure_readability
    elif isinstance(molrpr[0], tuple):
        read_fun = ensure_pair_readability
    else:
        raise ValueError("Cannot recognize input type.")

    LOGGER.info("Checking SMILES validity and removing unreadable ones...")
    valid_idx = read_fun(molrpr, read_f=read_f, cond_fs=cond_fs, verbose=verbose)

    if len(valid_idx) != len(molrpr):
        LOGGER.warning(
            f"Had to remove {len(molrpr) - len(valid_idx)} molecules. Please check stdout log."
        )
    molrpr = [molrpr[idx] for idx in valid_idx]

    if target is not None:
        target = [t for idx, t in enumerate(target) if idx in valid_idx]
        return molrpr, target
    return molrpr
