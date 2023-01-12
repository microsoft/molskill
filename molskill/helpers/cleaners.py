from typing import Callable, Iterable, List, Optional, Set, Tuple, Union

import rdkit
from rdkit.Chem import AtomValenceException, MolFromSmarts, MolFromSmiles
from rdkit.Chem.MolStandardize.rdMolStandardize import Cleanup
from rdkit.Chem.SaltRemover import SaltRemover
from tqdm import tqdm

from molskill.helpers.logging import get_logger

LOGGER = get_logger(__name__)


def ensure_readability(
    strings: Iterable[str],
    read_f: Callable = MolFromSmiles,
    cond_fs: Optional[List[Callable]] = None,
) -> Set[int]:
    """Checks readability of the molecular `strings` using `read_f` and the molecule agrees condition functions.

    Args:
        strings (Iterable[str]): A list of molecular strings (e.g., SMILES)
        read_f (Callable): A function to read those strings
        cond_fs (List[Callable]): list of functions that checks conditions of molecule, defauts to None, which is lambda x: True, always returns True

    Returns:
        Set[int]: A list of indexes of those molecules in `strings` that
                   could be successfully read
    """
    if cond_fs is None:
        # when no specific condition was applied, always return true
        cond_fs = [lambda x: True]

    valid_idx = []
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
) -> Set[int]:
    """Checks readability of the pairs of molecular `strings` using `read_f`.

    Args:
        pair_molrpr (List[Tuple[str,str]]): A list of pairs of molecular strings (e.g., SMILES)
        read_f (Callable, optional): A function to read those strings. Defaults to MolFromSmiles.
        cond_fs (List[Callable]): list of functions that checks conditions of molecule, defauts to None, which is lambda x: True, always returns True

    Returns:
        Set[int]: A list of indexes of those pairs that both molecules could be successfully read
    """
    valid_idx = []
    for ii in range(len(pair_molrpr)):
        if len(ensure_readability(pair_molrpr[ii], read_f, cond_fs)) == 2:
            valid_idx.append(ii)
    return set(valid_idx)


def ensure_readability_and_remove(
    molrpr: Union[List[str], List[Tuple[str, str]]],
    target: Optional[List[float]] = None,
    read_f: Callable = MolFromSmiles,
    cond_fs: Optional[List[Callable]] = None,
) -> Union[List, Tuple[List, List]]:
    """Checks whether the molecules in `molrpr` are readable and
    returns a new list with those that are in the same order as the
    original list.

    Args:
        molrpr (Union[List[str], List[Tuple[str, str]]]): Either a list with molecular strings or a list with pairs
        of molecular strings
        target (Optional[List[float]], optional): An optional target variable (e.g., ratings). Defaults to None.
        read_f (Callable, optional): A function to read those strings. Defaults to MolFromSmiles.
        cond_fs (List[Callable]): list of functions that checks conditions of molecule, defauts to None, which is lambda x: True, always returns True

    Returns:
        molrpr (Union[List[str], List[Tuple[str, str]]]): Readable molecular string in the same format as input
        target (Optional[List[float]], optional): An optional target variable, only for readable mols. When target is None, not returned
    """
    if isinstance(molrpr[0], str):
        read_fun = ensure_readability
    elif isinstance(molrpr[0], tuple):
        read_fun = ensure_pair_readability
    else:
        raise ValueError("Cannot recognize input type")

    LOGGER.info("Checking SMILES validity and removing unreadable ones...")
    valid_idx = read_fun(molrpr, read_f=read_f, cond_fs=cond_fs)

    if len(valid_idx) != len(molrpr):
        LOGGER.warning(
            f"Had to remove {len(molrpr) - len(valid_idx)} molecules. Please check stdout log."
        )
    molrpr = [molrpr[idx] for idx in valid_idx]

    if target is not None:
        target = [t for idx, t in enumerate(target) if idx in valid_idx]
        return molrpr, target
    return molrpr


def clean_wrapper(
    strings: Iterable[str], read_f: Callable, write_f: Callable
) -> Tuple[List[str], List[int]]:
    """Simple wrapper that cleans a list of molecular `strings`

    Args:
        strings (Iterable[str]): A list of molecular strings (e.g., SMILES)
        read_f (Callable): A function to read those strings
        write_f (Callable): A function to write mols into strings

    Returns:
        List[str]: List of clean molecular strings
        List[int]: List of indices for which cleaning was successful
    """
    new_strings = []
    idx_success = []

    for idx, string in enumerate(tqdm(strings)):
        mol = read_f(string)
        mol = mol_cleaner(mol)
        if mol is not None:
            new_strings.append(write_f(mol))
            idx_success.append(idx)
    return new_strings, idx_success


def neutralize_atoms(mol: rdkit.Chem.rdchem.Mol) -> rdkit.Chem.rdchem.Mol:
    """Neutralizes mols using O'Boyle's nocharge code
    Taken from https://www.rdkit.org/docs/Cookbook.html#neutralizing-molecules
    Accessed Mar. 24th 2022

    Args:
        mol (rdkit.Chem.rdchem.Mol): rdkit mol

    Returns:
        rdkit.Chem.rdchem.Mol: Neutralized mol
    """
    pattern = MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


POS_CARBON_SMARTS = MolFromSmarts("[C+]")


def contains_positively_charged_carbon(mol: rdkit.Chem.rdchem.Mol) -> bool:
    """Checks whether `mol` contains a positively-charged carbon atom"""
    return mol.HasSubstructMatch(POS_CARBON_SMARTS)


def mol_cleaner(mol: rdkit.Chem.rdchem.Mol) -> rdkit.Chem.rdchem.Mol:
    """Wrapper function that does basic cleaning of rdkit mols:
    removes salts, neutralizes atoms and standardizes them.

    Args:
        mol ( rdkit.Chem.rdchem.Mol): rdkit mole to be cleaned

    Returns:
        rdkit.Chem.rdchem.Mol: cleaned mol
    """
    mol = Cleanup(mol)
    s_remover = SaltRemover()
    mol = s_remover.StripMol(mol)
    try:
        mol = neutralize_atoms(mol)
    except AtomValenceException:
        LOGGER.warning("AtomValenceException encountered. Returning None.")
        return None

    if contains_positively_charged_carbon(mol):
        LOGGER.warning(
            "Molecule contains positively charged carbon that could not be neutralized. Returning None."
        )
        return None

    return mol
