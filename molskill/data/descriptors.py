# flake8: noqa
import os
import sys
from collections import defaultdict
from typing import Callable, List, Tuple

import numpy as np
import rdkit
from rdkit.Chem import Descriptors, MolFromSmiles, RDConfig
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm

from molskill.helpers.logging import get_logger

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # type: ignore

LOGGER = get_logger(__name__)


def get_num_fused_rings(mol: rdkit.Chem.rdchem.Mol) -> int:
    """Maximum number of fused rings in `mol`"""
    ring_info = mol.GetRingInfo()
    # finding all pairs of fused rings
    bond_rings = defaultdict(set)
    for ring_id, bond_ids in enumerate(ring_info.BondRings()):
        for bond_id in bond_ids:
            bond_rings[bond_id].add(ring_id)
    ring_pairs = [ring_ids for ring_ids in bond_rings.values() if len(ring_ids) == 2]
    # finding connected components
    components = []
    for ring_pair in ring_pairs:
        for component in components:
            if component & ring_pair:
                component |= ring_pair
                break
        else:
            components.append(ring_pair)
    # return the number of rings in the largest component
    # or 0, if there are no rings
    return max((len(x) for x in components), default=0)


def get_rdkit2D_desc(
    molrpr: List[str], read_f: Callable = MolFromSmiles
) -> Tuple[np.ndarray, List[str]]:
    """
    Taken from https://drzinph.com/rdkit_2d-descriptors-in-python-part-4/
    Assessed by OHC in May 2022

    args:
        molrpr: list of molecular representation strings. e.g., smiles
        read_f: callable function to read mol representation
    returns:
        np.ndarray: (n_molrpr x n_descriptors) sized calculated rdkit descriptors
        List[str]: List of descriptor names

    """
    mols = [read_f(rpr) for rpr in molrpr]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(
        [x[0] for x in Descriptors._descList]
    )
    desc_nms = list(calc.GetDescriptorNames())
    desc_nms.extend(["sascore", "num_fused_rings"])

    rdkit_2d_desc = np.zeros((len(mols), len(desc_nms)), dtype=np.float32)
    for ii, mol in tqdm(enumerate(mols), total=len(mols)):
        ds = list(calc.CalcDescriptors(mol))
        ds.append(sascorer.calculateScore(mol))
        ds.append(get_num_fused_rings(mol))
        rdkit_2d_desc[ii, :] = ds

    return rdkit_2d_desc, desc_nms
