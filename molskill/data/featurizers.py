import abc
from functools import partial
from typing import Dict, List, Optional, Set, Type

import numpy as np
import rdkit
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP, GetAvalonFP
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from molskill.data.standardization import get_population_moments


class Featurizer(abc.ABC):
    def __init__(self) -> None:
        """Base featurizer class"""
        super().__init__()

    def get_feat(self, mol: rdkit.Chem.rdchem.Mol) -> np.ndarray:
        """Base method to compute fingerprints

        Args:
            mol (rdkit.Chem.rdchem.Mol)s
        """
        raise NotImplementedError()

    def dim(self) -> int:
        """Size of the returned feature"""
        raise NotImplementedError()


class FingerprintFeaturizer(Featurizer):
    def __init__(self, nbits: int) -> None:
        """Base fingerprint class

        Args:
            nbits (int): Fingerprint length
        """
        self.nbits = nbits
        super().__init__()

    def dim(self) -> int:
        return self.nbits


AVAILABLE_FP_FEATURIZERS: Dict[str, Type[FingerprintFeaturizer]] = {}
AVAILABLE_FEATURIZERS: Dict[str, Type[Featurizer]] = {}

EXCLUDE_DESCRIPTORS_RDKIT: Set[str] = set(
    [
        "SMR_VSA8",
        "SlogP_VSA9",
        "fr_diazo",
        "fr_isocyan",
        "fr_isocyan",
        "fr_isothiocyan",
        "fr_prisulfonamd",
        "fr_thiocyan",
        "fr_AI_COO",
        "fr_ArN",
        "fr_Ar_COO",
        "fr_C_S",
        "fr_HOCCN",
        "fr_Imine",
        "fr_N_O",
        "fr_SH",
        "fr_aldehyde",
        "fr_benzodiazepine",
        "fr_furan",
        "fr_guanido",
        "fr_hdrzone",
        "fr_imidazole",
        "fr_imide",
        "fr_morpholine",
        "fr_nitrile",
        "fr_oxazole",
        "fr_oxime",
        "fr_phos_acid",
        "fr_phos_ester",
        "fr_piperzine",
        "fr_priamide",
        "fr_sulfide",
        "fr_sulfone",
        "fr_term_acetylene",
        "fr_tetrazole",
        "fr_thiazole",
        "fr_thiophene",
        "Ipc",
        "SubstructureMatches",
        "Min_N_O_filter",
        "Frac_N_O",
        "Covalent",
        "SpecialMol",
        "SeverityScore",
        "SeverityComment",
    ]
)
DESCRIPTORS_RDKIT = [
    descriptor[0]
    for descriptor in Descriptors._descList
    if descriptor[0] not in EXCLUDE_DESCRIPTORS_RDKIT
]


def register_featurizer(name: str):
    def register_function(cls: Type[Featurizer]):
        if issubclass(cls, FingerprintFeaturizer):
            AVAILABLE_FP_FEATURIZERS[name] = partial(cls, count=False)
            AVAILABLE_FP_FEATURIZERS[name + "_count"] = partial(cls, count=True)
        elif issubclass(cls, Featurizer) and not issubclass(cls, FingerprintFeaturizer):
            # rdkit descriptors
            AVAILABLE_FEATURIZERS[name] = partial(cls, desc_list=DESCRIPTORS_RDKIT)
            AVAILABLE_FEATURIZERS[name + "_norm"] = partial(
                cls, desc_list=DESCRIPTORS_RDKIT, normalize=True
            )
        else:
            raise ValueError("Not recognized descriptor type.")
        return cls

    return register_function


class MultiFeaturizer(Featurizer):
    def __init__(self, featurizers: List[Featurizer]):
        """Class for multiple concatenated features coming from
        different featurizers.

        Args:
            featurizers (List[Featurizer]): A list of featurizers
        """
        self.featurizers = featurizers
        self.feat_size = 0

        for featurizer in self.featurizers:
            self.feat_size += featurizer.dim()

    def get_feat(self, mol):
        return np.concatenate(
            [featurizer.get_feat(mol) for featurizer in self.featurizers], axis=0
        )

    def dim(self):
        return self.feat_size


@register_featurizer(name="morgan")
class MorganFingerprint(FingerprintFeaturizer):
    def __init__(
        self, nbits: int = 2048, bond_radius: int = 2, count: bool = False
    ) -> None:
        """Base class for Morgan fingerprinting featurizer

        Args:
            bond_radius (int, optional): Bond radius. Defaults to 2.
            count (bool, optional): Whether to use count fingerprints. Defaults to False.
        """
        self.bond_radius = bond_radius
        self.count = count
        super().__init__(nbits=nbits)

    def get_feat(self, mol: rdkit.Chem.rdchem.Mol) -> np.ndarray:
        fp_fun = (
            AllChem.GetHashedMorganFingerprint
            if self.count
            else AllChem.GetMorganFingerprintAsBitVect
        )
        fp = fp_fun(mol, self.bond_radius, self.nbits)
        arr = np.zeros((1,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr


@register_featurizer(name="avalon")
class AvalonFingerprint(FingerprintFeaturizer):
    def __init__(self, nbits: int = 2048, count: bool = False) -> None:
        """Base class for Avalon fingerprints

        Args:
            count (bool, optional): Whether to use count fingerprints.. Defaults to False.
        """
        self.count = count
        super().__init__(nbits=nbits)

    def get_feat(self, mol: rdkit.Chem.rdchem.Mol) -> np.ndarray:
        fp_fun = GetAvalonCountFP if self.count else GetAvalonFP
        fp = fp_fun(mol, nBits=self.nbits)
        arr = np.zeros((1,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr


@register_featurizer(name="rdkit_2d")
class Rdkit2dDescriptor(Featurizer):
    def __init__(self, desc_list: Optional[List[str]] = None, normalize: bool = False):
        """RDKit 2d descriptor featurizer
        Inspired by https://github.com/EBjerrum/scikit-mol/blob/main/scikit_mol/descriptors.py

        Args:
            desc_list (Optional[List[str]], optional): List of descriptors to be used.
            normalize (bool, optional): Whether to normalize descriptors with precomputed ChEMBL population mean and variance.
        """
        if desc_list is None:
            desc_list = DESCRIPTORS_RDKIT

        self.desc_list = self._get_valid_descriptors(desc_list)
        self.calculators = MolecularDescriptorCalculator(self.desc_list)
        self.normalize = normalize

        if self.normalize:
            self.moments = get_population_moments(desc_list=self.desc_list)

    def _get_valid_descriptors(self, desc_list: List[str]) -> List[str]:
        """
        Sanity checks that the provided descriptor names are valid.
        """
        unknown_descriptors = [
            desc_name
            for desc_name in desc_list
            if desc_name not in self.available_descriptors()
        ]
        assert not unknown_descriptors, f"""Unknown descriptor names {unknown_descriptors} specified,\n
        Must be a combination of: {self.available_descriptors}"""
        return desc_list

    @staticmethod
    def available_descriptors() -> List[str]:
        """Lists all available descriptor names that can be used"""
        return DESCRIPTORS_RDKIT

    @property
    def selected_descriptors(self) -> List[str]:
        """Property to get list of the selected descriptor names"""
        return list(self.calculators.GetDescriptorNames())

    def get_feat(self, mol: rdkit.Chem.rdchem.Mol) -> np.ndarray:
        feat = np.array(list(self.calculators.CalcDescriptors(mol)), dtype=np.float32)
        if self.normalize:
            feat -= self.moments["mean"]
            feat /= self.moments["std"]
        return feat

    def dim(self) -> int:
        return len(self.desc_list)


def get_featurizer(featurizer_name: str, **kwargs) -> Featurizer:
    """Basic factory function for fp featurizers"""
    return AVAILABLE_FEATURIZERS[featurizer_name](**kwargs)


AVAILABLE_FEATURIZERS |= AVAILABLE_FP_FEATURIZERS
AVAILABLE_FEATURIZERS |= {
    "morgan_count_rdkit_2d": partial(
        MultiFeaturizer,
        featurizers=[
            MorganFingerprint(count=True),
            Rdkit2dDescriptor(desc_list=DESCRIPTORS_RDKIT),
        ],
    ),
    "morgan_count_rdkit_2d_norm": partial(
        MultiFeaturizer,
        featurizers=[
            MorganFingerprint(count=True),
            Rdkit2dDescriptor(desc_list=DESCRIPTORS_RDKIT, normalize=True),
        ],
    ),
}
