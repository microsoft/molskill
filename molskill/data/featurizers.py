import abc
import os
from functools import partial
from typing import Dict, List, Optional, Type, Union

import numpy as np
import rdkit
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP, GetAvalonFP
from rdkit.Chem import AllChem, DataStructs
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


DESCRIPTORS_RDKIT: List[str] = [
    "MaxAbsEStateIndex",
    "MaxEStateIndex",
    "MinAbsEStateIndex",
    "MinEStateIndex",
    "qed",
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "BCUT2D_MWHI",
    "BCUT2D_MWLOW",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",
    "BCUT2D_MRLOW",
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "PEOE_VSA1",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA12",
    "PEOE_VSA13",
    "PEOE_VSA14",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA4",
    "PEOE_VSA5",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "SMR_VSA1",
    "SMR_VSA10",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA9",
    "SlogP_VSA1",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA7",
    "SlogP_VSA8",
    "TPSA",
    "EState_VSA1",
    "EState_VSA10",
    "EState_VSA11",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "VSA_EState1",
    "VSA_EState10",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "RingCount",
    "MolLogP",
    "MolMR",
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_bicyclic",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_halogen",
    "fr_hdrzine",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_piperdine",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfonamd",
    "fr_unbrch_alkane",
    "fr_urea",
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
    def __init__(
        self,
        desc_list: Optional[List[str]] = None,
        normalize: bool = False,
        moment_csv: Optional[Union[os.PathLike, str]] = None,
    ):
        """RDKit 2d descriptor featurizer
        Inspired by https://github.com/EBjerrum/scikit-mol/blob/main/scikit_mol/descriptors.py

        Args:
            desc_list (Optional[List[str]], optional): List of descriptors to be used.
            normalize (bool, optional): Whether to normalize descriptors with precomputed ChEMBL population mean and variance.
        """
        self.desc_list = DESCRIPTORS_RDKIT if desc_list is None else desc_list

        _validate_rdkit_descriptors(self.desc_list)
        self.calculators = MolecularDescriptorCalculator(self.desc_list)
        self.normalize = normalize

        if self.normalize:
            self.moments = get_population_moments(
                desc_list=self.desc_list, moment_csv=moment_csv
            )

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


def _validate_rdkit_descriptors(desc_list: List[str]) -> None:
    """
    Sanity checks that the provided descriptor names are valid.
    """
    unknown_descriptors = [
        desc_name for desc_name in desc_list if desc_name not in DESCRIPTORS_RDKIT
    ]
    assert not unknown_descriptors, f"""Unknown descriptor names {unknown_descriptors} specified,\n
    Must be a combination of: {DESCRIPTORS_RDKIT}"""


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
