"""Functions for generating fingerprints."""
from typing import Literal

import deepchem as dc
import numpy as np
import pandas as pd


class Fingerprints:
    """
    Class for generating fingerprints.
    """

    def __init__(self):
        self.fingerprints = {
            "ecfp": self.ecfp_fingerprint,
            "maccs": self.maccs_fingerprint,
            "pubchem": self.pubchem_fingerprint,
            "rdkit": self.rdkit_fingerprint,
        }

    def apply(self, name, smiles, **kwargs):
        """
        Apply the fingerprint function with the given name.
        :param name: name of the fingerprint function.
        :param smiles: list of SMILES strings.
        :param kwargs: additional arguments for the fingerprint function.
        :return: list of fingerprints.
        """
        if name in self.fingerprints:
            return self.fingerprints[name](smiles, **kwargs["kwargs"])
        raise ValueError(f"Fingerprint function '{name}' is not defined.")

    @staticmethod
    def ecfp_fingerprint(smiles: list, size: int = 1024, radius: int = 2) -> np.ndarray:
        """
        Generates ECFP fingerprints for the given SMILES strings.
        :param radius: radius for ecfp fingerprints.
        :param size: size of ecfp fingerprints.
        :param smiles: list of SMILES strings.
        :return: list of ECFP fingerprints.
        """
        featurizer = dc.feat.CircularFingerprint(size=size, radius=radius)
        return featurizer.featurize(smiles)

    @staticmethod
    def maccs_fingerprint(smiles: list, **kwargs) -> np.ndarray:  # pylint: disable=unused-argument
        """
        Generates MACCS fingerprints for the given SMILES strings.
        :param smiles: list of SMILES strings.
        :return: list of MACCS fingerprints.
        """
        featurizer = dc.feat.MACCSKeysFingerprint()
        return featurizer.featurize(smiles)

    @staticmethod
    def pubchem_fingerprint(smiles: list, **kwargs) -> np.ndarray:  # pylint: disable=unused-argument
        """
        Generates PubChem fingerprints for the given SMILES strings.
        :param smiles: list of SMILES strings.
        :return: list of PubChem fingerprints.
        """
        featurizer = dc.feat.PubChemFingerprint()
        return featurizer.featurize(smiles)

    @staticmethod
    def rdkit_fingerprint(smiles: list, **kwargs) -> np.ndarray:  # pylint: disable=unused-argument
        """
        Generates RDKit fingerprints for the given SMILES strings.
        :param smiles: list of SMILES strings.
        :return: list of RDKit fingerprints.
        """
        featurizer = dc.feat.RDKitDescriptors()
        return featurizer.featurize(smiles)


def fingerprints_dataset(df: pd.DataFrame, smiles_col: str, target_col: str, fingerprint_type: Literal["ecfp", "maccs", "element_property", "pubchem", "rdkit"], **kwargs) -> pd.DataFrame:
    """
    Generates fingerprints for the given dataset.
    :param df: dataframe with SMILES and target columns.
    :param smiles_col: name of the column with SMILES.
    :param target_col: name of the target column.
    :param fingerprint_type: type of fingerprint to generate.
    :param kwargs: additional arguments for the fingerprint generation.
    :return: dataframe with generated fingerprints.
    """
    smiles_list = df[smiles_col].tolist()
    target_list = df[target_col].tolist()
    feature_list = Fingerprints().apply(fingerprint_type, smiles_list, **kwargs)
    features_names = [f"{fingerprint_type}_f{i}" for i in range(feature_list.shape[1])]
    df_features = pd.DataFrame(feature_list, columns=features_names)
    df_features[target_col] = target_list
    df_features[smiles_col] = smiles_list
    return df_features
