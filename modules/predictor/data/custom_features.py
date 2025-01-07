"""Functions for handcrafted features."""
from typing import Literal

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from modules.core.features.descriptors import extract_data_from_mol
from modules.core.features.flatness import get_flatness_smiles
from modules.core.features.pore_size import calculate_hexagonal_pore_diameter
from modules.core.features.symmetries import analyse_symmetry_point_group
from modules.core.features.utils import get_pymatgen_molecule_from_smiles
from modules.predictor.data.utils import data_preprocessing


def calculate_atom_percentage(smiles: str, atom: str) -> float | None:
    """
    Calculates the percentage of a given atom in a molecule.
    :param smiles: smiles representation of a molecule.
    :param atom: atom name
    :return: percentage of the given atom in the molecule or None
    """
    atom_num = None

    if atom == "C":
        atom_num = 6
    elif atom == "N":
        atom_num = 7
    elif atom == "O":
        atom_num = 8
    elif atom == "F":
        atom_num = 9
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        num_carbon = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == atom_num)
        num_atoms = mol.GetNumAtoms()
        if num_atoms > 0:
            return (num_carbon / num_atoms) * 100
    return None


def handcrafted_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates selected features for a given dataframe (percentage of carbon, oxygen, nitrogen, fluorine, and MolLogP descriptor)
    :param df: dataframe with smiles
    :return: dataframe with handcrafted features
    """
    df["%C"] = df["smiles"].apply(lambda smiles: calculate_atom_percentage(smiles, "C"))
    df["%N"] = df["smiles"].apply(lambda smiles: calculate_atom_percentage(smiles, "N"))
    df["%O"] = df["smiles"].apply(lambda smiles: calculate_atom_percentage(smiles, "O"))
    df["%F"] = df["smiles"].apply(lambda smiles: calculate_atom_percentage(smiles, "F"))

    df["MolLogP"] = df["smiles"].apply(lambda smiles: Descriptors.MolLogP(Chem.MolFromSmiles(smiles)))

    # Add features from extract_data_from_mol
    df = pd.concat([df, df["smiles"].apply(lambda smiles: pd.Series(extract_data_from_mol(Chem.MolFromSmiles(smiles))))], axis=1)

    return df


def check_symmetry_smiles(smiles: str, translation_table_path: str) -> str | None:
    """
    Calculates a point group for a given molecule.
    :param smiles: smiles representation of a molecule
    :param translation_table_path: path to translation table (symmetry_translation.csv file)
    :return: point group symmetry of a given molecule
    """
    mol = get_pymatgen_molecule_from_smiles(smiles, save_file=True)
    if mol is None:
        return None

    _, _, pointgroup, _ = analyse_symmetry_point_group(mol, translation_table_path)

    return pointgroup


def feature_engineering(df: pd.DataFrame, translation_table_path: str) -> pd.DataFrame:
    """
    Calculates selected features from a given dataframe with smiles (flatness, pore size, symmetry, percentages of selected atoms, and MolLogP descriptor)
    :param translation_table_path: path to translation table (symmetry_translation.csv file)
    :param df: dataframe with smiles
    :return: dataframe with selected features
    """
    # Flatness
    df["flatness"] = df["smiles"].apply(get_flatness_smiles)

    # Pore size
    df["PS"] = df["smiles"].apply(calculate_hexagonal_pore_diameter)

    # Symmetry
    df["symmetry"] = df["smiles"].apply(lambda x: check_symmetry_smiles(x, translation_table_path))

    df = handcrafted_feature_engineering(df)

    return df


def remove_unuseful_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicated and rarely present features
    :param df: dataframe with generated features
    :return: dataframe without duplicated and rarely present features
    """
    # Drop features with the same values in all rows
    df.drop(columns=["num_conformers", "is_chiral", "is_aromatic", "is_branched", "is_cyclic", "is_heterocycle", "is_homocycle"], axis=1, inplace=True)

    # Value for only 5 rows
    df.drop(columns=["num_amide_bonds", "%F"], axis=1, inplace=True)

    # Number of atoms and number of heavy atoms are the same
    df.drop(columns=["num_heavy_atoms"], axis=1, inplace=True)

    # Number of rings and number of aromatic rings are almost the same
    df.drop(columns=["num_rings"], axis=1, inplace=True)

    return df


def data_preprocessing_and_feature_engineering(
    data_path: str, data_type: Literal["expert", "zhu", "saad", "expert2", "expert3"], translation_data_path: str, save_path: str | None = None, remove_unuseful: bool = False
) -> pd.DataFrame:
    """
    Preprocesses data and performs feature engineering
    :param translation_data_path: path to translation table (symmetry_translation.csv file)
    :param data_path: path to the data file
    :param data_type: type of data, one of the following: expert, zhu, saad, expert2
    :param save_path: path to save the dataframe
    :param remove_unuseful: whether to remove duplicated and rarely present features
    :return: preprocessed dataset with generated features
    """
    df = data_preprocessing(data_path, data_type)
    df = feature_engineering(df, translation_data_path)
    # df.dropna(inplace=True)
    if remove_unuseful:
        df = remove_unuseful_features(df)
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df
