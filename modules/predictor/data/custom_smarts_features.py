"""Contains custom SMARTS features for molecular prediction tasks."""
import pandas as pd
from rdkit import Chem


def _get_experts_patterns() -> dict[str, str]:
    """Generate a dictionary of expert SMARTS patterns for feature extraction."""
    patterns = [
        "[#6]1=[#6]2[#6]([#6]=[#6][#6]=[#6]2)=[#6][#6]=[#6]1",
        "[#6]1=[#6]2[#6]([#6]=[#6][#6]2)=[#6][#6]=[#6]1",
        "[#6]@[#6](@[#6])(@[#6])",
        "[#6]1=[#6]([#6]2=[#6][#6]=[#6][#6]=[#6]2)[#6]=[#6][#6]=[#6]1",
        "[#6]1=[#6][#6]=[#6][#7]=[#6]1",
        "[#6]1=[#6][#7]=[#6][#7]=[#6]1",
        "[#7]1=[#6][#7]=[#6][#7]=[#6]1",
        "[#6]1=[#6][#6]=[#16][#6]1",
        "[#6](=[#7]-[#6]=*)(-[#6]=*)-[#7](-[#6]=*)(-[#6]=*)",
        "[#6]1[#6]=[#6](-[#6]#[#7])[#6]=[#6][#6]=1",
        "*=[#6](-*)(-[#6]#[#7])",
        "*-[#6]=[#6]-[#6]=[#6]-*",
        "*-[#6]=[#7]-[#6]=[#6]-*",
        "*-[#7]=[#7]-*",
        "*=[#6]-[#6](=[#8])-[#6]=*",
        "[#6](=*)-[#7](-[#6]=*)(-[#6]=*)",
        "*=[#6]-[#6](=[#8])-[#7](-[#6]=*)(-[#6]=*)",
    ]
    patterns = {f"expert_feature_{i}": p for i, p in enumerate(patterns)}
    return patterns


def _get_custom_patterns() -> dict[str, str]:
    """Generate a dictionary of custom SMARTS patterns for feature extraction."""
    patterns = []
    patterns = {f"custom_feature_{i}": p for i, p in enumerate(patterns)}
    return patterns


def _get_smarts_features(smiles: str, patterns: dict) -> dict[str, float]:
    """Extract custom SMARTS features from a SMILES string.
    :param smiles: A SMILES string representing a molecule.
    :param patterns: A dictionary of SMARTS patterns where keys are feature names and values are SMARTS strings.
    :return: A dictionary of features where keys are feature names and values are counts of matches.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    features = {}
    for feature_name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        matches = mol.GetSubstructMatches(pattern)
        features[feature_name] = len(matches)
    return features


def _get_smarts_dataset(df: pd.DataFrame, smiles_col: str, target_col: str, patterns: dict, count: bool = False) -> pd.DataFrame:
    """
    Generate a dataset with custom SMARTS features from a DataFrame.
    :param df: DataFrame containing the data.
    :param smiles_col: String column name containing SMILES representations of molecules.
    :param target_col: String column name for the target variable.
    :param patterns: Dictionary of SMARTS patterns where keys are feature names and values are SMARTS strings.
    :param count: Whether to return counts of matches (True) or binary presence/absence (False).
    :return: DataFrame with custom SMARTS features.
    """
    smiles_list = df[smiles_col].tolist()
    target_list = df[target_col].tolist()
    features_list = []
    for smiles in smiles_list:
        features = _get_smarts_features(smiles, patterns)
        features_list.append(features)
    features_df = pd.DataFrame(features_list)
    if not count:
        features_df = features_df > 0
        features_df = features_df.astype(int)
    features_df[target_col] = target_list
    features_df[smiles_col] = smiles_list
    return features_df


def expert_features_dataset(df: pd.DataFrame, smiles_col: str = "smiles", target_col: str = "target", count: bool = False) -> pd.DataFrame:
    """
    Generate a dataset with expert SMARTS features.

    :param df: DataFrame containing the data.
    :param smiles_col: Column name containing SMILES strings.
    :param target_col: Column name for the target variable.
    :param count: If True, counts of matches are returned; if False, binary presence/absence is returned.
    :return: DataFrame with custom SMARTS features.
    """
    return _get_smarts_dataset(df, smiles_col, target_col, _get_experts_patterns(), count)


def custom_features_dataset(df: pd.DataFrame, smiles_col: str = "smiles", target_col: str = "target", count: bool = False) -> pd.DataFrame:
    """
    Generate a dataset with custom SMARTS features.

    :param df: DataFrame containing the data.
    :param smiles_col: Column name containing SMILES strings.
    :param target_col: Column name for the target variable.
    :param count: If True, counts of matches are returned; if False, binary presence/absence is returned.
    :return: DataFrame with custom SMARTS features.
    """
    return _get_smarts_dataset(df, smiles_col, target_col, _get_custom_patterns(), count)
