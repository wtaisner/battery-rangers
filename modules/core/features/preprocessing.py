"""Datasets preprocessing."""
import pandas as pd
from rdkit import Chem


def drop_problematic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops molecules with problematic conversion
    :param df: dataframe with molecules
    :return: dataframe without molecules with problematic conversion
    """
    problematic_no_target = [20, 21, 22, 23, 24, 44, 46, 50, 52]
    problematic_mol_conversion = [58]
    return df.drop(problematic_no_target + problematic_mol_conversion)


def preprocess_target(df: pd.DataFrame, targets: list) -> pd.DataFrame:
    """
    Preprocessing of the target column, selects maximal obtained value per molecule
    :param df: dataframe with molecules
    :param targets: columns with target values
    :return: preprocessed dataframe
    """
    return df.loc[:, targets].max(axis=1)


def expert_dataset_preprocessing(df_experts: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing of expert data
    :param df_experts: dataframe with expert data
    :return: preprocessed dataframe
    """
    column_names_mapping = {
        "zwiazek": "substance",
        "smiles": "smiles",
        "masa_molowa": "molecular_weight",
        "liczba_pierscieni": "number_of_rings",
        "srednia_liczba_N_w_pierscieniu": "average_number_of_N_in_ring",
        "liczba_substratu_w_porze": "number_of_substrates_in_pore",
        "energia Gibbsa": "Gibbs_energy",
        "odleglosc_miedzy_najdalszymi_grupami": "distance_between_furthest_groups",
        "kondensacja": "condensation",
        "piroliza": "pyrolysis",
        "temperatura": "temperature",
        "N_perc_triaz": "N_perc_triaz",
        "wielkosc_pora teoretyczna": "PS",  # Pore Size
        "BET": "SSA",  # BET surface area
        "objetosc": "PV",  # Pore Volume
        "objetosc_mikroporow": "micropore_volume",
        "pojemnosc_3_H2SO4_CV": "capacity_H2SO4_CV",
        "pojemnosc_3_H2SO4_GCD": "capacity_H2SO4_GCD",
        "pojemnosc_3_NaOH_CV": "capacity_NaOH_CV",
        "pojemnosc_3_NaOH_GCD": "capacity_NaOH_GCD",
        "SSSS": "SSSSS",
    }

    df_experts = df_experts.rename(columns=column_names_mapping)

    # Calculate mean capacity
    df_experts["capacity_H2SO4_mean"] = df_experts.loc[:, ["capacity_H2SO4_CV", "capacity_H2SO4_GCD"]].mean(axis=1)
    df_experts["capacity_NaOH_mean"] = df_experts.loc[:, ["capacity_NaOH_CV", "capacity_NaOH_GCD"]].mean(axis=1)
    df_experts["capacity_mean"] = df_experts.loc[:, ["capacity_H2SO4_mean", "capacity_NaOH_mean"]].mean(axis=1)

    # Drop problematic rows
    df_experts = drop_problematic(df_experts)

    # Process target
    targets = ["capacity_H2SO4_CV", "capacity_H2SO4_GCD", "capacity_NaOH_CV", "capacity_NaOH_GCD"]
    df_experts["capacity_max"] = preprocess_target(df_experts, targets=targets)

    df_experts = df_experts.loc[~df_experts["capacity_max"].isna()].reset_index(drop=True)

    df_experts["smiles"] = df_experts["smiles"].apply(Chem.CanonSmiles)

    return df_experts


def zhu_dataset_preprocessing(df_zhu: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing of literature data (Zhu)
    :param df_zhu: dataframe with Zhu data
    :return: preprocessed dataframe
    """
    column_names_mapping = {
        "SMILES": "smiles",
        "Capacitance (F/g)": "capacity_max",
    }
    df_zhu = df_zhu.rename(columns=column_names_mapping)

    # Drop problematic rows
    problematic_mol_conversion = [2, 31, 32, 33, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 97, 98]
    df_zhu = df_zhu.drop(problematic_mol_conversion)

    # Process target
    df_zhu = df_zhu.loc[~df_zhu["capacity_max"].isna()].reset_index(drop=True)
    df_zhu["smiles"] = df_zhu["smiles"].apply(Chem.CanonSmiles)

    return df_zhu


def saad_dataset_preprocessing(df_saad: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing of literature data (Saad)
    :param df_saad: pandas dataframe with Saad data
    :return: preprocessed dataframe
    """
    column_names_mapping = {"SMILES": "smiles", "Capacitance (F/g)": "capacity_max"}
    df_saad = df_saad.rename(columns=column_names_mapping)

    # Drop problematic rows
    problematic_mol_conversion = [6, 7, 8, 9]
    problematic_feature_engineering = [0, 1, 2, 3, 4, 5]
    df_saad = df_saad.drop(problematic_mol_conversion + problematic_feature_engineering)

    # Process target
    df_saad = df_saad.loc[~df_saad["capacity_max"].isna()].reset_index(drop=True)

    df_saad["smiles"] = df_saad["smiles"].apply(Chem.CanonSmiles)

    return df_saad
