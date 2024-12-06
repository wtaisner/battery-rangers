"""Utils functions for data."""
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modules.core.features.preprocessing import (
    canon_smiles,
    expert_dataset_preprocessing,
    saad_dataset_preprocessing,
    zhu_dataset_preprocessing,
)


def data_preprocessing(data_path: str, data_type: Literal["expert", "zhu", "saad", "expert2"]) -> pd.DataFrame:
    """
    preprocessing for the datasets
    :param data_path: path to data
    :param data_type: type of data (expert, ahu, saad, expert2)
    :return: pre-processed data
    """
    df = pd.read_csv(data_path)

    if data_type == "expert":
        df = expert_dataset_preprocessing(df)
    elif data_type == "zhu":
        df = zhu_dataset_preprocessing(df)
    elif data_type == "saad":
        df = saad_dataset_preprocessing(df)
    df = df.loc[:, ["smiles", "capacity_max"]]
    df = df.groupby("smiles").max("capacity_max").reset_index()
    if data_type == "expert2":
        df["smiles"] = df["smiles"].apply(canon_smiles)
        df.dropna(subset=["smiles"], inplace=True)
    return df


def complex_data_conversion(df: pd.DataFrame, features_list: list) -> tuple:
    """
    Converts complex numbers to two columns (one with real part, one with imaginary part)
    :param df: dataframe
    :param features_list: list of features, which contain complex numbers
    :return: dataframe after conversion with dropped original features, new features names
    """
    new_features = []
    for feature in features_list:
        values = df[feature].tolist()
        real, imag = [], []
        for v in values:
            if isinstance(v, tuple):
                v = v[0]
            v = complex(v)
            real.append(np.real(v))
            imag.append(np.imag(v))
        if len(np.unique(real)) > 1:
            new_features.append(f"{feature}_real")
            df[f"{feature}_real"] = real
        if len(np.unique(imag)) > 1:
            new_features.append(f"{feature}_imag")
            df[f"{feature}_imag"] = imag
        df.drop(columns=[feature], inplace=True)
    return df, new_features


def prepare_data_for_regressors(df: pd.DataFrame, numerical_features: list, categorical_features: list) -> pd.DataFrame:
    """
    Prepares data for regressors (standarization, one-hot encoding)
    :param df: dataframe with molecules and features.
    :param numerical_features: features to standardize (numerical).
    :param categorical_features: features to encode (categorical).
    :return: standardized dataframe
    """
    enc_features = df[categorical_features]
    enc = OneHotEncoder(handle_unknown="ignore")
    enc_features = enc.fit_transform(enc_features).toarray()
    enc_features_names = enc.get_feature_names_out(categorical_features)
    df[enc_features_names] = enc_features
    df.drop(categorical_features, axis=1, inplace=True)

    for c in categorical_features:
        if c in numerical_features:
            numerical_features.remove(c)
            numerical_features.extend(enc.get_feature_names_out([c]))

    st_features = df[numerical_features]
    scaler = StandardScaler().fit(st_features.values)
    st_features = scaler.transform(st_features.values)
    df[numerical_features] = st_features

    return df


def custom_discretization(df: pd.DataFrame, target: str, num_bins: int = 4) -> pd.DataFrame:
    """
    Discretization of the continuous target attribute.
    :param df: dataframe with molecules and target
    :param target: name of the target column
    :param num_bins: number of capacity bins to use
    :return: discretized target attribute
    """
    binned_capacity = pd.qcut(df[target], q=num_bins, labels=False)
    return binned_capacity


def custom_data_kfold(df: pd.DataFrame, target: str, num_splits: int, num_bins: int = 4, random_state: int = 42) -> list:
    """
    Performs custom data split on the provided data
    :param df: dataframe with molecules and generated features
    :param target: name of the target feature
    :param num_splits: number of folds
    :param num_bins: number of capacity bins to use
    :param random_state: random state (default: 23)
    :return: generated splits (indices)
    """
    features = [f for f in df.columns if f != target]
    binned_capacity = custom_discretization(df, target, num_bins)
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    kfolds = list(skf.split(df[features], binned_capacity))
    return kfolds


def custom_data_split(df: pd.DataFrame, target: str, train_size: float, num_bins: int = 4, random_state: int = 42) -> list:
    """

    :param df: dataframe
    :param target: name of the target feature
    :param train_size: required train size (percentage
    :param num_bins: number of bins for disretization
    :param random_state: random state (default: 42)
    :return: custom splits (indices)
    """
    idx = df.index.tolist()
    binned_capacity = custom_discretization(df, target, num_bins)
    train_ids, test_ids = train_test_split(idx, train_size=train_size, random_state=random_state, shuffle=True, stratify=binned_capacity)
    return [(train_ids, test_ids)]


def combine_split(df1: pd.DataFrame, split1: list, df2: pd.DataFrame, split2: list) -> tuple:
    """
    Combines two dataframes into one dataframe, combines splits of these dataframes.
    :param df1: first dataframe.
    :param split1: split of the first dataframe.
    :param df2: second dataframe.
    :param split2: split of the second dataframe.
    :return: list with splits, combined dataframe.
    """
    combined_split = []
    index_df1 = ["1_" + str(idx) for idx in df1.index]
    index_df2 = ["2_" + str(idx) for idx in df2.index]
    index_original = index_df1 + index_df2
    df_combined = pd.concat([df1, df2], ignore_index=True)
    index_mapping = pd.DataFrame({"ids_org": index_original, "ids_new": df_combined.index})
    for (train_idx1, test_idx1), (train_idx2, test_idx2) in zip(split1, split2):
        train1 = index_mapping.loc[index_mapping.ids_org.isin(["1_" + str(idx) for idx in train_idx1]), "ids_new"].tolist()
        test1 = index_mapping.loc[index_mapping.ids_org.isin(["1_" + str(idx) for idx in test_idx1]), "ids_new"].tolist()
        train2 = index_mapping.loc[index_mapping.ids_org.isin(["2_" + str(idx) for idx in train_idx2]), "ids_new"].tolist()
        test2 = index_mapping.loc[index_mapping.ids_org.isin(["2_" + str(idx) for idx in test_idx2]), "ids_new"].tolist()
        combined_split.append((train1 + train2, test1 + test2))
    return combined_split, df_combined
