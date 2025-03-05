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

# pylint: disable=invalid-name


def data_preprocessing(data_path: str, data_type: Literal["expert", "zhu", "saad", "expert2", "expert3"]) -> pd.DataFrame:
    """
    preprocessing for the datasets
    :param data_path: path to data
    :param data_type: type of data (expert, ahu, saad, expert2)
    :return: pre-processed data
    """
    df = pd.read_csv(data_path)

    if data_type in ["expert", "expert3"]:
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


def prepare_data_for_regressors(df_to_transform: pd.DataFrame, dfs: tuple, numerical_features: list, categorical_features: list) -> pd.DataFrame:
    """
    Prepares data for regressors (standarization, one-hot encoding)
    :param df_to_transform: dataframe with molecules and features to transform.
    :param dfs: dataframes with molecules and features to fit (both to one-hot, first to .
    :param numerical_features: features to standardize (numerical).
    :param categorical_features: features to encode (categorical).
    :return: standardized dataframe
    """
    df, _ = dfs
    if len(categorical_features) > 0:
        enc_features = pd.concat(dfs)[categorical_features]
        c_features = []
        for c in enc_features.columns:
            c_features.append(enc_features[c].unique())
        enc_features_to_transform = df_to_transform[categorical_features]
        enc = OneHotEncoder(drop="first", categories=c_features)
        enc.fit(enc_features)
        enc_features_to_transform = enc.transform(enc_features_to_transform).toarray()
        enc_features_names = enc.get_feature_names_out(categorical_features)
        df_to_transform[enc_features_names] = enc_features_to_transform
        df_to_transform.drop(categorical_features, axis=1, inplace=True)

    for c in categorical_features:
        if c in numerical_features:
            numerical_features.remove(c)

    if len(numerical_features) > 0:
        st_features = df[numerical_features]
        st_features_to_transform = df_to_transform[numerical_features]
        scaler = StandardScaler().fit(st_features.values)
        st_features_to_transform = scaler.transform(st_features_to_transform.values)
        df_to_transform[numerical_features] = st_features_to_transform

    return df_to_transform


def custom_discretization(y: pd.DataFrame, num_bins: int = 4) -> pd.DataFrame:
    """
    Discretization of the continuous target attribute.
    :param y: dataframe with target
    :param num_bins: number of capacity bins to use
    :return: discretized target attribute
    """
    binned_capacity = pd.qcut(y[y.columns[0]], q=num_bins, labels=False)
    return binned_capacity


def custom_data_kfold(X: pd.DataFrame, y: pd.DataFrame, num_splits: int, num_bins: int = 4, random_state: int = 42) -> list:
    """
    Performs custom data split on the provided data
    :param X: dataframe with molecules and generated features
    :param y: dataframe with target
    :param num_splits: number of folds
    :param num_bins: number of capacity bins to use
    :param random_state: random state (default: 23)
    :return: generated splits (indices)
    """
    binned_capacity = custom_discretization(y, num_bins)
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    kfolds = list(skf.split(X, binned_capacity))
    return kfolds


def custom_data_split(X: pd.DataFrame, y: pd.DataFrame, train_size: float, num_bins: int = 4, random_state: int = 42) -> list:
    """

    :param X: dataframe with features
    :param y:dataframe with target
    :param train_size: required train size (percentage
    :param num_bins: number of bins for disretization
    :param random_state: random state (default: 42)
    :return: custom splits (indices)
    """
    idx = X.index.tolist()
    binned_capacity = custom_discretization(y, num_bins)
    train_ids, test_ids = train_test_split(idx, train_size=train_size, random_state=random_state, shuffle=True, stratify=binned_capacity)
    return [(train_ids, test_ids)]


def combine_split(df1: pd.DataFrame, split1: list, df2: pd.DataFrame) -> tuple:
    """
    Combines two dataframes into one dataframe, combines splits of these dataframes.
    :param df1: first dataframe.
    :param split1: split of the first dataframe.
    :param df2: second dataframe.
    :return: list with splits, combined dataframe.
    """
    combined_split = []
    index_df1 = ["1_" + str(idx) for idx in df1.index]
    index_df2 = ["2_" + str(idx) for idx in df2.index]
    index_original = index_df1 + index_df2
    df_combined = pd.concat([df1, df2], ignore_index=True, join="outer").reset_index(drop=True)
    df_combined.fillna(0, inplace=True)
    index_mapping = pd.DataFrame({"ids_org": index_original, "ids_new": df_combined.index})
    for train_idx1, test_idx1 in split1:
        train1 = index_mapping.loc[index_mapping.ids_org.isin(["1_" + str(idx) for idx in train_idx1]), "ids_new"].tolist()
        test1 = index_mapping.loc[index_mapping.ids_org.isin(["1_" + str(idx) for idx in test_idx1]), "ids_new"].tolist()
        train2 = index_mapping.loc[index_mapping["ids_org"].str.startswith("2_"), "ids_new"].tolist()
        combined_split.append((train1 + train2, test1))
    return combined_split, df_combined
