"""Utils functions for data."""
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data_for_regressors(df_to_transform: pd.DataFrame, dfs: tuple, numerical_features: list, categorical_features: list) -> pd.DataFrame:
    """
    Prepares data for regressors (standarization, one-hot encoding)
    :param df_to_transform: dataframe with molecules and features to transform.
    :param dfs: dataframes with molecules and features to fit (both to one-hot and standard scaler).
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
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=num_splits, random_state=random_state)
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
