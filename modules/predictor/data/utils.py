"""Utils functions for data."""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

    st_features = df[numerical_features]
    scaler = StandardScaler().fit(st_features.values)
    st_features = scaler.transform(st_features.values)
    df[numerical_features] = st_features

    return df
