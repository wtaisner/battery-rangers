"""Utils functions for data."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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

    for c in categorical_features:
        if c in numerical_features:
            numerical_features.remove(c)
            numerical_features.extend(enc.get_feature_names_out([c]))

    st_features = df[numerical_features]
    scaler = StandardScaler().fit(st_features.values)
    st_features = scaler.transform(st_features.values)
    df[numerical_features] = st_features

    return df


def custom_data_split(df: pd.DataFrame, target: str, num_splits: int, num_bins: int = 4, random_state: int = 23) -> list:
    """
    Performs custom data split on the provided data
    :param df: dataframe with molecules and generated features
    :param target: name of the ratger feature
    :param num_splits: number of folds
    :param num_bins: number of capacity bins to use
    :param random_state: random state (default: 23)
    :return: generated splits (indices)
    """
    features = [f for f in df.columns if f != target]
    binned_capacity = pd.qcut(df[target], q=num_bins, labels=False)
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    kfolds = list(skf.split(df[features], binned_capacity))
    return kfolds


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
