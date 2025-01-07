"""Script for generating comparison matrix."""
import os

import pandas as pd
import yaml

from modules.predictor.data.utils import combine_split, custom_data_kfold  # pylint: disable=import-error
from modules.predictor.training_and_evaluation.train_eval_pipeline import (  # pylint: disable=import-error
    dataset_preprocess_and_train,
)

if __name__ == "__main__":
    CONFIG_PATH = "../configs/comparison_matrix.yaml"
    with open(CONFIG_PATH, encoding="UTF-8") as file:
        config = yaml.safe_load(file)

    root_data_dir = config["root_dir"]
    os.makedirs(config["save_matrix_dir"], exist_ok=True)
    save_matrix_path = os.path.join(config["save_matrix_dir"], f"{config['combo_name']}.csv")

    COMBO_NAME = config["combo_name"]
    data_csvs = config["data_dict"]
    feature_dict = config["features_dict"]

    dataset_combos = config["dataset_combos"]
    feature_combos = config["feature_combos"]

    feature_combos = [[group[0], group[1][i]] for group in feature_combos for i in range(len(group[1]))]

    full_results = []

    cat_features_dict = config["cat_features_dict"]

    target = config["target"]
    num_splits = config["num_splits"]
    num_bins = config["num_bins"]
    random_state = config["random_state"]

    binary_features = {}
    for features, data_path in feature_dict.items():
        df = pd.read_csv(os.path.join(root_data_dir, data_path, data_csvs["expert"]))
        binary_features[features] = [c for c in df.columns if (df[c].nunique() <= 2 and set(df[c].dropna().unique()).issubset({0, 1}))]

    for datasets in dataset_combos:
        for combo in feature_combos:
            combo = list(set(combo))

            dataset_names = [data_csvs[dataset] for dataset in datasets]
            dataset_paths = {data_name: [os.path.join(root_data_dir, feature_dict[c], data_csvs[data_name]) for c in combo] for data_name in datasets}

            dfs = {}
            try:
                for data_name, paths in dataset_paths.items():
                    df = [pd.read_csv(data_path) for data_path in paths]
                    df = pd.concat(df, axis=1).reset_index(drop=True)
                    df.columns = df.columns.str.replace(r"[\[\]>]", "", regex=True)
                    df = df.loc[:, ~df.columns.duplicated()].sort_values(by="smiles").reset_index(drop=True)
                    dfs[data_name] = df
            except FileNotFoundError:
                for model in ["knn", "xgboost", "random_forest", "lasso"]:
                    full_results.append(
                        {
                            "model": model,
                            "features": "_".join(combo),
                            "datasets": "_".join(datasets),
                        }
                    )
                continue

            if len(datasets) > 1:
                df_rest = pd.concat([df for data_name, df in dfs.items() if data_name != "expert"], ignore_index=True).sort_values(by="smiles").reset_index(drop=True)
            else:
                df_rest = None  # pylint: disable=invalid-name
            df_expert1 = dfs["expert"]

            folds_experts1 = custom_data_kfold(df_expert1, target, num_splits, num_bins, random_state)
            if df_rest is not None:
                folds_all, df_all = combine_split(df_expert1, folds_experts1, df_rest)
            else:
                folds_all = folds_experts1
                df_all = df_expert1

            if "custom" in combo and "symmetry" in df_all.columns:
                cat_features = cat_features_dict["custom"]
            else:
                cat_features = []
            bin_features = [binary_features[c] for c in combo]
            num_features = [c for c in df_all.columns if c not in ["smiles", target, *cat_features, *bin_features]]

            DATASET_NAME = "_".join(datasets)
            COMBO_NAME = "_".join(combo)
            DATANAME = f"{DATASET_NAME}_{COMBO_NAME}"
            results = dataset_preprocess_and_train(df_all, DATANAME, cat_features, num_features, folds_all, target)
            for key, value in results.items():
                results_model = value
                results_model["model"] = key
                results_model["features"] = COMBO_NAME
                results_model["datasets"] = DATASET_NAME
                full_results.append(results_model)

    full_results = pd.DataFrame(full_results)
    full_results.to_csv(save_matrix_path, index=False)
