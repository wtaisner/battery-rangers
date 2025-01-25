""" Script for training models on different datasets and feature sets """
import os
from datetime import datetime

import pandas as pd
import yaml

from modules.predictor.data.utils import combine_split, custom_data_kfold
from modules.predictor.training_and_evaluation.train_eval import CrossValidationPipeline

if __name__ == "__main__":
    CONFIG_PATH = "../configs/feature_selection.yaml"
    with open(CONFIG_PATH, encoding="UTF-8") as file:
        config = yaml.safe_load(file)

    metrics = config["metrics"]
    models = config["models"]
    root_data_dir = config["root_dir"]
    os.makedirs(config["save_dir"], exist_ok=True)
    save_path = os.path.join(config["save_dir"], f"{config['combo_name']}.csv")
    num_splits, num_bins, random_state = config["num_splits"], config["num_bins"], config["random_state"]
    target = config["target"]
    dataset_combos = config["dataset_combos"]
    primary_data = config["selected_dict"]["primary"]
    secondary_data = config["selected_dict"]["to_add"]

    cat_features_dict = config["cat_features_dict"]

    feature_dict = config["features_dict"]

    data_csvs = config["data_dict"]

    full_results = []
    full_selected = []

    binary_features = []
    fixed_features = []
    for features, data_path in feature_dict.items():
        if features in [primary_data, secondary_data]:
            df = pd.read_csv(os.path.join(root_data_dir, data_path, data_csvs["expert"]))
            binary_features.extend([c for c in df.columns if (df[c].nunique() <= 2 and set(df[c].dropna().unique()).issubset({0, 1}))])
            if features == primary_data:
                fixed_features = [c for c in df.columns if c not in [target, "smiles"]]

    for datasets in dataset_combos:
        dataset_names = [data_csvs[dataset] for dataset in datasets]

        dataset_paths = {data_name: [os.path.join(root_data_dir, feature_dict[c], data_csvs[data_name]) for c in [primary_data, secondary_data]] for data_name in datasets}

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
                        "combo": "_".join([primary_data, secondary_data]),
                        "datasets": "_".join(datasets),
                    }
                )
            continue

        if len(datasets) > 1:
            df_rest = pd.concat([df for data_name, df in dfs.items() if data_name != "expert"], ignore_index=True).sort_values(by="smiles").reset_index(drop=True)
        else:
            df_rest = None  # pylint: disable=invalid-name
        df_expert1 = dfs["expert"]

        folds_experts1 = custom_data_kfold(df_expert1, df_expert1.loc[:, [target]], num_splits, num_bins, random_state)
        if df_rest is not None:
            folds_all, df_all = combine_split(df_expert1, folds_experts1, df_rest)
        else:
            folds_all = folds_experts1
            df_all = df_expert1

        if "custom" in [primary_data, secondary_data] and "symmetry" in df_all.columns:
            cat_features = cat_features_dict["custom"]
        else:
            cat_features = []
        num_features = [c for c in df_all.columns if c not in ["smiles", target, *cat_features, *binary_features]]

        DATASET_NAME = "_".join(sorted(datasets))
        COMBO_NAME = "_".join(sorted([primary_data, secondary_data]))
        DATANAME = f"{DATASET_NAME}_{COMBO_NAME}"

        date = datetime.today().strftime("%d-%m-%Y")
        SAVE_DIR = f"../../../results/{DATANAME}/{date}/"
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        X_all = df_all.drop(columns=[target, "smiles"])
        y_all = df_all.loc[:, [target]]

        cv = CrossValidationPipeline(X_all, y_all, num_features, cat_features, folds_all, metrics, SAVE_DIR, DATANAME, (True, "grid_search"), (True, fixed_features), verbose=True)
        results = cv.batch_train_and_eval(models)

        for key, value in results.items():
            results_model = value
            results_model["model"] = key
            results_model["features"] = COMBO_NAME
            results_model["datasets"] = DATASET_NAME
            full_results.append(results_model)

    full_results = pd.DataFrame(full_results)
    full_results.to_csv(save_path, index=False)
