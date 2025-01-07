"""Script for generating comparison matrix."""
import os

import pandas as pd

from modules.predictor.data.utils import combine_split, custom_data_kfold
from modules.predictor.training_and_evaluation.train_eval_pipeline import dataset_preprocess_and_train

if __name__ == "__main__":
    SAVE_MATRIX_PATH = "../../../results/comparison_matrix_test_expert1.csv"

    data_csvs = {"expert1": "data_experts1.csv", "expert2": "data_experts2.csv", "saad": "data_saad.csv", "zhu": "data_zhu.csv"}
    data_paths = {
        "custom": "../../../data/processed_selected_custom_features",
        "dft": "../../../data/aggregated_dft_features",
        "ecfp": "../../../data/fingerprints_ecfp_features",
        "maccs": "../../../data/fingerprints_maccs_features",
    }
    COMBO_NAME = "all_data_all_custom_dft_ecfp_maccs"
    dataset_combos = [["expert1"], ["expert2", "expert1"], ["zhu", "expert1", "saad"], ["expert2", "zhu", "expert1", "saad"]]
    combo_features = [["custom", i] for i in ["custom", "dft", "ecfp", "maccs"]] + [["dft", i] for i in ["dft", "ecfp", "maccs"]] + [["ecfp", "ecfp"], ["maccs", "maccs"]]

    full_results = []

    cat_features_dict = {"custom": ["symmetry"]}

    binary_features = {}
    for features, data_path in data_paths.items():
        df = pd.read_csv(os.path.join(data_path, data_csvs["expert1"]))
        binary_features[features] = [c for c in df.columns if (df[c].nunique() <= 2 and set(df[c].dropna().unique()).issubset({0, 1}))]

    TARGET = "capacity_max"
    NUM_SPLITS = 4
    NUM_BINS = 4
    RANDOM_STATE = 42

    for datasets in dataset_combos:
        for combo in combo_features:
            combo = list(set(combo))

            dataset_names = [data_csvs[dataset] for dataset in datasets]
            dataset_paths = {data_name: [os.path.join(data_paths[c], data_csvs[data_name]) for c in combo] for data_name in datasets}
            dfs = {data_name: [pd.read_csv(data_path).sort_values(by="smiles").reset_index(drop=True) for data_path in paths] for data_name, paths in dataset_paths.items()}

            dfs = {data_name: pd.concat(dataframes, axis=1).reset_index(drop=True) for data_name, dataframes in dfs.items()}
            dfs = {data_name: df.loc[:, ~df.columns.duplicated()].sort_values(by="smiles").reset_index(drop=True) for data_name, df in dfs.items()}
            if len(datasets) > 1:
                df_rest = pd.concat([df for data_name, df in dfs.items() if data_name != "expert1"], ignore_index=True).sort_values(by="smiles").reset_index(drop=True)
            else:
                df_rest = None  # pylint: disable=invalid-name
            df_expert1 = dfs["expert1"]

            folds_experts1 = custom_data_kfold(df_expert1, TARGET, NUM_SPLITS, NUM_BINS, RANDOM_STATE)
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
            num_features = [c for c in df_all.columns if c not in ["smiles", TARGET, *cat_features, *bin_features]]

            DATASET_NAME = "_".join(datasets)
            COMBO_NAME = "_".join(combo)
            DATANAME = f"{DATASET_NAME}_{COMBO_NAME}"
            results = dataset_preprocess_and_train(df_all, DATANAME, cat_features, num_features, folds_all, TARGET)
            results["features"] = COMBO_NAME
            results["datasets"] = DATASET_NAME
            full_results.append(results)
    pd.DataFrame(full_results).to_csv(SAVE_MATRIX_PATH, index=False)
