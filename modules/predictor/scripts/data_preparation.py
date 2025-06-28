"""Data preparation script. Generating custom, dft, fingerprints and additive groups features."""
import os

import pandas as pd
import yaml

from modules.predictor.data.custom_features import data_preprocessing_and_feature_engineering
from modules.predictor.data.custom_smarts_features import expert_features_dataset
from modules.predictor.data.dft_features import df_to_features_pyscf
from modules.predictor.data.fingerprints import fingerprints_dataset
from modules.predictor.data.group_additivity import AdditiveGroups
from modules.predictor.data.utils import data_preprocessing

if __name__ == "__main__":
    CONFIG_PATH = "../configs/data_preparation.yaml"
    with open(CONFIG_PATH, encoding="UTF-8") as file:
        config = yaml.safe_load(file)

    root_dir, raw_data_dir, data_dict = config["root_dir"], config["raw_data_dir"], config["data_dict"]
    data_path = os.path.join(root_dir, raw_data_dir)

    if config["custom_features"]:
        config_custom_features = config["custom_features_params"]
        translation_table_path = os.path.join(root_dir, config_custom_features["translation_table"])
        save_path = os.path.join(root_dir, config_custom_features["dir"])
        os.makedirs(save_path, exist_ok=True)
        for data_type, data in data_dict.items():
            print(data_type)
            data_preprocessing_and_feature_engineering(os.path.join(data_path, data), data_type, translation_table_path, os.path.join(save_path, data), remove_unuseful=True)

    if config["dft_features"]:
        config_dft_features = config["dft_features_params"]
        save_path = os.path.join(root_dir, config_dft_features["dir"])
        os.makedirs(save_path, exist_ok=True)
        for data_type, data in data_dict.items():
            print(f"{data_type}, dft")
            df = data_preprocessing(os.path.join(data_path, data), data_type)
            df = df_to_features_pyscf(df, "smiles", "capacity_max", functional=config_dft_features["functional"], n_jobs=config_dft_features["n_jobs"])
            df = df.loc[:, df.nunique() > 1]
            df.to_csv(os.path.join(save_path, data), index=False)

    if config["fingerprints"]:
        config_fingerprints = config["fingerprints_types"]
        for fingerprint, use in config_fingerprints.items():
            if use:
                settings = config["fingerprints_params"][fingerprint]
                save_path = os.path.join(root_dir, settings["dir"])
                os.makedirs(save_path, exist_ok=True)
                kwargs = settings["kwargs"] if "kwargs" in settings else {}
                for data_type, data in data_dict.items():
                    print(f"{data_type}, {fingerprint}")
                    df = data_preprocessing(os.path.join(data_path, data), data_type)
                    df = fingerprints_dataset(df, "smiles", "capacity_max", fingerprint, kwargs=kwargs)
                    df = df.loc[:, df.nunique() > 1]
                    df.to_csv(os.path.join(save_path, data), index=False)

    if config["expert_substructures"]:
        config_expert_substructure = config["expert_substructures_params"]
        save_path = os.path.join(root_dir, config_expert_substructure["dir"])
        os.makedirs(save_path, exist_ok=True)
        for data_type, data in data_dict.items():
            print(f"{data_type}, expert substructures")
            df = data_preprocessing(os.path.join(data_path, data), data_type)
            df = expert_features_dataset(df, "smiles", "capacity_max", count=config_expert_substructure["count"])
            df = df.loc[:, df.nunique() > 1]
            df.to_csv(os.path.join(save_path, data), index=False)

    if config["additive_groups"]:
        config_additive_groups = config["additive_groups_params"]
        save_path = os.path.join(root_dir, config_additive_groups["dir"])
        os.makedirs(save_path, exist_ok=True)
        training_set = config_additive_groups["training_set"]
        training_df = []
        for data_type in training_set:
            data = data_dict[data_type]
            df = data_preprocessing(os.path.join(data_path, data), data_type)
            training_df.append(df)
        training_df = pd.concat(training_df)
        molecules = training_df["smiles"].tolist()

        ag = AdditiveGroups(molecules)

        for data_type, data in data_dict.items():
            print(f"{data_type}, additive groups")
            df = data_preprocessing(os.path.join(data_path, data), data_type)
            _, df_groups, _ = ag.generate_groups(df["smiles"].tolist())
            df_groups["capacity_max"] = df["capacity_max"]
            df_features = df.loc[:, df.nunique() > 1]
            df_groups.to_csv(os.path.join(save_path, data), index=False)
