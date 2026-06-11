"""Script for processing SMILES that earlier resulted in errors"""

import os

import pandas as pd
import yaml

from modules.predictor.data.dft_features import smiles_to_features_pyscf

if __name__ == "__main__":
    CONFIG_PATH = "../configs/dft_fix.yaml"
    with open(CONFIG_PATH, encoding="UTF-8") as file:
        config = yaml.safe_load(file)

    must_have_features = config["must_have_features"]
    files_to_fix = config["files_to_fix"]

    for file_to_fix in files_to_fix:
        data_path = os.path.join(config["root_dir"], file_to_fix)
        df = pd.read_csv(data_path)
        num_cols = len(df.columns)
        nan_rows = df[must_have_features].isna().any(axis=1)
        nan_rows_ids = df.index[nan_rows].tolist()
        for i in nan_rows_ids:
            smiles = df.loc[i, ["smiles"]].tolist()[0]
            capacity_max = df.loc[i, ["capacity_max"]].tolist()[0]
            dft_features = smiles_to_features_pyscf(
                smiles,
                capacity_max,
                "capacity_max",
                functional=config["functional"],
                dft_type=config["dft_type"],
                spin=config["spin"],
            )
            print(file_to_fix)
            print(dft_features)
            df.loc[i] = dft_features
        df.to_csv(data_path, index=False)
