"""Script for aggregating DFT features."""
import os

import pandas as pd

from modules.predictor.data.dft_features import aggregate_and_drop_dft

if __name__ == "__main__":
    DATA_PATH = "../../../data/processed_dft_features/"
    SAVE_PATH = "../../../data/aggregated_dft_features/"

    DF_EXPERTS1_PATH = "data_experts1.csv"
    DF_EXPERTS2_PATH = "data_experts2.csv"
    DF_SAAD_PATH = "data_saad.csv"
    DF_ZHU_PATH = "data_zhu.csv"

    data_paths = [DF_EXPERTS1_PATH, DF_ZHU_PATH, DF_SAAD_PATH, DF_EXPERTS2_PATH]
    data_types = ["expert", "zhu", "saad", "expert2"]

    for df_path, data_type in zip(data_paths, data_types):
        print(data_type)
        df = pd.read_csv(os.path.join(DATA_PATH, df_path))
        df_aggregated = aggregate_and_drop_dft(df)
        df.to_csv(os.path.join(SAVE_PATH, df_path), index=False)
