"""Script for aggregating DFT features."""

import os

import pandas as pd
import yaml

from modules.predictor.data.dft_features import aggregate_and_drop_dft  # pylint: disable=import-error

if __name__ == "__main__":
    CONFIG_PATH = "../configs/dft_aggregation.yaml"
    with open(CONFIG_PATH, encoding="UTF-8") as file:
        config = yaml.safe_load(file)

    data_path = os.path.join(config["root_dir"], config["data_dir"])
    save_path = os.path.join(config["root_dir"], config["save_dir"])

    os.makedirs(save_path, exist_ok=True)

    data_dict = config["data_dict"]

    for data_type, data in data_dict.items():
        print(data_type)
        df = pd.read_csv(os.path.join(data_path, data))
        df_aggregated = aggregate_and_drop_dft(df)
        df.drop(columns=config["drop_columns"], inplace=True)
        df.to_csv(os.path.join(save_path, data), index=False)
