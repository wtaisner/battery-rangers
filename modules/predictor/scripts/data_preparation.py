"""Data preparation script."""
import os

from modules.predictor.data.custom_features import data_preprocessing_and_feature_engineering

if __name__ == "__main__":
    DATA_PATH = "../../../data/raw/"
    SAVE_PATH = "../../../data/processed_selected_custom_features/"

    TRANSLATION_TABLE_PATH = "../../../data/symmetries/symmetry_translation.csv"

    DF_EXPERTS1_PATH = "data_experts1.csv"
    DF_EXPERTS2_PATH = "data_experts2.csv"
    DF_SAAD_PATH = "data_saad.csv"
    DF_ZHU_PATH = "data_zhu.csv"

    data_paths = [DF_EXPERTS1_PATH, DF_ZHU_PATH, DF_SAAD_PATH, DF_EXPERTS2_PATH]
    data_types = ["expert", "zhu", "saad", "expert2"]

    for df_path, data_type in zip(data_paths, data_types):
        print(data_type)
        data_preprocessing_and_feature_engineering(os.path.join(DATA_PATH, df_path), data_type, TRANSLATION_TABLE_PATH, os.path.join(SAVE_PATH, df_path), remove_unuseful=True)
