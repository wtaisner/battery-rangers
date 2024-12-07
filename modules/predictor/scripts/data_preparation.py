"""Data preparation script."""
import os

from modules.predictor.data.custom_features import data_preprocessing_and_feature_engineering
from modules.predictor.data.dft_features import df_to_features_pyscf
from modules.predictor.data.fingerprints import fingerprints_dataset
from modules.predictor.data.utils import data_preprocessing

if __name__ == "__main__":
    DATA_PATH = "../../../data/raw/"
    SAVE_PATH = "../../../data/processed_selected_custom_features/"
    SAVE_PATH_DFT = "../../../data/processed_dft_features/"

    TRANSLATION_TABLE_PATH = "../../../data/symmetries/symmetry_translation.csv"

    DF_EXPERTS1_PATH = "data_experts1.csv"
    DF_EXPERTS2_PATH = "data_experts2.csv"
    DF_SAAD_PATH = "data_saad.csv"
    DF_ZHU_PATH = "data_zhu.csv"

    fingerprint_dict = {
        "ecfp": "../../../data/fingerprints_ecfp_features",
        "maccs": "../../../data/fingerprints_maccs_features",
        "rdkit": "../../../data/fingerprints_rdkit_features",
    }

    GROUP_ADDITIVITY_PATH = "../../../data/group_additivity"
    additivity_dict = {
        "BensonGA": "../../../data/group_additivity_BensonGA_features",
        "PtSurface2023": "../../../data/group_additivity_PtSurface2023_features",
        "GuSolventGA2017Aq": "../../../data/group_additivity_GuSolventGA2017Aq_features",
        "XieGA2022": "../../../data/group_additivity_XieGA2022_features",
        "GRWSurface2018": "../../../data/group_additivity_GRWSurface2018_features",
        "GRWAqueous2018": "../../../data/group_additivity_GRWAqueous2018_features",
    }

    data_paths = [DF_EXPERTS1_PATH, DF_ZHU_PATH, DF_SAAD_PATH, DF_EXPERTS2_PATH]
    data_types = ["expert", "zhu", "saad", "expert2"]

    for df_path, data_type in zip(data_paths, data_types):
        print(data_type)
        data_preprocessing_and_feature_engineering(os.path.join(DATA_PATH, df_path), data_type, TRANSLATION_TABLE_PATH, os.path.join(SAVE_PATH, df_path), remove_unuseful=True)

    for df_path, data_type in zip(data_paths, data_types):
        print(f"{data_type}, dft")
        df = data_preprocessing(os.path.join(DATA_PATH, df_path), data_type)
        df = df_to_features_pyscf(df, "smiles", "capacity_max", functional="pbe")
        df.to_csv(os.path.join(SAVE_PATH_DFT, df_path), index=False)

    for df_path, data_type in zip(data_paths, data_types):
        for fingerprint, fingerprint_path in fingerprint_dict.items():
            print(f"{data_type}, {fingerprint}")
            if not os.path.exists(fingerprint_path):
                os.makedirs(fingerprint_path)
            df = data_preprocessing(os.path.join(DATA_PATH, df_path), data_type)
            df = fingerprints_dataset(df, "smiles", "capacity_max", fingerprint, save_path=fingerprint_path, kwargs={"radius": 6, "size": 128})
            df.to_csv(os.path.join(fingerprint_path, df_path), index=False)
