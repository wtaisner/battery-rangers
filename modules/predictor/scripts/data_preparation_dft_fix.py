"""Script for processing SMILES that earlier resulted in errors"""
import pandas as pd

from modules.predictor.data.dft_features import smiles_to_features_pyscf

if __name__ == "__main__":
    MUST_HAVE_FEATURES = [
        "total_energy",
        "homo",
        "lumo",
        "dipole_x",
        "diploe_y",
        "dipole_z",
        "vibrational_frequencies_mean",
        "vibrational_frequencies_max",
        "vibrational_frequencies_min",
        "internal_energy_0K",
        "internal_energy_298K",
        "zpe",
        "free_enegry",
        "enthalpy_energy",
    ]
    FILE_TO_FIX = "../../../data/processed_dft_features/data_experts1.csv"
    df = pd.read_csv(FILE_TO_FIX)
    num_cols = len(df.columns)
    nan_rows = df[MUST_HAVE_FEATURES].isna().any(axis=1)
    nan_rows_ids = df.index[nan_rows].tolist()
    for i in nan_rows_ids:
        smiles = df.loc[i, ["smiles"]].tolist()[0]
        capacity_max = df.loc[i, ["capacity_max"]].tolist()[0]
        dft_features = smiles_to_features_pyscf(smiles, capacity_max, "capacity_max", functional="pbe", dft_type="UKS", spin=None)
        print(dft_features)
        df.loc[i] = dft_features
    df.to_csv("../../../data/processed_dft_features/data_experts1_fixed.csv", index=False)
