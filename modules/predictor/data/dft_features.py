"""Functions for calculating dft features."""

import re
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyscf import dft, gto
from pyscf.hessian import thermo

from modules.core.features.utils import smiles_to_3d

# pylint: disable=broad-exception-caught


def run_pyscf_dft(
    symbols: list,
    coords: np.ndarray,
    functional: str = "B3LYP",
    dft_type: Literal["RKS", "UKS"] = "RKS",
    spin: int | None = 0,
) -> dict:  # pylint: disable=too-many-statements
    """Perform DFT calculations using PySCF and extract quantum features.
    :param symbols: list of symbols.
    :param coords: list of atom coordinates.
    :param functional: Exchange-correlation functional (default: "B3LYP").
    :param dft_type: whether to perform restricted or unrestricted dft (RKS or UKS).
    :param spin: for building the mole (default 0).
    :return:  Dictionary of quantum features
    """
    mol = gto.M(atom=[[symbols[i], tuple(coords[i])] for i in range(len(symbols))], spin=spin)
    mol.build(spin=spin)
    try:
        if dft_type == "RKS":
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = functional
        mf.init_guess = "atom"
        mf.run()
        energy = mf.kernel()  # Total energy
    except Exception:
        energy = None

    try:
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        if dft_type == "UKS":
            mo_energy = mo_energy[0]
            mo_occ = mo_occ[0]
        mo_energy_names = [f"mo_energy_{i}" for i in range(len(mo_energy))]
        mo_occ_names = [f"mo_occ_{i}" for i in range(len(mo_occ))]
        mo_energy_dict = dict(zip(mo_energy_names, mo_energy))
        mo_occ_dict = dict(zip(mo_occ_names, mo_occ))
        homo = mo_energy[mo_occ > 0].max()
        lumo = mo_energy[mo_occ == 0].max()
    except Exception:
        mo_energy_dict = {}
        mo_occ_dict = {}
        homo = None
        lumo = None

    # scf summary
    try:
        scf_summary = mf.scf_summary
    except Exception:
        scf_summary = {}

    # atomic charges
    try:
        analysis = mf.analyze(verbose=0)[0]
        atomic_charges = analysis[1]
        atomic_charges_names = [f"charge_{ia}{mol.atom_symbol(ia)}" for ia in range(mol.natm)]
        atomic_charges_dict = dict(zip(atomic_charges_names, atomic_charges))
    except Exception:
        atomic_charges_dict = {}

    # Dipole moment
    try:
        dipole = mf.dip_moment()
        dipole_x, dipole_y, dipole_z = dipole[0], dipole[1], dipole[2]
    except Exception:
        dipole_x, dipole_y, dipole_z = None, None, None

    # Vibrational frequencies
    try:
        hessian = mf.Hessian().kernel()
        freq_info = thermo.harmonic_analysis(mf.mol, hessian)
        freq = freq_info["freq_wavenumber"]
        freq_min = np.min(freq).item()
        freq_mean = np.mean(freq).item()
        freq_max = np.max(freq).item()
        thermo_info = thermo.thermo(mf, freq_info["freq_au"], 298.15, 101325)
        rot_const = thermo_info["rot_const"][0]
        rot_const_names = [f"rot_const_{i}" for i in range(len(rot_const))]
        rot_const_dict = dict(zip(rot_const_names, rot_const))
        zpe = thermo_info["ZPE"][0]
        internal_energy_298 = thermo_info["E_tot"][0]
        internal_energy_0 = thermo_info["E_0K"][0]
        free_energy = thermo_info["G_tot"][0]
        enthalpy_energy = thermo_info["H_tot"][0]
    except Exception:
        freq_min = None
        freq_mean = None
        freq_max = None
        rot_const_dict = {}
        zpe = None
        internal_energy_298 = None
        internal_energy_0 = None
        free_energy = None
        enthalpy_energy = None

    return {
        "total_energy": energy,
        **mo_energy_dict,
        **mo_occ_dict,
        **scf_summary,
        **atomic_charges_dict,
        "homo": homo,
        "lumo": lumo,
        "dipole_x": dipole_x,
        "diploe_y": dipole_y,
        "dipole_z": dipole_z,
        "vibrational_frequencies_mean": freq_mean,  # Average frequency
        "vibrational_frequencies_min": freq_min,
        "vibrational_frequencies_max": freq_max,
        "internal_energy_0K": internal_energy_0,
        "internal_energy_298K": internal_energy_298,
        **rot_const_dict,
        "zpe": zpe,
        "free_enegry": free_energy,
        "enthalpy_energy": enthalpy_energy,
    }


def smiles_to_features_pyscf(
    smiles: str,
    target: float,
    target_col: str,
    functional: str = "B3LYP",
    dft_type: Literal["RKS", "UKS"] = "RKS",
    spin: int | None = 0,
) -> dict:
    """Generates quantum features using PySCF for a given SMILES.
    :param smiles: SMILES string.
    :param target: value of the target for the given SMILES.
    :param target_col: name of the target column.
    :param functional: Exchange-correlation functional (default: "B3LYP").
    :param dft_type: whether to perform restricted or unrestricted dft (RKS or UKS).
    :param spin: for building the mole (default 0).
    :return: dict with generated features.
    """
    print(f"Starting: {smiles}")
    try:
        _, symbols, coords = smiles_to_3d(smiles)

        # Step 3: Run DFT calculation using PySCF
        dft_features = run_pyscf_dft(symbols, coords, functional, dft_type, spin=spin)

        # Combine all features
        features = {
            "smiles": smiles,
            **dft_features,
            target_col: target,
        }
        print(f"Processed: {smiles}")
        return features
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return {
            "smiles": smiles,
            target_col: target,
        }


def df_to_features_pyscf(
    df: pd.DataFrame,
    smiles_col: str,
    target_col: str,
    functional: str = "B3LYP",
    n_jobs: int = 8,
) -> pd.DataFrame:
    """Takes a dataframe with smiles strings and generates quantum features using PySCF.
    :param df: dataframe with smiles and target columns.
    :param smiles_col: name of the column with smiles.
    :param target_col: name of the target column.
    :param functional: Exchange-correlation functional (default: "B3LYP").
    :param n_jobs: number of parallel jobs.
    :return: dataframe with generated quantum features.
    """
    smiles_list = df[smiles_col].tolist()
    target_list = df[target_col].tolist()

    feature_list = Parallel(n_jobs=n_jobs)(delayed(smiles_to_features_pyscf)(smiles, target, target_col, functional) for smiles, target in zip(smiles_list, target_list))

    df_features = pd.DataFrame(feature_list)
    return df_features


def aggregate_and_drop_dft(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate DFT features.
    :param df: dataframe with DFT features.
    :return: dataframe with aggregated DFT features.
    """
    col_names = df.columns.tolist()
    df.drop(
        columns=[
            "vibrational_frequencies_min",
            "vibrational_frequencies_max",
            "vibrational_frequencies_mean",
            "internal_energy_0K",
            "internal_energy_298K",
        ],
        inplace=True,
    )

    stats = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
    }

    # mo_energy by occ
    mo_occ = [name for name in col_names if name.startswith("mo_occ")]
    mo_energy = [name for name in col_names if name.startswith("mo_energy")]
    mo_energy.sort(key=lambda x: int(x.split("_")[-1]))
    mo_occ.sort(key=lambda x: int(x.split("_")[-1]))
    unique_occ_values = pd.concat([df[col] for col in mo_occ]).dropna().unique()
    # print(unique_occ_values)

    for occ_value in unique_occ_values:
        for stat in stats:
            df[f"mo_energy_{stat}_occ_{occ_value}"] = 0.0
    for index, row in df.iterrows():
        for occ_value in unique_occ_values:
            filtered_mo_energy = [row[mo_energy_col] for mo_energy_col, mo_occ_col in zip(mo_energy, mo_occ) if row[mo_occ_col] == occ_value]
            if filtered_mo_energy:
                for stat, func in stats.items():
                    df.at[index, f"mo_energy_{stat}_occ_{occ_value}"] = func(filtered_mo_energy)
    df.drop(columns=mo_energy + mo_occ, inplace=True)

    # charge by atom
    charge = [name for name in col_names if name.startswith("charge")]
    atoms = np.unique([re.split(r"(\d+)", name)[-1] for name in charge])
    # print(atoms)
    for atom in atoms:
        for stat in stats:
            df[f"charge_{stat}_{atom}"] = 0.0
    for index, row in df.iterrows():
        for atom in atoms:
            filtered_charge = np.array([row[charge_col] for charge_col in charge if re.split(r"(\d+)", charge_col)[-1] == atom])
            filtered_charge = list(filtered_charge[~np.isnan(filtered_charge)])
            if filtered_charge:
                for stat, func in stats.items():
                    df.at[index, f"charge_{stat}_{atom}"] = func(filtered_charge)
    df.drop(columns=charge, inplace=True)

    # rotation constants
    col = "rot_const"
    to_aggregate = [name for name in col_names if name.startswith("rot_const")]
    df[f"{col}_mean"] = df[to_aggregate].mean(axis=1, skipna=True)
    df.drop(columns=to_aggregate, inplace=True)
    return df
