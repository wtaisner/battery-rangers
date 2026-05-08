import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyscf import dft, gto
from pyscf.hessian import thermo
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_3d(smiles: str) -> tuple | None:
    """
    Converts a SMILES string to a 3D optimized molecular structure.
    :param smiles: a SMILES string.
    :return: tuple with molecular structure graph, list of atom symbols, and array of 3d coordinates of atoms.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)  # Add hydrogens
    a = AllChem.EmbedMolecule(
        mol, randomSeed=42, maxAttempts=500
    )  # Generate initial 3D structure
    if a < 0:
        a = AllChem.EmbedMolecule(
            mol, randomSeed=42, maxAttempts=500, useRandomCoords=True
        )
        if a < 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol)
    coords = mol.GetConformer().GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return mol, symbols, coords


def run_pyscf_dft(symbols: list, coords: np.ndarray, functional: str = "B3LYP") -> dict:
    """
    Perform DFT calculations using PySCF and extract quantum features.
    :param symbols: list of symbols.
    :param coords: list of atom coordinates.
    :param functional: Exchange-correlation functional (default: "B3LYP").
    :return:  Dictionary of quantum features
    """
    mol = gto.M(atom=[[symbols[i], tuple(coords[i])] for i in range(len(symbols))])
    mol.build()
    try:
        mf = dft.RKS(mol)
        mf.xc = functional
        mf.init_guess = "atom"
        mf.run()
        energy = mf.kernel()  # Total energy
    except Exception:
        energy = None

    #
    try:
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
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
        atomic_charges_names = [
            f"charge_{ia}{mol.atom_symbol(ia)}" for ia in range(mol.natm)
        ]
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
        "dipole_y": dipole_y,
        "dipole_z": dipole_z,
        "vibrational_frequencies_mean": freq_mean,  # Average frequency
        "vibrational_frequencies_min": freq_min,
        "vibrational_frequencies_max": freq_max,
        "internal_energy_0K": internal_energy_0,
        "internal_energy_298K": internal_energy_298,
        **rot_const_dict,
        "zpe": zpe,
        "free_energy": free_energy,
        "enthalpy_energy": enthalpy_energy,
    }


def smiles_to_features_pyscf(
    smiles: str, target: float, target_col: str, functional: str = "B3LYP"
) -> dict:
    """
    generates quantum features using PySCF for a given SMILES.
    :param smiles: SMILES string.
    :param target: value of the target for the given SMILES.
    :param target_col: name of the target column.
    :param functional: Exchange-correlation functional (default: "B3LYP").
    :return: dict with generated features.
    """
    print(f"Starting: {smiles}")
    try:
        mol, symbols, coords = smiles_to_3d(smiles)

        # Step 3: Run DFT calculation using PySCF
        dft_features = run_pyscf_dft(symbols, coords, functional)

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
    df: pd.DataFrame, smiles_col: str, target_col: str, functional: str = "B3LYP"
) -> pd.DataFrame:
    """
    Takes a dataframe with smiles strings and generates quantum features using PySCF.
    :param df: dataframe with smiles and target columns.
    :param smiles_col: name of the column with smiles.
    :param target_col: name of the target column.
    :param functional: Exchange-correlation functional (default: "B3LYP").
    :return: dataframe with generated quantum features.
    """
    smiles_list = df[smiles_col].tolist()
    target_list = df[target_col].tolist()

    feature_list = Parallel(n_jobs=8)(
        delayed(smiles_to_features_pyscf)(smiles, target, target_col, functional)
        for smiles, target in zip(smiles_list, target_list)
    )

    df_features = pd.DataFrame(feature_list)
    return df_features


if __name__ == "__main__":
    df = pd.read_csv("data_substrates_june2025.csv")
    df = df_to_features_pyscf(df, "smiles", "capacity_max", functional="pbe")
    df.to_csv("dft_data_substrates_june2025.csv", index=False)
