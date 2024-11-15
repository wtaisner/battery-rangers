from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def calculate_repeating_unit_length(smiles: str):
    """
    Calculates the length of the repeating unit for a given molecule as the largest distance between any two atoms.

    Parameters:
    - smiles: SMILES string of the molecule

    Returns:
    - The length of the repeating unit in Angstroms.
    """

    try:
        molecule = Chem.MolFromSmiles(smiles)
    except Exception:
        raise ValueError("Invalid SMILES string.")

    AllChem.EmbedMolecule(molecule)
    AllChem.MMFFOptimizeMolecule(molecule)

    conf = molecule.GetConformer()
    max_distance = 0

    for i in range(molecule.GetNumAtoms()):
        for j in range(i + 1, molecule.GetNumAtoms()):
            pos_i = np.array(conf.GetAtomPosition(i))
            pos_j = np.array(conf.GetAtomPosition(j))

            # Calculate the distance between the two points
            distance = np.linalg.norm(pos_i - pos_j)

            if distance > max_distance:
                max_distance = distance

    return max_distance


def calculate_hexagonal_pore_diameter(smiles: str):
    """
    Calculates the diameter of a hexagonal pore given the length of the unit.

    Parameters:
    - smiles: SMILES string of the molecule

    Returns:
    - pore_diameter_nm: The diameter of the pore in nanometers.
    """

    unit_length_pm = calculate_repeating_unit_length(smiles)
    perimeter = unit_length_pm * 6

    pore_diameter = perimeter / np.pi

    pore_diameter_nm = pore_diameter / 10  # Convert to nanometers

    return pore_diameter_nm


def sanity_check_with_experts():
    smiles_Terephtalonitrile = "N#CC1=CC=C(C#N)C=C1"
    smiles_2CNPP = "CC(C)C(C=C1)=CC=C1N2C(C=C(C3=CC=C(C#N)C=C3)N4C5=CC=C(C(C)C)C=C5)=C4C=C2C6=CC=C(C#N)C=C6"
    pore_diameter_Terephtalonitrile = calculate_hexagonal_pore_diameter(
        smiles_Terephtalonitrile
    )
    pore_diameter_2CNPP = calculate_hexagonal_pore_diameter(smiles_2CNPP)
    print(f"Pore diameter for Terephtalonitrile {pore_diameter_Terephtalonitrile}")
    print(f"Pore diameter for 2CNPP {pore_diameter_2CNPP}")


if __name__ == "__main__":
    sanity_check_with_experts()
