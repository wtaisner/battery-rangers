"""Functions for calculating descriptors of a molecule"""

from typing import Dict

from rdkit import Chem
from rdkit.Chem import Mol, rdMolDescriptors


def extract_data_from_mol(mol: Mol) -> Dict[str, float | bool]:
    """Extract a bunch of descriptor data from a molecule object.
    :param: mol:
    More: https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html
    """
    return {
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "num_amide_bonds": rdMolDescriptors.CalcNumAmideBonds(mol),
        "num_heteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "num_h_acceptors": rdMolDescriptors.CalcNumLipinskiHBA(mol),
        "num_h_donors": rdMolDescriptors.CalcNumLipinskiHBD(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "mol_wt": rdMolDescriptors.CalcExactMolWt(mol),
        "num_conformers": mol.GetNumConformers(),
        "is_chiral": mol.GetNumAtoms() != mol.GetNumAtoms(onlyExplicit=True),
        "is_aromatic": mol.GetNumAtoms(onlyExplicit=True) != mol.GetNumAtoms(onlyExplicit=True),
        "is_branched": rdMolDescriptors.CalcNumSpiroAtoms(mol) > 0,
        "is_cyclic": rdMolDescriptors.CalcNumSpiroAtoms(mol) == 0,
        "is_heterocycle": rdMolDescriptors.CalcNumSpiroAtoms(mol) == 0,
        "is_homocycle": rdMolDescriptors.CalcNumSpiroAtoms(mol) == 0,
    }


# Example usage
if __name__ == "__main__":
    mol = Chem.MolFromSmiles("CCO")
    print(extract_data_from_mol(mol))
