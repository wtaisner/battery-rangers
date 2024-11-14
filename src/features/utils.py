import logging
import os

from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdDepictor

from pymatgen.core.structure import Molecule, Structure
from pymatgen.vis.structure_vtk import StructureVis


def smiles_to_xyz(smiles: str, dir: str = "./tmp") -> str:
    """
    Convert a SMILES string to a 3D xyz file using RDKit.
    """
    cannonic_smiles = Chem.CanonSmiles(smiles)
    rdkit_mol = Chem.MolFromSmiles(cannonic_smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    rdDepictor.Compute2DCoords(rdkit_mol)
    rdDistGeom.EmbedMolecule(rdkit_mol)
    rdDistGeom.EmbedMultipleConfs(rdkit_mol, 10, randomSeed=123)

    logging.debug(f"Number of conformers: {rdkit_mol.GetNumConformers()}")
    logging.debug(f"Is 3D: {rdkit_mol.GetConformer().Is3D()}")
    logging.debug(f"Number of atoms: {rdkit_mol.GetNumAtoms()}")

    if not os.path.exists(dir):
        os.makedirs(dir)

    save_dir = f"{dir}/rdkit_mol.xyz"

    Chem.rdmolfiles.MolToXYZFile(rdkit_mol, save_dir)

    return save_dir


def get_pymatgen_molecule_from_smiles(smiles: str, dir: str = "./tmp") -> None:
    """
    Convert a SMILES string to a pymatgen Molecule object.
    """
    path = smiles_to_xyz(smiles, dir)
    mol = Molecule.from_file(path)

    return mol


def visualize_structure(structure: Structure, **kwargs) -> None:
    """
    Visualizes the given structure using StructureVis.

    Parameters:
    structure (Structure): The structure to be visualized.

    Returns:
    None
    """
    stvis = StructureVis(**kwargs)
    stvis.set_structure(structure)
    stvis.show()
