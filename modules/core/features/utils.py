"""Utility functions for the features."""
import logging
import os

import networkx as nx
import pymatgen.core
from pymatgen.core.structure import Molecule, Structure
from pymatgen.vis.structure_vtk import StructureVis
from rdkit import Chem
from rdkit.Chem import MolToXYZFile, rdDepictor, rdDistGeom


def smiles_to_xyz(smiles: str, directory: str = "./tmp") -> str:
    """
    Convert a SMILES string to a 3D xyz file using RDKit.

    Args:
        smiles: A SMILES string.
        directory: The directory to save the xyz file.
    Returns:
        The path to the saved xyz file.
    """
    canonic_smiles = Chem.CanonSmiles(smiles)
    rdkit_mol = Chem.MolFromSmiles(canonic_smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    rdDepictor.Compute2DCoords(rdkit_mol)
    rdDistGeom.EmbedMolecule(rdkit_mol)
    rdDistGeom.EmbedMultipleConfs(rdkit_mol, 10, randomSeed=123)

    logging.debug(f"Number of conformers: {rdkit_mol.GetNumConformers()}")
    logging.debug(f"Is 3D: {rdkit_mol.GetConformer().Is3D()}")
    logging.debug(f"Number of atoms: {rdkit_mol.GetNumAtoms()}")

    if not os.path.exists(directory):
        os.makedirs(directory)

    save_dir = f"{directory}/rdkit_mol.xyz"

    MolToXYZFile(rdkit_mol, save_dir)

    return save_dir


def get_pymatgen_molecule_from_smiles(smiles: str, directory: str = "./tmp") -> pymatgen.core.Molecule:
    """
    Convert a SMILES string to a pymatgen Molecule object.

    Args:
        smiles: A SMILES string.
        directory: The directory to save the xyz file.
    Returns:

    """
    path = smiles_to_xyz(smiles, directory)
    mol = Molecule.from_file(path)

    return mol


def visualize_structure(structure: Structure, **kwargs) -> None:
    """
    Visualizes the given structure using StructureVis.

    Args:
        structure (Structure): The structure to be visualized.
    Returns:
        None
    """
    stvis = StructureVis(**kwargs)
    stvis.set_structure(structure)
    stvis.show()


def get_graph_from_smile(smile: str) -> nx.Graph:
    """
    Get a graph from a SMILE string.
    """
    mol = Chem.MolFromSmiles(smile)

    if mol is None:
        print(f"Mol: {str(mol)} of {smile} is NONE")

    adjacency_matrix = Chem.GetAdjacencyMatrix(mol, useBO=True)
    return nx.from_numpy_array(adjacency_matrix)
