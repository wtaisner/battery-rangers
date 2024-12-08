"""Utility functions for the features."""
import os

import networkx as nx
import pymatgen.core
from pymatgen.core.structure import Molecule, Structure
from pymatgen.vis.structure_vtk import StructureVis
from rdkit import Chem
from rdkit.Chem import AllChem, MolToXYZFile, rdDepictor, rdDistGeom

from modules.core.features.preprocessing import canon_smiles


def smiles_to_xyz(smiles: str, save_file: bool = False, directory: str = "./tmp") -> str | None:
    """
    Convert a SMILES string to a 3D xyz file using RDKit.

    Args:
        smiles: A SMILES string.
        save_file: Whether to save the xyz file.
        directory: The directory to save the xyz file.
    Returns:
        The path to the saved xyz file.
    """
    canonic_smiles = canon_smiles(smiles)
    rdkit_mol = Chem.MolFromSmiles(canonic_smiles, sanitize=True)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    rdDepictor.Compute2DCoords(rdkit_mol, sampleSeed=42)
    a = rdDistGeom.EmbedMolecule(rdkit_mol, randomSeed=42, maxAttempts=500)
    if a < 0:
        a = rdDistGeom.EmbedMolecule(rdkit_mol, randomSeed=42, maxAttempts=500, useRandomCoords=True)
        if a < 0:
            return None
        a = 3
    if a == 3:
        rdDistGeom.EmbedMultipleConfs(rdkit_mol, 10, randomSeed=123, useRandomCoords=True)
    else:
        rdDistGeom.EmbedMultipleConfs(rdkit_mol, 10, randomSeed=123)

    save_dir = f"{directory}/rdkit_mol.xyz"

    if save_file:
        if not os.path.exists(directory):
            os.makedirs(directory)
        MolToXYZFile(rdkit_mol, save_dir)

    return save_dir


def get_pymatgen_molecule_from_smiles(smiles: str, save_file: bool = False, directory: str = "./tmp") -> pymatgen.core.Molecule | None:
    """
    Convert a SMILES string to a pymatgen Molecule object.

    Args:
        smiles: A SMILES string.
        save_file: Whether to save the xyz file.
        directory: The directory to save the xyz file.
    Returns:
        The pymatgen Molecule object or None if the conversion failed.

    """
    path = smiles_to_xyz(smiles, save_file, directory)
    if path is None:
        return None
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
    a = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=500)  # Generate initial 3D structure
    if a < 0:
        a = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=500, useRandomCoords=True)
        if a < 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol)
    coords = mol.GetConformer().GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return mol, symbols, coords


def get_graph_from_smile(smile: str) -> nx.Graph:
    """
    Get a graph from a SMILE string.
    """
    mol = Chem.MolFromSmiles(smile)

    if mol is None:
        print(f"Mol: {str(mol)} of {smile} is NONE")

    adjacency_matrix = Chem.GetAdjacencyMatrix(mol, useBO=True)
    return nx.from_numpy_array(adjacency_matrix)
