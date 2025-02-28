"""Utility functions for the features."""
import os

import networkx as nx
import pymatgen.core
from pymatgen.core.structure import Molecule, Structure
from pymatgen.vis.structure_vtk import StructureVis
from rdkit import Chem
from rdkit.Chem import AllChem, Mol, MolToXYZFile, rdDepictor

from modules.core.features.preprocessing import canon_smiles


def mol_to_xyz(molecule: str | Mol, save_file: bool = False, directory: str = "./tmp") -> str | None:
    """
    Convert a SMILES string or RDKit molecule to a 3D xyz file using RDKit.

    Args:
        molecule: A SMILES string or RDKit molecule.
        save_file: Whether to save the xyz file.
        directory: The directory to save the xyz file.
    Returns:
        The path to the saved xyz file.
    """
    if isinstance(molecule, str):
        canonic_smiles = canon_smiles(molecule)
        rdkit_mol = Chem.MolFromSmiles(canonic_smiles, sanitize=True)
    else:
        rdkit_mol = molecule
    rdkit_mol = Chem.AddHs(rdkit_mol)
    rdDepictor.Compute2DCoords(rdkit_mol, sampleSeed=42)
    # a = rdDistGeom.EmbedMolecule(rdkit_mol, randomSeed=42, maxAttempts=500)
    # if a < 0:
    #     a = rdDistGeom.EmbedMolecule(rdkit_mol, randomSeed=42, maxAttempts=500, useRandomCoords=True)
    #     if a < 0:
    #         return None
    #     a = 3
    # if a == 3:
    #     rdDistGeom.EmbedMultipleConfs(rdkit_mol, 10, randomSeed=123, useRandomCoords=True)
    # else:
    #     rdDistGeom.EmbedMultipleConfs(rdkit_mol, 10, randomSeed=123)

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
    path = mol_to_xyz(smiles, save_file, directory)
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


def get_graph_from_molecule(molecule: str | Mol) -> nx.Graph:
    """
    Get a graph from a SMILE string.
    """
    if isinstance(molecule, str):
        molecule = Chem.MolFromSmiles(molecule)

    if molecule is None:
        print(f"Mol: {str(molecule)} of {molecule} is NONE")

    adjacency_matrix = Chem.GetAdjacencyMatrix(molecule, useBO=True)
    return nx.from_numpy_array(adjacency_matrix)


def get_aromatic_rings(mol: Chem.Mol) -> list[tuple[int]]:
    """
    Get all aromatic rings in a molecule.

    Args:
        mol: A RDKit molecule.
    Returns:
        A list of tuples with the indices of the atoms in the aromatic rings.

    Example:
            N#Cc1c(Cl)c(C#N)c(Cl)c(C#N)c1Cl
            [(14, 12, 9, 7, 4, 2)]

            N#Cc1c2c(c(C#N)c3ccccc13)CCCC2
            [(17, 19, 7, 4, 3, 2), (9, 10, 11, 12, 19, 8)]

            N#CC(=N)C#N
            []
    """
    ri = mol.GetRingInfo()

    aromatic_rings = []
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(atom).GetIsAromatic() for atom in ring):
            aromatic_rings.append(tuple(ring))
    return aromatic_rings
