"""Utility functions for the features."""

import networkx as nx
import pymatgen.core
from pymatgen.core.structure import Molecule, Structure
from pymatgen.vis.structure_vtk import StructureVis
from rdkit import Chem
from rdkit.Chem import Mol, rdDistGeom


def compute_conformer(molecule: str | Mol, num_conformers: int = 10, max_attempts: int = 1) -> Mol | None:
    """
    Compute a 3D conformation for a molecule and return a new RDKit molecule with the conformation.

    Args:
        molecule: A SMILES string or RDKit molecule.
        num_conformers: The number of conformers to generate (default is 10).
        max_attempts: The maximum number of attempts to generate a conformation (default is 1).
    Returns:
        The RDKit molecule with a 3D conformation or None if it failed.
    """

    if isinstance(molecule, str):
        rdkit_mol = Chem.MolFromSmiles(molecule)
    else:
        rdkit_mol = molecule

    random_coords = rdkit_mol.GetNumAtoms() > 90 or rdkit_mol.GetNumBonds() > 100  # rule of thumb, obtained from data
    rdkit_mol = Chem.AddHs(rdkit_mol)

    rdDistGeom.EmbedMultipleConfs(rdkit_mol, numConfs=num_conformers, maxAttempts=max_attempts, randomSeed=23, numThreads=-1, useRandomCoords=random_coords)

    # check if conformer was generated
    if rdkit_mol.GetNumConformers() > 0:
        return rdkit_mol

    print(f"Failed to generate conformation for molecule: {molecule if isinstance(molecule, str) else Chem.MolToSmiles(molecule)}")

    return None


def mol2pymatgen(molecule: str | Mol, **kwargs) -> pymatgen.core.Molecule | None:
    """
    Convert a molecule from SMILES or RDKit Mol to a pymatgen Molecule object.
    Args:
        molecule: A SMILES string or RDKit molecule.
    kwargs: Additional keyword arguments for the compute_conformer function.
    Returns:
        The pymatgen Molecule object or None if the conversion failed.

    """
    if isinstance(molecule, str):
        rdkit_mol = compute_conformer(molecule, **kwargs)
    else:
        rdkit_mol = molecule
        # check if rdkit_mol has a conformation
        if rdkit_mol.GetNumConformers() == 0:
            rdkit_mol = compute_conformer(rdkit_mol, **kwargs)

    # check if conformer is present
    if rdkit_mol is None:
        return None

    conformer = rdkit_mol.GetConformer(0)

    species = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
    coords = [conformer.GetAtomPosition(i) for i in range(rdkit_mol.GetNumAtoms())]
    coords = [[pos.x, pos.y, pos.z] for pos in coords]

    return Molecule(species, coords)


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


def mol2graph(molecule: str | Mol) -> nx.Graph:
    """
    Get a graph representation of a molecule.

    Args:
        molecule: A SMILES string or RDKit molecule.
    Returns:
        (nx.Graph) The graph representation of the molecule
    """
    if isinstance(molecule, str):
        molecule = Chem.MolFromSmiles(molecule)

    if molecule is None:
        raise ValueError(f"Mol: {str(molecule)} of {molecule} is None")

    adjacency_matrix = Chem.GetAdjacencyMatrix(molecule, useBO=True)
    return nx.from_numpy_array(adjacency_matrix)


def get_aromatic_rings(molecule: Mol | str) -> list[tuple[int]]:
    """
    Get all aromatic rings in a molecule.

    Args:
        molecule (Mol | str): The molecule to get aromatic rings from.
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
    if isinstance(molecule, str):
        mol = Chem.MolFromSmiles(molecule)
    elif isinstance(molecule, Mol):
        mol = molecule
    else:
        raise ValueError("Input must be a SMILES string or an RDKit Mol object.")

    ri = mol.GetRingInfo()

    aromatic_rings = []
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(atom).GetIsAromatic() for atom in ring):
            aromatic_rings.append(tuple(ring))
    return aromatic_rings
