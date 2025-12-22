"""Utility functions for the features."""
import logging

import networkx as nx

# import pymatgen.core
# from pymatgen.core.structure import Molecule, Structure
# from pymatgen.vis.structure_vtk import StructureVis
from rdkit import Chem
from rdkit.Chem import AllChem, Mol, rdDistGeom

# debug logger
logger = logging.getLogger(__name__)  # __name__ ensures the logger is specific to this module
logger.setLevel(logging.INFO)


def compute_conformer(molecule: str | Chem.Mol, num_conformers: int = 1, max_attempts: int = 2000, save_file: bool = False, filename: str = "") -> Chem.Mol | None:
    """
    Compute 3D conformer(s) for a given molecule using RDKit's ETKDGv3 algorithm and store in file.
    Args:
        molecule: A SMILES string or RDKit molecule.
        num_conformers: Number of conformers to generate.
        max_attempts: Maximum attempts for embedding.
        save_file: Whether to save the generated conformer(s) to an SDF file.
        filename: The filename to save the SDF file. If empty, uses molecule name.
    Returns:
        The RDKit molecule with 3D coordinates or None if generation failed.
    """

    # 1. Standardize Input
    if isinstance(molecule, str):
        mol = Chem.MolFromSmiles(molecule)
    else:
        mol = Chem.Mol(molecule)  # Create a copy to avoid modifying original input

    if not mol:
        logger.warning(f"Invalid molecule input: {molecule}")
        return None

    use_random_coords = mol.GetNumAtoms() > 90 or mol.GetNumBonds() > 100

    mol_with_hs = Chem.AddHs(mol)

    if not mol_with_hs.HasProp("_Name"):
        mol_with_hs.SetProp("_Name", Chem.MolToSmiles(mol))

    # 5. Generate Conformers
    conf_ids = rdDistGeom.EmbedMultipleConfs(mol=mol_with_hs, randomSeed=23, numConfs=num_conformers, maxAttempts=max_attempts, useRandomCoords=use_random_coords, numThreads=8, ETversion=2)

    if not conf_ids:
        logger.info(f"Failed to generate conformation for: {mol_with_hs.GetProp('_Name')}")
        return None

    try:
        AllChem.MMFFOptimizeMolecule(mol_with_hs)
    except Exception:
        pass

    if save_file:
        if not filename:
            safe_name = mol_with_hs.GetProp("_Name")
            filename = f"{safe_name}.sdf"

        try:
            with Chem.SDWriter(filename) as writer:
                writer.write(mol_with_hs)
        except Exception as e:
            logger.error(f"Failed to save conformation for {Chem.MolToSmiles(mol)}: {e}")

    mol_no_hs = Chem.RemoveHs(mol_with_hs)

    return mol_no_hs


# def mol2pymatgen(molecule: str | Mol, **kwargs) -> pymatgen.core.Molecule | None:
#     """
#     Convert a molecule from SMILES or RDKit Mol to a pymatgen Molecule object.
#     Args:
#         molecule: A SMILES string or RDKit molecule.
#     kwargs: Additional keyword arguments for the compute_conformer function.
#     Returns:
#         The pymatgen Molecule object or None if the conversion failed.

#     """
#     if isinstance(molecule, str):
#         rdkit_mol = compute_conformer(molecule, **kwargs)
#     else:
#         rdkit_mol = molecule
#         # check if rdkit_mol has a conformation
#         if rdkit_mol.GetNumConformers() == 0:
#             rdkit_mol = compute_conformer(rdkit_mol, **kwargs)

#     # check if conformer is present
#     if rdkit_mol is None:
#         return None

#     conformer = rdkit_mol.GetConformer(0)

#     species = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
#     coords = [conformer.GetAtomPosition(i) for i in range(rdkit_mol.GetNumAtoms())]
#     coords = [[pos.x, pos.y, pos.z] for pos in coords]

#     return Molecule(species, coords)


# def visualize_structure(structure: Structure, **kwargs) -> None:
#     """
#     Visualizes the given structure using StructureVis.

#     Args:
#         structure (Structure): The structure to be visualized.
#     Returns:
#         None
#     """
#     stvis = StructureVis(**kwargs)
#     stvis.set_structure(structure)
#     stvis.show()


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
