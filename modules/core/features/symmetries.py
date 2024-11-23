"""Symmetry analysis functions."""
import logging
import os
from enum import Enum

import pandas as pd
from pymatgen.core.structure import Molecule, Structure
from pymatgen.symmetry.analyzer import PointGroupAnalyzer, SpacegroupAnalyzer

from modules.core.features.utils import get_pymatgen_molecule_from_smiles, visualize_structure


class AvailableSymmetry(Enum):
    """Enum for available symmetry types."""

    NODE = "node"
    EDGE = "edge"
    RING_BONDS = "ring_bonds"
    RING_NODES = "ring_nodes"
    RING_OUTER_PLANE = "ring_outer_plane"


def translate_point_group_to_symmetry_description(symmetry_code: str) -> str:
    """
    Translates the point group code e.g. D3h to the corresponding description.

    Args:
        symmetry_code (str): The symmetry code to translate.

    Returns:
        str: The translated symmetry code.
    """
    translation_table_path = os.path.join("../../../data/symmetries/symmetry_translation.csv")  # TODO: change

    table = pd.read_csv(translation_table_path).astype(str)

    if symmetry_code in table["Point group"].to_list():
        return table[table["Point group"] == symmetry_code]["Simple description of typical geometry"].to_list()[0]

    logging.debug(
        f'Could not find symmetry code {symmetry_code} in translation table: \
            {table["Point group"].to_list()}'
    )
    return "Unknown"


def analyse_symmetry_point_group(molecule: Molecule, visualize: bool = False) -> tuple:
    """
    Analyzes the symmetry of a given crystal structure.

    Args:
        molecule (Molecule): Molecule to analyze.
        visualize (bool, optional): Whether to visualize the structure. Defaults to False.

    Returns:
        tuple: A tuple containing the following symmetry analysis results:
             - symmetry_operations (list): List of symmetry operations.
             - rotational_symmetry (int): The rotational symmetry number.
             - point_group (str): The point group of the structure.
             - equivalent_atoms (list): List of equivalent atoms.
    """

    # Get the space group
    pga = PointGroupAnalyzer(molecule)
    symmetry_operations = pga.get_symmetry_operations()
    rotational_symmetry = pga.get_rotational_symmetry_number()
    point_group = pga.get_pointgroup()
    equivalent_atoms = pga.get_equivalent_atoms()

    # Log the results
    logging.debug(f"Symmetry operations: {symmetry_operations}")
    logging.debug(f"Rotational symmetry number: {rotational_symmetry}")
    logging.debug(f"Point group: {point_group}")
    logging.debug(f"Equivalent atoms: {equivalent_atoms}")

    symmetry_description = translate_point_group_to_symmetry_description(str(point_group))
    logging.debug(f"Symmetry description: {symmetry_description}")

    if visualize:
        structure = molecule.get_boxed_structure(23, 23, 23)
        visualize_structure(structure, show_polyhedron=False)

    return (
        symmetry_operations,
        rotational_symmetry,
        str(point_group),
        equivalent_atoms,
    )


def analyse_symmetry_space_group(structure: Structure, visualize: bool = False) -> tuple:
    """
    Analyzes the symmetry of a given crystal structure.

    Args:
        structure (Structure): The crystal structure to analyze.
        visualize (bool, optional): Whether to visualize the structure. Defaults to False.

    Returns:
        tuple: A tuple containing the following symmetry analysis results:
            - symmetry_operations (list): List of symmetry operations.
            - space_group_symbol (str): The space group symbol.
            - space_group_number (int): The space group number.
            - point_group_symbol (str): The point group symbol.
            - point_group_operations (list): List of point group operations.
            - space_group_operations (list): List of space group operations.
    """
    logging.debug("Starting analyse_symmetry_space_group")

    # Analyze the symmetry using SpaceGroupAnalyzer
    sga = SpacegroupAnalyzer(structure)
    symmetry_operations = sga.get_symmetry_operations()
    space_group_symbol = sga.get_space_group_symbol()
    space_group_number = sga.get_space_group_number()
    point_group_symbol = sga.get_point_group_symbol()
    point_group_operations = sga.get_point_group_operations()
    space_group_operations = sga.get_space_group_operations()
    is_laue = sga.is_laue()

    # Log the results
    logging.debug(f"Space group symbol: {space_group_symbol}")
    logging.debug(f"Space group number: {space_group_number}")
    logging.debug(f"Point group symbol: {point_group_symbol}")
    logging.debug(f"Is Laue: {is_laue}")

    if visualize:
        visualize_structure(structure)

    return (
        symmetry_operations,
        space_group_symbol,
        space_group_number,
        point_group_symbol,
        point_group_operations,
        space_group_operations,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Starting pymatgen_playground.py")

    SMILES = "Nc1ccc(-c2nc(-c3ccc(N)cc3)nc(-c3ccc(N4C(=O)c5ccc6c7c(ccc(c57)C4=O)C(=O)OC6=O)cc3)n2)cc1"

    molecule = get_pymatgen_molecule_from_smiles(SMILES)

    analyse_symmetry_point_group(molecule, visualize=True)
