"""Filter that leaves molecules without steric hindrance."""
from itertools import combinations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.features.utils import compute_conformer
from modules.core.filters.generic_filter import GenericMoleculeFilter


class StericHindranceFilter(GenericMoleculeFilter):
    """
    Filter that leaves molecules without steric hindrance between nitrile groups.
    This version replicates the user's method of generating a fresh 3D
    conformation and checking it without optimization.
    """

    def __init__(self):
        """
        Initialize the filter by compiling the SMARTS pattern once.
        """
        self.nitrile_pattern = Chem.MolFromSmarts("N~*")

    def apply(self, molecules: list[Mol], **kwargs) -> list[Mol]:
        """
        Apply the filter to a list of RDKIT molecules.

        Args:
            molecules (list[Mol]): The list of RDKIT molecules to filter.
        Returns:
            list[Mol]: The list of RDKIT molecules that passed the filter.
        """
        no_steric_hindrance_molecules = []
        distance_threshold = 4.1  # Angstroms

        for original_mol in molecules:
            # Find nitrile groups first to avoid unnecessary 3D generation.
            matches = original_mol.GetSubstructMatches(self.nitrile_pattern)
            if len(matches) < 2:
                no_steric_hindrance_molecules.append(original_mol)
                continue

            if original_mol.GetNumConformers() == 0:
                original_mol = compute_conformer(original_mol)

            if original_mol is None or original_mol.GetNumConformers() == 0:  # No conformer generated
                continue

            nitrogen_indices = [match[0] for match in matches]

            conformer = original_mol.GetConformer(0)
            # Check distances between all pairs of nitrile nitrogen atoms
            is_hindered = False
            for idx1, idx2 in combinations(nitrogen_indices, 2):
                pos1 = np.array(conformer.GetAtomPosition(idx1))
                pos2 = np.array(conformer.GetAtomPosition(idx2))
                distance = np.linalg.norm(pos1 - pos2)

                if distance < distance_threshold:
                    is_hindered = True
                    break

            if not is_hindered:
                no_steric_hindrance_molecules.append(original_mol)

        return no_steric_hindrance_molecules
