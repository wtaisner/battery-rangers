"""Filter that leaves molecules without steric hindrance."""
import os
from itertools import combinations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.features.filters.generic_filter import GenericMoleculeFilter
from modules.core.features.utils import mol_to_xyz


class StericHindranceFilter(GenericMoleculeFilter):
    """Filter that leaves molecules without steric hindrance."""

    def apply(self, molecules: list[Mol], **kwargs) -> list[Mol]:
        """
        Apply the filter to a list of RDKIT molecules.

        Args:
            molecules (list[Mol]): The list of RDKIT molecules to filter.
        Returns:
            list[Mol]: The list of RDKIT molecules that passed the filter.
        """
        no_steric_hindrance_molecules = []
        for mol in molecules:
            path = mol_to_xyz(mol, save_file=True, directory="sh_tmp")
            coordinates = np.loadtxt(path, skiprows=1, usecols=(1, 2, 3))

            # remove the file after loading
            try:
                os.remove(path)
            except OSError as e:
                print(f"Error removing file {path}: {e}")

            nitrogen_indices = self.get_indices_of_n(mol)
            distances = self.get_distances_between_n(nitrogen_indices, coordinates)
            if np.any(distances < 4.1):
                continue
            no_steric_hindrance_molecules.append(mol)
        return no_steric_hindrance_molecules

    @staticmethod
    def get_indices_of_n(mol: Mol) -> list[int]:
        """
        Get the indices of nitrogen atoms involved in triple bonds with carbon in a given molecule.

        Args:
            mol (Mol): A RDKit Mol object

        Returns:
            List[int]: A list of indices for nitrogen atoms bonded to carbon via a triple bond.
        """

        nitrogen_indices = [
            bond.GetEndAtomIdx() if bond.GetBeginAtom().GetSymbol() == "C" and bond.GetEndAtom().GetSymbol() == "N" else bond.GetBeginAtomIdx()
            for bond in mol.GetBonds()
            if bond.GetBondType() == Chem.BondType.TRIPLE and {"C", "N"} == {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}
        ]

        return nitrogen_indices

    @staticmethod
    def get_distances_between_n(nitrogen_indices: list[int], xyz: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise distances between nitrogen atoms based on their 3D coordinates.

        Args:
            nitrogen_indices (List[int]): Indices of nitrogen atoms.
            xyz (np.ndarray): A numpy array of shape (N, 3) representing 3D coordinates of atoms.

        Returns:
            np.ndarray: A 1D array of pairwise distances between nitrogen atoms.
        """
        if not isinstance(xyz, np.ndarray) or xyz.shape[1] != 3:
            raise ValueError("xyz must be a numpy array with shape (N, 3)")

        try:
            distances = [np.linalg.norm(xyz[idx1] - xyz[idx2]) for idx1, idx2 in combinations(nitrogen_indices, 2)]
        except IndexError as e:
            print("Error in calculating distances.: ", e)
            distances = [5.0]
        return np.array(distances)
