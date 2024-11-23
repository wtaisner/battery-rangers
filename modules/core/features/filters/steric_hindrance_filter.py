"""Filter that leaves molecules without steric hindrance."""
import numpy as np
from rdkit import Chem

from modules.core.features.filters.generic_filter import GenericMoleculeFilter
from modules.core.features.utils import smiles_to_xyz


class StericHindranceFilter(GenericMoleculeFilter):
    """Filter that leaves molecules without steric hindrance."""

    def apply(self, smiles: list[str], **kwargs) -> list[str]:
        """
        Apply the filter to a list of SMILES strings.

        Args:
            smiles (list[str]): The list of SMILES strings to filter.
        Returns:
            list[str]: The list of SMILES strings that passed the filter.
        """
        no_steric_hindrance_smiles = []
        for sml in smiles:
            path = smiles_to_xyz(sml, "sh_tmp")
            coordinates = np.loadtxt(path, skiprows=1, usecols=(1, 2, 3))
            nitrogen_indices = self.get_indices_of_n(sml)
            distances = self.get_distances_between_n(nitrogen_indices, coordinates)
            if np.any(distances < 4.1):
                continue
            no_steric_hindrance_smiles.append(sml)
        return no_steric_hindrance_smiles

    @staticmethod
    def get_indices_of_n(smiles: str) -> list[int]:
        """
        Get the indices of nitrogen atoms involved in triple bonds with carbon in a given SMILES string.

        Args:
            smiles (str): A SMILES representation of the molecule.

        Returns:
            List[int]: A list of indices for nitrogen atoms bonded to carbon via a triple bond.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

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

        distances = [np.linalg.norm(xyz[idx1] - xyz[idx2]) for i, idx1 in enumerate(nitrogen_indices) for idx2 in nitrogen_indices[i + 1 :]]

        return np.array(distances)
