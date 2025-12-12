"""Filter that leaves molecules without steric hindrance."""
from itertools import combinations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.features.utils import compute_conformer
from modules.core.filters.generic_filter import GenericMoleculeFilter
from modules.generation.utils import score_value_exponential


class StericHindranceFilter(GenericMoleculeFilter):
    """
    Filter that leaves molecules without steric hindrance between nitrile groups.

    Functionality:
    1. Binary Filter: Discards molecules if any pair of nitrile nitrogens is closer than `distance_threshold`.
    2. Continuous Reward: Uses `score_value_exponential` to penalize violations.
    """

    def __init__(
        self,
        distance_threshold: float = 4.1,
    ):
        """
        Args:
            distance_threshold (float): The minimum allowed distance (Angstroms).
                                        This maps to `min_val` in scoring.
        """
        self.nitrile_pattern = Chem.MolFromSmarts("N~*")
        self.distance_threshold = distance_threshold

    def apply(self, molecules: list[Mol], **kwargs) -> list[Mol]:
        """
        Apply the binary filter to a list of RDKit molecules.
        """
        return [mol for mol in molecules if self.check_steric_hindrance(mol)]

    def check_steric_hindrance(self, mol: Mol) -> bool:
        """
        Binary check: Returns True if the molecule passes (NO steric hindrance).
        Returns False if atoms are too close.
        """
        min_dist = self._get_min_nitrile_distance(mol)

        if min_dist is None:
            # Passes if there are < 2 nitriles.
            # Fails if conformer generation broke (depending on policy, usually fail).
            matches = mol.GetSubstructMatches(self.nitrile_pattern)
            return len(matches) < 2

        # Binary strict check
        return min_dist >= self.distance_threshold

    def get_reward(self, mol: Mol) -> float:
        """
        Calculates a continuous score (0.0 to 1.0) for RL using `score_value_exponential`.
        """
        if mol is None:
            return 0.0

        min_dist = self._get_min_nitrile_distance(mol)

        # Case 1: Less than 2 nitriles -> No hindrance possible -> Perfect score
        if min_dist is None:
            matches = mol.GetSubstructMatches(self.nitrile_pattern)
            if len(matches) < 2:
                return 1.0
            # Case 2: Conformer generation failed -> Zero score
            return 0.0

        # Case 3: Calculate Score
        # We define the "optimal range" as [4.1, infinity]
        # Any distance < 4.1 triggers the exponential decay in the helper function.
        return score_value_exponential(value=min_dist, min_val=self.distance_threshold, max_val=np.inf)  # e.g., 4.1

    def _get_min_nitrile_distance(self, mol: Mol) -> float | None:
        """
        Helper: Generates 3D conformer and finds the minimum distance between any two nitrile nitrogens.
        Returns None if < 2 nitriles or conformer generation fails.
        """
        matches = mol.GetSubstructMatches(self.nitrile_pattern)
        if len(matches) < 2:
            return None

        # Check for existing conformer, else compute one
        if mol.GetNumConformers() == 0:
            mol_3d = compute_conformer(mol)
            if mol_3d is None or mol_3d.GetNumConformers() == 0:
                return None
            conformer = mol_3d.GetConformer(0)
        else:
            conformer = mol.GetConformer(0)

        nitrogen_indices = [match[0] for match in matches]

        min_distance = float("inf")
        found_pair = False

        # Check all pairs to find the "worst case" (closest) distance
        for idx1, idx2 in combinations(nitrogen_indices, 2):
            pos1 = np.array(conformer.GetAtomPosition(idx1))
            pos2 = np.array(conformer.GetAtomPosition(idx2))
            dist = np.linalg.norm(pos1 - pos2)

            if dist < min_distance:
                min_distance = dist
            found_pair = True

        return min_distance if found_pair else None

    def filter_from_property(self, properties: dict) -> bool:
        """
        Reads properties from a dictionary (database) and decides whether to filter the molecule.
        """
        # Logic: If 'steric_hindrance' is True, we filter it OUT.
        # So we return False (don't keep).
        return not properties.get("steric_hindrance", False)
