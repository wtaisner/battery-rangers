"""A filter that matches SMARTS patterns with inclusion and exclusion rules and given cardinality."""
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.filters.generic_filter import GenericMoleculeFilter


@dataclass(frozen=True)
class InclusionRule:
    """
    Positive Rule: A molecule is APPROVED if it matches this pattern with specific cardinality.

    Args:
        smarts_pattern (str): The SMARTS pattern to look for.
        min_cardinality (int): The molecule is approved if the pattern
                               appears AT LEAST this number of times.
    """

    smarts_pattern: str
    min_cardinality: int = 2


@dataclass(frozen=True)
class ExclusionRule:
    """
    Negative Rule: A molecule is DISCARDED if it matches these patterns.

    Args:
        smarts_pattern (str): The SMARTS pattern to look for.
        max_cardinality (int): The molecule is discarded if the pattern
                               appears MORE than this number of times.
                               Set to 0 to discard if present at all.
        penalty_sensitivity (float): Used for RL scoring. Controls how sharply the
                                     reward drops when the rule is violated.
                                     Default 1.0. Higher = steeper penalty.
    """

    smarts_pattern: str
    max_cardinality: int = 0
    penalty_sensitivity: float = 1.0


class SMARTSFilter(GenericMoleculeFilter):
    """
    Filter that leaves molecules matching a set of expressive SMARTS rules.

    A molecule is kept if it satisfies ANY of the provided `Rule` objects.
    A `Rule` is satisfied if the molecule contains at least `min_cardinality`
    matches for EVERY SMARTS pattern defined within that rule.

    Args:
        inclusion_rules (list[InclusionRule] | None): A list of Rule objects to filter by.
            If None, defaults to a simple list where each original SMARTS
            pattern must be present at least once.
        exclusion_rules (list[ExclusionRule] | None): A list of ExclusionRule objects.
            If a molecule matches any of these exclusion rules, it is discarded.
    """

    def __init__(self, inclusion_rules: list[InclusionRule] | None = None, exclusion_rules: list[ExclusionRule] | None = None):
        if inclusion_rules is None:
            self.inclusion_rules = [
                # Function 1: CTF pattern
                InclusionRule("C#N", 2),
            ]
        else:
            self.inclusion_rules = inclusion_rules

        if exclusion_rules is None:
            self.exclusion_rules = [
                # Function 1: has_rings_less_than_5_atoms
                ExclusionRule(smarts_pattern="[r3,r4]", max_cardinality=0),
                # Function 2: has_multiple_triple_C_C_bonds
                ExclusionRule(smarts_pattern="[#6]#[#6]", max_cardinality=1),
                # Function 3: has_N_N_bond
                ExclusionRule(smarts_pattern="[#7]~[#7]", max_cardinality=0),
            ]
        else:
            self.exclusion_rules = exclusion_rules

        self._compiled_inclusion_rules = []
        for rule in self.inclusion_rules:
            pattern = Chem.MolFromSmarts(rule.smarts_pattern)

            if pattern is None:
                raise ValueError(f"Invalid inclusion SMARTS pattern: {rule.smarts_pattern}")

            self._compiled_inclusion_rules.append((pattern, rule.min_cardinality))

        # Storing (pattern_object, max_cardinality, penalty_sensitivity)
        self._compiled_exclusion_rules = []
        for rule in self.exclusion_rules:
            pattern = Chem.MolFromSmarts(rule.smarts_pattern)

            if pattern is None:
                raise ValueError(f"Invalid exclusion SMARTS pattern: {rule.smarts_pattern}")

            self._compiled_exclusion_rules.append((pattern, rule.max_cardinality, rule.penalty_sensitivity))

    def apply(self, molecules: list[Mol], **kwargs) -> list[Mol]:
        """
        Apply the filter to a list of RDKit molecules.

        Args:
            molecules (list[Mol]): The list of RDKit molecules to filter.
        Returns:
            list[Mol]: The list of RDKit molecules that passed the filter.
        """
        return [mol for mol in molecules if self.check_smarts(mol)]

    def check_smarts(self, mol: Chem.Mol) -> bool:
        """
        Binary check: Does the molecule pass the filter?

        Passes if:
        1. It does NOT violate any ExclusionRule.
        2. It satisfies **ALL** InclusionRules.

        Args:
            mol (Chem.Mol): The molecule to evaluate.
        Returns:
            bool: True if the molecule passes the filter, False otherwise.
        """
        if mol is None:
            return False

        # 1. Check Exclusions (Fail fast)
        for pattern, max_cardinality, _ in self._compiled_exclusion_rules:
            matches = mol.GetSubstructMatches(pattern)
            if len(matches) > max_cardinality:
                return False

        # 2. Check Inclusions (Must satisfy ALL rules)
        for pattern, min_cardinality in self._compiled_inclusion_rules:
            num_matches = len(mol.GetSubstructMatches(pattern))
            if num_matches < min_cardinality:
                return False

        return True

    def get_reward(self, mol: Chem.Mol) -> float:
        """
        Calculates a continuous score (0.0 to 1.0) for RL.

        - 1.0: Perfect match (passes binary filter).
        - < 1.0: Partial match or contains forbidden structures.

        Scoring Logic:
        - Exclusions: Score = 1 / (1 + (excess_count * sensitivity))
        - Inclusions: Score = min(1.0, count / target)
        - Final: Arithmetic mean of all rule scores.

        Args:
            mol (Chem.Mol): The molecule to evaluate.
        Returns:
            float: A score between 0.0 and 1.0.
        """
        if mol is None:
            return 0.0

        scores = []

        # --- 1. Score Exclusions (Hyperbolic Decay) ---
        for pattern, max_allowed, sensitivity in self._compiled_exclusion_rules:
            count = len(mol.GetSubstructMatches(pattern))
            if count <= max_allowed:
                scores.append(1.0)
            else:
                excess = count - max_allowed
                # Decay score based on how much we exceeded the limit
                score = 1.0 / (1.0 + (excess * sensitivity))
                scores.append(score)

        # --- 2. Score Inclusions (Linear Ramp) ---
        for pattern, min_required in self._compiled_inclusion_rules:
            count = len(mol.GetSubstructMatches(pattern))
            if count >= min_required:
                scores.append(1.0)
            else:
                # Partial credit: e.g., found 1 but needed 2 -> 0.5
                scores.append(count / float(min_required))

        # --- 3. Final Aggregation ---
        if not scores:
            return 0.0

        # Return average of all component scores
        return sum(scores) / len(scores)

    def filter_from_property(self, properties: dict) -> bool:
        """
        Reads properties from a dictionary (database) and decides whether to filter the molecule.
        """
        return properties.get("smarts_filter", False)
