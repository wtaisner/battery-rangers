"""Filter that leaves molecules with certain substructures present at the "end" of the molecule."""
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.filters.generic_filter import GenericMoleculeFilter


@dataclass(frozen=True)
class Rule:
    """
    A molecule matches this rule if it contains at least `min_cardinality`
    substructure matches for EVERY SMARTS pattern in the `smarts_patterns` tuple.
    """

    smarts_patterns: list[str]  # List of SMARTS patterns to match
    min_cardinality: int = 2


class SMARTSFilter(GenericMoleculeFilter):
    """
    Filter that leaves molecules matching a set of expressive SMARTS rules.

    A molecule is kept if it satisfies ANY of the provided `Rule` objects.
    A `Rule` is satisfied if the molecule contains at least `min_cardinality`
    matches for EVERY SMARTS pattern defined within that rule.

    Args:
        rules (list[Rule] | None): A list of Rule objects to filter by.
            If None, defaults to a simple list where each original SMARTS
            pattern must be present at least once.
    """

    def __init__(self, rules: list[Rule] | None = None):
        if rules is None:
            self.rules = [
                Rule(["C#N"], 2),
                Rule(["[NH2]", "Br"], 1),
                Rule(["[NH2]", "Cl"], 1),
                Rule(["Cl"], 2),  # TODO: should be depended on the number of symmetry axes - i.e. 2 for 1 and 3 for 3
                Rule(["Br"], 2),  # TODO: as above
                Rule(["[CH]=O", "[NH2]"], 1),
                Rule(["[NH2]", "c1nccc1"], 2),
                Rule(["[NH2]", "[#6](-c)-[#7r6]-[#6](-c)"], 2),
                Rule(["[OH]", "O1-B-O-c:c1"], 1),
                Rule(["[#6]=[#8]", "[#7]1~[#6]~[#6]~[#7]~[#6]~[#6]~1"], 1),
            ]

        else:
            self.rules = rules

        self._compiled_rules = []
        for rule in self.rules:
            compiled_patterns = tuple(Chem.MolFromSmarts(s) for s in rule.smarts_patterns)

            # Check for invalid SMARTS
            if not all(compiled_patterns):
                invalid_smarts = [s for s, p in zip(rule.smarts_patterns, compiled_patterns) if p is None]
                raise ValueError(f"Invalid SMARTS pattern(s) found: {invalid_smarts}")

            self._compiled_rules.append((compiled_patterns, rule.min_cardinality))

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
        Check if the molecule satisfies any of the defined rules.

        A molecule passes if it satisfies at least one rule.
        A rule is satisfied if all of its SMARTS patterns meet the minimum
        cardinality requirement.

        Args:
            mol (Chem.Mol): The molecule to evaluate.
        Returns:
            bool: True if the molecule passes the filter, False otherwise.
        """
        if mol is None:
            return False

        for idx, (compiled_patterns, min_cardinality) in enumerate(self._compiled_rules):
            rule_is_satisfied = True

            for pattern in compiled_patterns:
                # Get the number of non-overlapping matches
                num_matches = len(mol.GetSubstructMatches(pattern))

                # If this pattern does not meet the minimum count, the rule fails.
                if num_matches < min_cardinality:
                    rule_is_satisfied = False
                    break  # Exit the inner loop and check the next rule.

            # If the inner loop completed without being broken, the rule is satisfied.
            if rule_is_satisfied:
                return True  # The molecule passed the filter.

        # If we've checked all rules and none were satisfied, the molecule fails.
        # print(f" {Chem.MolToSmiles(mol)} did not satisfy any rules.")
        return False
