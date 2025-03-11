"""Filter that leaves molecules with certain substructures present at the "end" of the molecule."""
from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.features.filters.generic_filter import GenericMoleculeFilter


class SMARTSFilter(GenericMoleculeFilter):
    """Filter that leaves molecules with certain substructures present at the "end" of the molecule.
    Args:
        smarts (list[str] | None): The SMARTS patterns to filter by. If None, defaults to a predefined list.
    """

    def __init__(self, smarts: list[str] | None = None):
        if smarts is None:
            self.smarts = [
                "C#N",
            ]
        else:
            self.smarts = smarts

    def apply(self, molecules: list[Mol], **kwargs) -> list[Mol]:
        """
        Apply the filter to a list of SMILES strings.

        Args:
            molecules (list[Mol]): The list of RDKit molecules to filter.
        Returns:
            list[Mol]: The list of RDKit molecules that passed the filter.
        """
        return [mol for mol in molecules if self.check_smarts(mol)]

    def check_smarts(self, mol: Chem.Mol) -> bool:
        """
        Check if the molecule contains at least one of the SMARTS patterns in cardinality >=2.
        Args:
            mol (Chem.Mol): The molecule to evaluate
        Returns:
            bool: True if the molecule contains at least one of the SMARTS patterns in cardinality >=2, False otherwise.
        """
        for smarts in self.smarts:
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            if len(matches) >= 2:
                return True

        return False
