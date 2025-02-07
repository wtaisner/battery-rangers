"""Filter that leaves molecules with triple bonds between C and N atoms."""
from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.features.filters.generic_filter import GenericMoleculeFilter


class CNTripleBondsFilter(GenericMoleculeFilter):
    """Filter that leaves molecules with triple bonds."""

    def apply(self, molecules: list[Mol], **kwargs) -> list[Mol]:
        """
        Apply the filter to a list of SMILES strings.

        Args:
            molecules (list[Mol]): The list of RDKit molecules to filter.
        Returns:
            list[Mol]: The list of RDKit molecules that passed the filter.
        """
        to_pass = []
        for i, mol in enumerate(molecules):
            if self.count_cn_triple_bonds(mol) >= 2:
                to_pass.append(molecules[i])
        return to_pass

    @staticmethod
    def count_cn_triple_bonds(mol: Chem.Mol) -> int:
        """Count the number of triple bonds between C and N atoms in a molecule.

        Args:
            mol (Chem.Mol): The molecule to analyze.
        Returns:
            int: The number of C#N or N#C bonds in the molecule.
        """

        return len(mol.GetSubstructMatches(Chem.MolFromSmarts("C#N")))
