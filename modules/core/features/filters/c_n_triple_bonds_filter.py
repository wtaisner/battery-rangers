"""Filter that leaves molecules with triple bonds between C and N atoms."""
from rdkit import Chem

from modules.core.features.filters import GenericMoleculeFilter


class CNTripleBondsFilter(GenericMoleculeFilter):
    """Filter that leaves molecules with triple bonds."""

    def apply(self, smiles: list[str], **kwargs) -> list[str]:
        """
        Apply the filter to a list of SMILES strings.

        Args:
            smiles (list[str]): The list of SMILES strings to filter.
        Returns:
            list[str]: The list of SMILES strings that passed the filter.
        """
        tmp_smiles = []
        mols = [Chem.MolFromSmiles(x) for x in smiles]
        mols_sym_triple_bonds = []
        for i, mol in enumerate(mols):
            if self.count_cn_triple_bonds(mol) >= 2:
                mols_sym_triple_bonds.append(mol)
                tmp_smiles.append(smiles[i])
        return tmp_smiles

    @staticmethod
    def count_cn_triple_bonds(mol: Chem.Mol) -> int:
        """Count the number of triple bonds between C and N atoms in a molecule.

        Args:
            mol (Chem.Mol): The molecule to analyze.
        Returns:
            int: The number of triple bonds between C and N atoms.
        """
        triple_bond_count = 0
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.TRIPLE:
                begin_atom = bond.GetBeginAtom().GetSymbol()
                end_atom = bond.GetEndAtom().GetSymbol()
                if (begin_atom == "C" and end_atom == "N") or (begin_atom == "N" and end_atom == "C"):
                    triple_bond_count += 1
        return triple_bond_count
