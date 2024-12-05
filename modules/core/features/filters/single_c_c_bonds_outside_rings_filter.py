"""Filter that leaves molecules without single C-C bonds outside rings."""

from rdkit import Chem

from modules.core.features.filters.generic_filter import GenericMoleculeFilter


class SingleCCBondsOutsideRingsFilter(GenericMoleculeFilter):
    """Filter that leaves molecules without single C-C bonds outside rings."""

    def apply(self, smiles: list[str], **kwargs) -> list[str]:
        """
        Apply the filter to a list of SMILES strings.

        Args:
            smiles (list[str]): The list of SMILES strings to filter.
        Returns:
            list[str]: The list of SMILES strings that passed the filter.
        """
        final_smiles = []
        for sml in smiles:
            if not self.check_single_carbon_bond_outside_ring(sml):
                final_smiles.append(sml)
        return final_smiles

    @staticmethod
    def check_single_carbon_bond_outside_ring(smiles: str) -> bool:
        """
        Check if there exists a single carbon-carbon bond outside any ring in the molecule.

        Args:
            smiles (str): A SMILES representation of the molecule.

        Returns:
            bool: True if there is at least one single carbon-carbon bond outside a ring, False otherwise.
        """
        m = Chem.MolFromSmiles(smiles)
        ri = m.GetRingInfo()
        if ri.AtomRings():
            atoms_in_rings = set()
            for ring in ri.AtomRings():
                for atom in ring:
                    atoms_in_rings.add(atom)
            for bond in m.GetBonds():
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    if bond.GetBeginAtom().GetSymbol() == "C" and bond.GetEndAtom().GetSymbol() == "C":
                        if bond.GetBeginAtom().GetIdx() not in atoms_in_rings or bond.GetEndAtom().GetIdx() not in atoms_in_rings:
                            return True
                        return False
                    return False
                return False
        else:
            for bond in m.GetBonds():
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    if bond.GetBeginAtom().GetSymbol() == "C" and bond.GetEndAtom().GetSymbol() == "C":
                        return True
        return False
