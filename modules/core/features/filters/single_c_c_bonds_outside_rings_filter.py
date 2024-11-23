"""Filter that leaves molecules without single C-C bonds outside rings."""
from warnings import deprecated

from rdkit import Chem

from modules.core.features.filters import GenericMoleculeFilter


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
    @deprecated
    def __check_single_carbon_bond_outside_ring(smiles):  # pylint: disable=unused-private-member
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

    @staticmethod
    def check_single_carbon_bond_outside_ring(smiles: str) -> bool:
        """
        Check if there exists a single carbon-carbon bond outside any ring in the molecule.

        Args:
            smiles (str): A SMILES representation of the molecule.

        Returns:
            bool: True if there is at least one single carbon-carbon bond outside a ring, False otherwise.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # Retrieve ring information
        ring_info = mol.GetRingInfo()

        # Get all atoms that are part of any ring
        atoms_in_rings = set()
        for ring in ring_info.AtomRings():
            atoms_in_rings.update(ring)

        # Iterate through bonds and check for single C-C bonds outside rings
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
                if begin_atom.GetSymbol() == "C" and end_atom.GetSymbol() == "C":
                    # Check if either atom is outside the ring
                    if begin_atom.GetIdx() not in atoms_in_rings or end_atom.GetIdx() not in atoms_in_rings:
                        return True

        return False
