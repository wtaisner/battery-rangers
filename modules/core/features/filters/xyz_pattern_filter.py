"""Filter that leaves molecules without X-Y-Z pattern.

Za "brak sprzężenia"  (oczywiście jest to pojęcie bardzo uproszczone) w tej strukturze związku odpowiadają atomy -S-C-C - i pojedyncze wiązania pomiędzy nimi. Należy zaznaczyć, że ten skrót myślowy - brak sprzężenia - dotyczy elektronów pi znajdujących się na orbitalach p w danym atomie, które tworzą wiązania wielokrotne pomiędzy atomami (podwójne i potrójne). Przykładowo w pierścieniu benzenowym, mamy wiązania wielokrotne podwójne, pomiędzy atomami C=C oraz C=N. Przykładowo, jedno wiązanie w C=C jest wiązaniem sigma, czyli para elektronów (1 elektron od 1 atomu C oraz 1elektron od 2 atomu C) tworzy wiązanie chemiczne pomiędzy atomami C-C. Ta para elektronów znajduje się na osi rdzeni atomowych atomów C-C, dlatego jest nazywane sigma. Są one silnie przyciągane przez rdzenie atomowe obu atomów, stad nie mogą one przemieszczać się pomiędzy atomami. Wszystkie wiązania pojedyncze w strukturach związków mają taki charakter. Czyli w odniesieniu do -S-C-C, wszystkie wiązania mają charakter sigma, elektrony nie przemieszczają się.
W wiązaniach wielokrotnych, jedno wiązanie ma zawsze charakter sigma, a pozostałe mają charakter pi. Czyli w C=C, 1 wiązanie to sigma a 2 wiązanie to pi. Wiązanie chemiczne pi, oznacza, ze tworzą to wiązanie elektrony pi znajdujące się na orbitalach p. Jeżeli orbitale p sa zorientowane w przestrzeni w taki sposób, ze nie leża w osi rdzeni atomowych, to są słabej przyciagne przez nie i mają większą swobodę ruchu w przestrzeni wokół rdzeni (po orbitalach atomowych). Orbital atomowy - w ujęciu matematycznych - jest to przestrzeń wokół jadra atomowego, która można opisać funkcja, największe prawdopodobieństwo ruchu elektronu wokół jadra atomowego. Orbital s przedstawiany jest jako sfera, a orbital p jako dwie pętle stykające się końcami.
Analizując strukturę związku chemicznego poniżej przedstawionego, w pierscieniu benzenowym, mamy w sumie 6 wiązań sigma (4 C-C oraz 2 C-N) i 3 wiązania pi. Jeżeli odległość pomiędzy wiązaniami pi jest mała (czyli maksymalnie jedno wiazanie pojedyncze - sigma) je rozdzielające, to elektrony pi mogą swobodnie przemieszczać się w obrębie danej przestrzeni. Czyli w obrębie pierścienia benzenowego, 6 elektronów pi może swobodnie przemieszczać się.
Dalej idąc w górę struktury, mamy 1 atom C z pierścienia benzenowego, który związany jest z kolejnym atomem C, który dalej związany jest z atomem N wiązaniem potrójnym (w wiązaniu potrójnym mamy 1 wiązanie sigma i 2 wiazania pi, utworzone przez 4 elektrony - te mogą swobodnie poruszać się). Zatem mamy układ C=C-C=(potrójne)N. Elektrony pi rozdzielone są tylko 1 wiązaniem sigma (pojedynczym), więc mogą swobodnie poruszać się wzdłuż C=C-C=(potrójne)N, czyli w uproszczeniu mówimy o "sprzężeniu". Czyli elektrony pi tworzące wiązanie C=C "sprzęgaja się" z C=(potrójne)N.
Wracając do pierwotnego pytania. Elektrony pi z pierścienia benzenowego nie mogą przemieścić się do C=(potrójne)N, ponieważ rozdziela je C-S-C-C, w związku z tym, nie ma "sprzężenia" pomiędzy pierścieniem benzenowym a C=(potrójne)N.
Siarkę można oznaczyć symbolicznie jako X, ponieważ istota tutaj jest charakter wiązania chemicznego pomiędzy atomami a nie rodzaj atomu.
"""

from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.features.filters.generic_filter import GenericMoleculeFilter


class XYZPatternFilter(GenericMoleculeFilter):
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
            if not self.check_x_y_z_pattern(sml):
                final_smiles.append(sml)
        return final_smiles

    @staticmethod
    def check_x_y_z_pattern(smiles: str) -> bool:
        """Check whether a molecule has a X-Y-Z pattern. If it does, it will be removed."""
        mol: Mol = Chem.MolFromSmiles(smiles)
        if not mol:  # Ensure the molecule is valid.
            raise ValueError("Invalid SMILES string.")

        # Define a generic SMARTS pattern for any three connected atoms
        pattern_single_aromatic = Chem.MolFromSmarts("*-*-*")  # * matches any atom, - matches single bonds
        matches = mol.GetSubstructMatches(pattern_single_aromatic, uniquify=True)

        if len(matches) == 0:
            return False
        return True

    @staticmethod
    def __check_single_carbon_bond_outside_ring(smiles: str) -> bool:  # pylint: disable=unused-private-member
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
