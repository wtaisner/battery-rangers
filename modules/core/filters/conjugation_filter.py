"""A filter that removes non-conjugated molecules.

https://www.masterorganicchemistry.com/2017/01/24/conjugation-and-resonance/

Za "brak sprzężenia"  (oczywiście jest to pojęcie bardzo uproszczone) w tej strukturze
związku odpowiadają atomy -S-C-C - i pojedyncze wiązania pomiędzy nimi. Należy zaznaczyć,
że ten skrót myślowy - brak sprzężenia - dotyczy elektronów pi znajdujących się na
orbitalach p w danym atomie, które tworzą wiązania wielokrotne pomiędzy atomami (podwójne
i potrójne). Przykładowo w pierścieniu benzenowym, mamy wiązania wielokrotne podwójne,
pomiędzy atomami C=C oraz C=N. Przykładowo, jedno wiązanie w C=C jest wiązaniem sigma,
czyli para elektronów (1 elektron od 1 atomu C oraz 1elektron od 2 atomu C) tworzy
wiązanie chemiczne pomiędzy atomami C-C. Ta para elektronów znajduje się na osi rdzeni
atomowych atomów C-C, dlatego jest nazywane sigma. Są one silnie przyciągane przez
rdzenie atomowe obu atomów, stad nie mogą one przemieszczać się pomiędzy atomami.
Wszystkie wiązania pojedyncze w strukturach związków mają taki character. Czyli w
odniesieniu do -S-C-C, wszystkie wiązania mają character sigma, elektrony nie
przemieszczają się.

W wiązaniach wielokrotnych, jedno wiązanie ma zawsze character sigma, a pozostałe mają
character pi. Czyli w C=C, 1 wiązanie to sigma a 2 wiązanie to pi. Wiązanie chemiczne
pi, oznacza, ze tworzą to wiązanie elektrony pi znajdujące się na orbitalach p. Jeżeli
orbitale p sa zorientowane w przestrzeni w taki sposób, ze nie leża w osi rdzeni
atomowych, to są słabej przyciagne przez nie i mają większą swobodę ruchu w przestrzeni
wokół rdzeni (po orbitalach atomowych). Orbital atomowy - w ujęciu matematycznych - jest
to przestrzeń wokół jadra atomowego, która można opisać funkcja, największe
prawdopodobieństwo ruchu elektronu wokół jadra atomowego. Orbital s przedstawiany jest
jako sfera, a orbital p jako dwie pętle stykające się końcami.

Analizując strukturę związku chemicznego poniżej przedstawionego, w pierscieniu
benzenowym, mamy w sumie 6 wiązań sigma (4 C-C oraz 2 C-N) i 3 wiązania pi. Jeżeli
odległość pomiędzy wiązaniami pi jest mała (czyli maksymalnie jedno wiazanie pojedyncze
- sigma) je rozdzielające, to elektrony pi mogą swobodnie przemieszczać się w obrębie
danej przestrzeni. Czyli w obrębie pierścienia benzenowego, 6 elektronów pi może
swobodnie przemieszczać się.

Dalej idąc w górę struktury, mamy 1 atom C z pierścienia benzenowego, który związany jest
z kolejnym atomem C, który dalej związany jest z atomem N wiązaniem potrójnym (w
wiązaniu potrójnym mamy 1 wiązanie sigma i 2 wiazania pi, utworzone przez 4 elektrony -
te mogą swobodnie poruszać się). Zatem mamy układ C=C-C=(potrójne)N. Elektrony pi
rozdzielone są tylko 1 wiązaniem sigma (pojedynczym), więc mogą swobodnie poruszać się
wzdłuż C=C-C=(potrójne)N, czyli w uproszczeniu mówimy o "sprzężeniu". Czyli elektrony pi
tworzące wiązanie C=C "sprzęgaja się" z C=(potrójne)N.

Wracając do pierwotnego pytania. Elektrony pi z pierścienia benzenowego nie mogą
przemieścić się do C=(potrójne)N, ponieważ rozdziela je C-S-C-C, w związku z tym, nie
ma "sprzężenia" pomiędzy pierścieniem benzenowym a C=(potrójne)N.

Siarkę można oznaczyć symbolicznie jako X, ponieważ istota tutaj jest character
wiązania chemicznego pomiędzy atomami a nie rodzaj atomu.
"""

import itertools
from collections import deque

from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.filters.generic_filter import GenericMoleculeFilter


class ConjugationFilter(GenericMoleculeFilter):
    """Filters molecules based on a conditional, multi-criteria approach to conjugation.

    The filter applies a hierarchical logic:

    1.  **For molecules with 2 or more nitrogen atoms:** The primary criterion is
        electronic connectivity. The molecule passes **if and only if** all nitrogen
        atoms are mutually connected through a fully conjugated path. The presence of
        other non-conjugated groups is ignored.

    2.  **For molecules with 0 or 1 nitrogen atoms:** The N-N connectivity rule is
        not applicable. The molecule is judged on its overall structural integrity.
        It passes **if and only if** it does not contain any "conjugation-breaking"
        linkers (e.g., a C-C-C alkyl chain).
    """

    def apply(self, molecules: list[Mol], **kwargs) -> list[Mol]:
        """Applies the conjugation filter to a list of RDKit molecules.

        Args:
            molecules (list[Mol]): The list of RDKit molecules to be filtered.
            **kwargs: Additional keyword arguments (for API compatibility, unused).

        Returns:
            list[Mol]: A new list containing only the molecules that passed the
                       conjugation criteria.

        """
        passed_molecules: list[Mol] = []
        for mol in molecules:
            if mol is None:
                continue

            # First, determine which rule set applies based on nitrogen count.
            nitrogen_indices: list[int] = [match[0] for match in mol.GetSubstructMatches(Chem.MolFromSmarts("[N]"))]

            # --- CONDITIONAL LOGIC ---
            if len(nitrogen_indices) >= 2:
                # Rule set 1: For molecules with multiple nitrogens, only N-N path matters.
                if self._are_all_nitrogens_connected(mol, nitrogen_indices):
                    passed_molecules.append(mol)
            # Rule set 2: For molecules with < 2 nitrogens, only structural integrity matters.
            elif not self._contains_conjugation_break(mol):
                passed_molecules.append(mol)

        return passed_molecules

    @staticmethod
    def _contains_conjugation_break(mol: Mol | None) -> bool:
        """Checks if the molecule contains a definitive break in conjugation.
        Returns True if a break is found, False otherwise.
        """
        if mol is None:
            return True

        pattern = Chem.MolFromSmarts("[!#1]~[!#1]~[!#1]")
        for match in mol.GetSubstructMatches(pattern, uniquify=True):
            atom1_idx, atom2_idx, atom3_idx = match
            bond1 = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
            bond2 = mol.GetBondBetweenAtoms(atom2_idx, atom3_idx)
            if bond1.GetBondType() == Chem.rdchem.BondType.SINGLE and bond2.GetBondType() == Chem.rdchem.BondType.SINGLE and not bond1.GetIsConjugated() and not bond2.GetIsConjugated():
                return True
        return False

    @staticmethod
    def _are_all_nitrogens_connected(mol: Mol, nitrogen_indices: list[int]) -> bool:
        """Checks if a fully conjugated path exists between every pair of nitrogen atoms.
        This method assumes it is only called when len(nitrogen_indices) >= 2.
        """
        # No need to check mol is None or len, as that's handled in apply()
        nitrogen_pairs = itertools.combinations(nitrogen_indices, 2)
        for start_idx, end_idx in nitrogen_pairs:
            if not ConjugationFilter._find_conjugated_path_bfs(mol, start_idx, end_idx):
                return False  # A non-connected pair was found.
        return True  # All pairs were successfully connected.

    @staticmethod
    def _find_conjugated_path_bfs(mol: Mol, start_idx: int, end_idx: int) -> bool:
        """Efficiently finds if a conjugated path exists between two atoms using BFS."""
        queue: deque[int] = deque([start_idx])
        visited: set[int] = {start_idx}
        while queue:
            current_idx = queue.popleft()
            if current_idx == end_idx:
                return True
            current_atom = mol.GetAtomWithIdx(current_idx)
            for neighbor in current_atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    bond = mol.GetBondBetweenAtoms(current_idx, neighbor_idx)
                    if bond.GetIsConjugated():
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)
        return False

    def filter_from_property(self, properties: dict) -> bool:
        """Reads properties from a dictionary (database) and decides whether to filter the molecule."""
        return properties.get("conjugation_filter", False)
