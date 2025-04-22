"""Filter that removes molecules which are not conjugated.
https://www.masterorganicchemistry.com/2017/01/24/conjugation-and-resonance/
Za "brak sprzężenia"  (oczywiście jest to pojęcie bardzo uproszczone) w tej strukturze związku odpowiadają atomy -S-C-C - i pojedyncze wiązania pomiędzy nimi. Należy zaznaczyć, że ten skrót myślowy - brak sprzężenia - dotyczy elektronów pi znajdujących się na orbitalach p w danym atomie, które tworzą wiązania wielokrotne pomiędzy atomami (podwójne i potrójne). Przykładowo w pierścieniu benzenowym, mamy wiązania wielokrotne podwójne, pomiędzy atomami C=C oraz C=N. Przykładowo, jedno wiązanie w C=C jest wiązaniem sigma, czyli para elektronów (1 elektron od 1 atomu C oraz 1elektron od 2 atomu C) tworzy wiązanie chemiczne pomiędzy atomami C-C. Ta para elektronów znajduje się na osi rdzeni atomowych atomów C-C, dlatego jest nazywane sigma. Są one silnie przyciągane przez rdzenie atomowe obu atomów, stad nie mogą one przemieszczać się pomiędzy atomami. Wszystkie wiązania pojedyncze w strukturach związków mają taki charakter. Czyli w odniesieniu do -S-C-C, wszystkie wiązania mają charakter sigma, elektrony nie przemieszczają się.
W wiązaniach wielokrotnych, jedno wiązanie ma zawsze charakter sigma, a pozostałe mają charakter pi. Czyli w C=C, 1 wiązanie to sigma a 2 wiązanie to pi. Wiązanie chemiczne pi, oznacza, ze tworzą to wiązanie elektrony pi znajdujące się na orbitalach p. Jeżeli orbitale p sa zorientowane w przestrzeni w taki sposób, ze nie leża w osi rdzeni atomowych, to są słabej przyciagne przez nie i mają większą swobodę ruchu w przestrzeni wokół rdzeni (po orbitalach atomowych). Orbital atomowy - w ujęciu matematycznych - jest to przestrzeń wokół jadra atomowego, która można opisać funkcja, największe prawdopodobieństwo ruchu elektronu wokół jadra atomowego. Orbital s przedstawiany jest jako sfera, a orbital p jako dwie pętle stykające się końcami.
Analizując strukturę związku chemicznego poniżej przedstawionego, w pierscieniu benzenowym, mamy w sumie 6 wiązań sigma (4 C-C oraz 2 C-N) i 3 wiązania pi. Jeżeli odległość pomiędzy wiązaniami pi jest mała (czyli maksymalnie jedno wiazanie pojedyncze - sigma) je rozdzielające, to elektrony pi mogą swobodnie przemieszczać się w obrębie danej przestrzeni. Czyli w obrębie pierścienia benzenowego, 6 elektronów pi może swobodnie przemieszczać się.
Dalej idąc w górę struktury, mamy 1 atom C z pierścienia benzenowego, który związany jest z kolejnym atomem C, który dalej związany jest z atomem N wiązaniem potrójnym (w wiązaniu potrójnym mamy 1 wiązanie sigma i 2 wiazania pi, utworzone przez 4 elektrony - te mogą swobodnie poruszać się). Zatem mamy układ C=C-C=(potrójne)N. Elektrony pi rozdzielone są tylko 1 wiązaniem sigma (pojedynczym), więc mogą swobodnie poruszać się wzdłuż C=C-C=(potrójne)N, czyli w uproszczeniu mówimy o "sprzężeniu". Czyli elektrony pi tworzące wiązanie C=C "sprzęgaja się" z C=(potrójne)N.
Wracając do pierwotnego pytania. Elektrony pi z pierścienia benzenowego nie mogą przemieścić się do C=(potrójne)N, ponieważ rozdziela je C-S-C-C, w związku z tym, nie ma "sprzężenia" pomiędzy pierścieniem benzenowym a C=(potrójne)N.
Siarkę można oznaczyć symbolicznie jako X, ponieważ istota tutaj jest charakter wiązania chemicznego pomiędzy atomami a nie rodzaj atomu.
"""
import itertools

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.features.filters.generic_filter import GenericMoleculeFilter


class ConjugationFilter(GenericMoleculeFilter):
    """Filter that removes molecules which are not conjugated."""

    def apply(self, molecules: list[Mol], **kwargs) -> list[Mol]:
        """
        Apply the filter to a list of RDKit molecules.

        Args:
            molecules (list[Mol]): The list of RDKit molecules to filter.
        Returns:
            list[Mol]: The list of RDKit molecules that passed the filter.
        """
        final_smiles = []
        for sml in molecules:
            if not self.check_conjugation(sml) or self.check_any_n_n_path(sml):
                final_smiles.append(sml)
        return final_smiles

    @staticmethod
    def check_conjugation(mol: Mol) -> bool:
        """Check whether a molecule is conjugated. If it does, it will be removed.

        Args:
            mol (Mol): The molecule to check.
        Returns:
            bool: True if the molecule is conjugated, False otherwise.
        """

        if mol is None:
            return False

        # Define a generic SMARTS pattern for any three connected atoms
        pattern_single_aromatic = Chem.MolFromSmarts("*-*-*")  # * matches any atom, - matches single bonds
        matches = mol.GetSubstructMatches(pattern_single_aromatic, uniquify=True)
        # for each match, check whether two consecutive bonds are conjugated
        num_conjugated = 0
        for match in matches:
            for i in range(len(match) - 2):
                bond1 = mol.GetBondBetweenAtoms(match[i], match[i + 1])
                bond2 = mol.GetBondBetweenAtoms(match[i + 1], match[i + 2])

                if bond1.GetIsConjugated() and bond2.GetIsConjugated():
                    num_conjugated += 1

        if len(matches) == num_conjugated or len(matches) == 0:
            return False  # All matches are conjugated or there are no matches
        return True  # At least one match is not conjugated

        # for each match, check whether two consecutive bonds are conjugated
        # num_conjugated = 0
        # for match in matches:
        #     for i in range(len(match) - 2):
        #         middle_atom = mol.GetAtomWithIdx(match[i + 1])
        #         symbol = middle_atom.GetSymbol()
        #         # get free electrons
        #         # maximum number of valence electrons, i.e. for oxygen it is 6
        #         max_valence = Chem.GetPeriodicTable().GetNOuterElecs(middle_atom.GetSymbol())
        #
        #         lone_electrons = max_valence - middle_atom.GetTotalValence()
        #         if lone_electrons > 0:
        #             num_conjugated += 1
        # if len(matches) == num_conjugated or len(matches) == 0:
        #     return False  # All matches are conjugated or there are no matches
        # return True  # At least one match is not conjugated

    def check_any_n_n_path(self, mol: Mol) -> bool:
        """Check whether there exists a path between two nitrogen atoms that has sprzężenie.

        Args:
            mol (Mol): The molecule to check.
        Returns:
            bool: True if there exists a path between two nitrogen atoms that has sprzężenie, False otherwise.
        """
        # Find all nitrogen atoms in the molecule
        smarts = "N#*"  # N atom with a triple bond to any atom
        pattern = Chem.MolFromSmarts(smarts)

        # Get all matches of the SMARTS pattern
        matches = mol.GetSubstructMatches(pattern, uniquify=True)

        # Check if there are at least two nitrogen atoms in the molecule
        if len(matches) < 2:
            return False

        nitrogens = [x if mol.GetAtomWithIdx(x).GetSymbol() == "N" else y for x, y in matches]

        # get all possible pairs of nitrogen atoms
        pairs = itertools.combinations(nitrogens, 2)

        # Get all paths between each pair of nitrogen atoms
        all_paths = []
        for pair in pairs:
            all_paths += self.get_all_paths_dfs(mol, pair[0], pair[1])

        # check if any path has an X-Y-Z pattern
        valid_paths = [path for path in all_paths if not self.has_two_single_bonds_in_path(mol, path)]

        return len(valid_paths) > 0

    def get_all_paths_dfs(self, mol: Chem.Mol, start_idx: int, end_idx: int) -> list[list[int]]:
        """
        Find all paths between two atoms in a molecule using DFS.

        Args:
            mol (rdkit.Chem.Mol): The molecule to search.
            start_idx (int): Index of the starting atom.
            end_idx (int): Index of the target atom.

        Returns:
            list of list: A list containing all possible paths, each as a list of atom indices.
        """
        adj_list = self.get_adjacency_list(mol)

        def dfs(current: int, target: int, visited: set, path: list[int]) -> None:
            """
            Recursive DFS function.

            Args:
                current (int): Current atom index.
                target (int): Target atom index.
                visited (set): Set of visited atom indices.
                path (list): Current path being explored.

            Returns:
                None. Appends valid paths to the result list.
            """
            path.append(current)
            visited.add(current)

            if current == target:
                paths.append(path[:])  # Save a copy of the current path
            else:
                for neighbor in adj_list[current]:
                    if neighbor not in visited:
                        dfs(neighbor, target, visited, path)

            path.pop()
            visited.remove(current)

        paths: list[list[int]] = []
        dfs(start_idx, end_idx, set(), [])
        return paths

    @staticmethod
    def has_two_single_bonds_in_path(mol: Chem.Mol, atom_path: list[int]) -> bool:
        """
        Check if there exists any triplet in the given atom path where both bonds are single.

        Args:
            mol (Chem.Mol): RDKit molecule object.
            atom_path (list[int]): A list of atom IDs representing a valid path in the molecule.

        Returns:
            bool: True if at least one triplet has two single bonds, False otherwise.
        """
        # Iterate over consecutive triplets in the path
        for i in range(len(atom_path) - 2):
            a, b, c = int(atom_path[i]), int(atom_path[i + 1]), int(atom_path[i + 2])

            # Get the bonds in the triplet
            bond1 = mol.GetBondBetweenAtoms(a, b)
            bond2 = mol.GetBondBetweenAtoms(b, c)

            # Check if both bonds are single
            if bond1.GetBondType() == Chem.rdchem.BondType.SINGLE and bond2.GetBondType() == Chem.rdchem.BondType.SINGLE:
                return True  # Found a triplet with two single bonds

        return False  # No triplet with two single bonds found

    @staticmethod
    def get_adjacency_list(mol: Chem.Mol) -> dict[int, list[int]]:
        """
        Generate an adjacency list for the molecule using RDKit's adjacency matrix.

        Args:
            mol (rdkit.Chem.Mol): The molecule to process.

        Returns:
            dict: A dictionary where keys are atom indices and values are lists of neighboring atom indices.
        """
        # Get the adjacency matrix as a NumPy array
        adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
        adjacency_list = {i: list(np.nonzero(adjacency_matrix[i])[0]) for i in range(len(adjacency_matrix))}
        return adjacency_list
