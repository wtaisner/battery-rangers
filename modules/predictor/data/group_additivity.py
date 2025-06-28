"""Functions for additive group generation."""
import os

import numpy as np
import pandas as pd
from rdkit import Chem


def _smiles_to_graphs(molecules: list) -> list:
    """
    Transforms smiles to graph representation.
    :param molecules: list of smiles.
    :return: list of adjacency matrices.
    """
    m_graphs = []
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        elements = [atom.GetSymbol() + "_" + str(atom.GetIdx()) for atom in mol.GetAtoms()]
        adjacency_matrix = Chem.GetAdjacencyMatrix(mol, useBO=True)
        m_graph = pd.DataFrame(adjacency_matrix, index=elements, columns=elements)
        m_graphs.append(m_graph)
    return m_graphs


def graph_to_groups(graph: pd.DataFrame, permit_groups: list | None = None) -> tuple[dict, dict]:
    """
    Transforms a graph (adjacency matrix) to additive groups.
    :param graph: adjacency matrix.
    :param permit_groups: permitted groups, if not given extracts all groups from the graph
    :return: Dictionary with group counts, dictionary with atoms and their groups.
    """
    groups = {}
    groups_rows = graph[graph.astype(bool).sum(axis=1) >= 2]
    atoms = {atom: [] for atom in graph.index}
    for row in groups_rows.iterrows():
        central_atom = row[0]
        neighbors = []
        nonzero_neighbors = row[1][row[1] > 0]
        neighbors_ids, neighbors_counts = nonzero_neighbors.index, nonzero_neighbors.values
        for n_id, n_count in zip(neighbors_ids, neighbors_counts):
            n_atom, n_id = n_id.split("_")
            neighbors.append((str(int(n_count)) + str(n_atom)))
        neighbors.sort()
        group = central_atom.split("_")[0] + "[" + ",".join(neighbors) + "]"
        if permit_groups is not None and group not in permit_groups:
            continue
        atoms[central_atom].append(group)
        for atom in neighbors_ids:
            atoms[atom].append(group)
        if group in groups:
            groups[group] += 1
        else:
            groups[group] = 1
    zero_atoms = [atom for atom, count in atoms.items() if len(count) == 0]
    if len(zero_atoms) > 0:
        raise ValueError(f"Zero count atoms: {zero_atoms}")
    return groups, atoms


class AdditiveGroups:
    """
    Class for generating additive groups.
    """

    def __init__(self, molecules: list, groups: dict | None = None):
        """
        :param molecules: list of training smiles.
        :param groups: list of permitted groups (optional).
        """
        self.molecules = molecules
        self.groups = None
        self.groups, self.smiles_groups, self.coverage = self.generate_groups(self.molecules, groups)
        print(len(self.groups))
        new_groups = self.reduce_groups()
        self.groups, self.smiles_groups, self.coverage = self.generate_groups(self.molecules, new_groups)
        print(len(self.groups))

    def generate_groups(self, molecules: list, permit_groups: list | None = None) -> tuple:
        """
        Generates additive groups from the given molecules.
        :param molecules: list of molecules (smiles).
        :param permit_groups: list of permitted groups.
        :return: dictionary with groups and their count, dataframe with groups for each molecule, dictionary with groups and atoms belonging to them.
        """
        groups = {}
        coverage = {}
        groups_matrix = []
        graphs = _smiles_to_graphs(molecules)
        if self.groups is not None and permit_groups is None:
            permit_groups = list(self.groups.keys())
        for i, graph in enumerate(graphs):
            graph_groups, atoms = graph_to_groups(graph, permit_groups)

            for key, values in atoms.items():
                for value in values:
                    if value not in coverage:
                        coverage[value] = []
                    coverage[value].append(str(i) + "_" + key)

            for group, count in graph_groups.items():
                if group in groups:
                    groups[group] += count
                else:
                    groups[group] = count
            groups_matrix.append({"smiles": molecules[i], **graph_groups})
        groups_matrix = pd.DataFrame(groups_matrix)
        groups_matrix.fillna(0, inplace=True)
        return groups, groups_matrix, coverage

    def reduce_groups(self) -> list:
        """
        Reduces the number of groups by selecting the most informative ones (set of groups that covers all atoms using greedy algorithm for set coverage problem).
        :return: selected groups.
        """
        all_atoms = len(set(atom for atoms in self.coverage.values() for atom in atoms))
        atoms_covered = set()
        costs = {group: (len(self.molecules) - value) / len(self.molecules) for group, value in self.groups.items()}
        all_groups = list(self.groups.keys())
        selected_groups = []
        while len(atoms_covered) < all_atoms:
            best_score, best_group = np.inf, None
            for group in all_groups:
                newly_added_elements = len(set(atom for atom in self.coverage[group] if atom not in atoms_covered))
                score = costs[group] / newly_added_elements if newly_added_elements > 0 else np.inf
                if score < best_score:
                    best_score, best_group = score, group
            selected_groups.append(best_group)
            atoms_covered.update(set(self.coverage[best_group]))
            all_groups.remove(best_group)
        return selected_groups


if __name__ == "__main__":
    DIR_PATH = "../../../data/old/processed_selected_custom_features/"
    SAVE_PATH = "../../../data/old/additive_groups/"
    os.makedirs(SAVE_PATH, exist_ok=True)
    data_names = ["data_experts1.csv", "data_experts2.csv", "data_saad.csv", "data_zhu.csv"]
    TARGET = "capacity_max"
    datasets = [DIR_PATH + dataset for dataset in data_names]
    dfs = [pd.read_csv(dataset) for dataset in datasets]
    df = pd.concat(dfs)
    molecules = df["smiles"].tolist()

    ag = AdditiveGroups(molecules)

    for df_name, df in zip(data_names, dfs):
        _, df_groups, _ = ag.generate_groups(df["smiles"].tolist())
        df_groups[TARGET] = df[TARGET]
        df_groups.to_csv(os.path.join(SAVE_PATH, df_name), index=False)
        print(f"Groups for {df_name} dataset saved.")
