"""Module for filtering molecules based on symmetry properties."""
import logging

import networkx as nx
from networkx.algorithms.cycles import simple_cycles
from networkx.algorithms.isomorphism.ismags import ISMAGS

from modules.core.features.filters.generic_filter import GenericMoleculeFilter
from modules.core.features.symmetries import AvailableSymmetry
from modules.core.features.utils import get_graph_from_smile


class SymmetryFilter(GenericMoleculeFilter):
    """Check if any of the defined symmetries are present in the molecule."""

    def apply(self, smiles: list[str], **kwargs) -> list[str]:
        """
        Filters the given list of SMILES strings by symmetry. This filter transforms the SMILES strings into graphs first,
        then checks the presence of the defined symmetries in the graphs.

        Args:
            smiles (List[str]): A list of SMILES strings representing molecules.
            **kwargs: Additional keyword arguments.

        Returns:
            List[str]: A list of SMILES strings that pass the filter.
        """
        mol_graphs = [get_graph_from_smile(smiles) for smiles in smiles]
        symmetrical = []
        for i, mol in enumerate(mol_graphs):
            s = self._check_any_symmetry(mol)
            if s:
                symmetrical.append(smiles[i])
        return symmetrical

    def _check_any_symmetry(self, graph: nx.Graph) -> bool:
        """
        Check all symmetries available and return if any of them can be satisfied

        Parameters:
            - graph: nx.Graph

        Returns:
            - bool: is any symmetry present in the graph?
        """
        return any(self._check_symmetry(graph, symmetry_type=t) for t in AvailableSymmetry)

    @staticmethod
    def check_isomorphism(graph1: nx.Graph, graph2: nx.Graph) -> bool:
        """
        Check if two graphs are isomorphic
        """
        ismags = ISMAGS(graph1, graph2)

        return ismags.is_isomorphic()

    @staticmethod
    def find_all_cycles_of_len(graph: nx.Graph, length: int | None) -> list:
        """
        Find all cycles of a given length in a graph
        """
        # Find all cycles of length in the graph
        cycles = list(simple_cycles(graph))

        if length is None:
            cycles_of_len = list(cycles)
        else:
            cycles_of_len = [cycle for cycle in cycles if len(cycle) == length]

        logging.debug(f"All cycles of length {length}: {cycles_of_len}")

        return cycles_of_len

    @staticmethod
    def check_if_edge_is_part_of_cycle(edge: tuple, cycles: list | None) -> bool:
        """
        Check if an edge is a part of any cycle
        """
        if cycles is None:
            return False

        u, v = edge
        for cycle in cycles:
            if u in cycle and v in cycle:
                return True
        return False

    def _generate_candidate_set(  # pylint: disable=too-many-branches
        self,
        symmetry_type: AvailableSymmetry,
        graph: nx.Graph,
        dont_cut_on_cycles: bool = True,
        cycles_len: int = 6,
    ) -> set:
        """
        Generate a candidate set of substructures for symmetry analysis based on the given symmetry type.

        Args:
            symmetry_type (AvailableSymmetry): The type of symmetry to check.
                Must be one of the values in the `AvailableSymmetry` enum.
            graph (nx.Graph): The input graph to analyze.
            dont_cut_on_cycles (bool, optional): Whether to exclude nodes or edges that are part of cycles.
                Defaults to True.
            cycles_len (int, optional): The length of cycles to consider for symmetry types that require cycle analysis.
                Defaults to 6.

        Returns:
            Set: A set of candidate substructures depending on the selected symmetry type. The format of the candidates
            varies by type:
                - NODE: A set of node indices.
                - EDGE: A set of (u, v) tuples representing edges.
                - RING_BONDS: A set of pairs of edge tuples.
                - RING_NODES: A set of pairs of node indices.
                - RING_OUTER_PLANE: A set of pairs of neighboring nodes from rings.

        Raises:
            ValueError: If `cycles_len` is invalid for the chosen symmetry type.
        """
        # Initialize variables
        nodes = graph.nodes()
        blacklist: set[int] = set()
        cycles = None

        # Determine cycles and blacklist nodes if necessary
        if dont_cut_on_cycles:
            if symmetry_type == AvailableSymmetry.RING_OUTER_PLANE:
                cycles_len = None
            cycles = self.find_all_cycles_of_len(graph, cycles_len)
            if cycles:
                blacklist = {node for cycle in cycles for node in cycle}

        # Get nodes not in the blacklist
        nodes_not_in_blacklist = [node for node in nodes if node not in blacklist]

        # Generate candidates based on symmetry type
        if symmetry_type == AvailableSymmetry.NODE:
            candidates = set(nodes_not_in_blacklist)

        elif symmetry_type == AvailableSymmetry.EDGE:
            candidates = set()
            for u, v in graph.edges():
                # Avoid duplicates (u, v) and (v, u) and ensure the edge is not part of a cycle
                if (v, u) not in candidates and not self.check_if_edge_is_part_of_cycle((u, v), cycles):
                    candidates.add((u, v))

        elif symmetry_type == AvailableSymmetry.RING_BONDS:
            if cycles_len % 2 != 0:
                raise ValueError("cycles_len must be even; odd cycles are not supported for RING_BONDS symmetry.")

            candidates = set()
            for cycle in cycles:
                for i in range(len(cycle) // 2):
                    u1, u2 = cycle[i], cycle[(i + 1) % len(cycle)]
                    u1_second, u2_second = cycle[i + len(cycle) // 2], cycle[(i + 1 + len(cycle) // 2) % len(cycle)]
                    candidate = ((u1, u2), (u1_second, u2_second))
                    if candidate not in candidates:
                        candidates.add(candidate)

        elif symmetry_type == AvailableSymmetry.RING_NODES:
            if cycles_len % 2 != 0:
                raise ValueError("cycles_len must be even; odd cycles are not supported for RING_NODES symmetry.")

            candidates = set()
            for cycle in cycles:
                for i in range(len(cycle) // 2):
                    u, u_second = cycle[i], cycle[i + len(cycle) // 2]
                    if (u_second, u) not in candidates:
                        candidates.add((u, u_second))

        elif symmetry_type == AvailableSymmetry.RING_OUTER_PLANE:
            candidates = set()
            for cycle in cycles:
                for i, u in enumerate(cycle):
                    u_second = cycle[(i + 1) % len(cycle)]
                    if (u_second, u) not in candidates:
                        candidates.add((u, u_second))

        else:
            raise ValueError(f"Unsupported symmetry type: {symmetry_type}")

        return candidates

    def _check_symmetry(
        self,
        graph: nx.Graph,
        symmetry_type: AvailableSymmetry = AvailableSymmetry.NODE,
        cycles_len: int = 6,
        dont_cut_on_cycles: bool = True,
    ) -> bool:
        """
        Check if cutting on any candidate element (node, edge, or ring structure) results in two isomorphic subgraphs.

        Args:
            graph (nx.Graph): The input graph to analyze.
            symmetry_type (AvailableSymmetry, optional): The type of symmetry to check.
                Must be one of the values in the `AvailableSymmetry` enum. Defaults to AvailableSymmetry.NODE.
            cycles_len (int, optional): Length of cycles to consider for ring-based symmetry types. Defaults to 6.
            dont_cut_on_cycles (bool, optional): Whether to avoid cutting nodes or edges that are part of cycles.
                Defaults to True.

        Returns:
            bool: True if any cut produces two isomorphic subgraphs, False otherwise.

        Raises:
            ValueError: If an invalid symmetry type is provided.
        """
        # Generate the candidate set
        candidates = self._generate_candidate_set(symmetry_type, graph, dont_cut_on_cycles, cycles_len)

        for obj in candidates:
            # Create a copy of the graph
            graph_copy = graph.copy()

            # Handle node and edge removal based on symmetry type
            if symmetry_type == AvailableSymmetry.NODE:
                node = obj
                graph_copy.remove_node(node)

            elif symmetry_type == AvailableSymmetry.EDGE:
                u, v = obj
                graph_copy.remove_edge(u, v)

            elif symmetry_type == AvailableSymmetry.RING_BONDS:
                ((u1, u2), (u1_second, u2_second)) = obj
                graph_copy.remove_edge(u1, u2)
                graph_copy.remove_edge(u1_second, u2_second)

            elif symmetry_type == AvailableSymmetry.RING_NODES:
                (u, u_second) = obj
                graph_copy.remove_node(u)
                graph_copy.remove_node(u_second)

            elif symmetry_type == AvailableSymmetry.RING_OUTER_PLANE:
                (u, u_second) = obj
                graph_copy.remove_node(u)
                graph_copy.remove_node(u_second)

            else:
                raise ValueError(f"Unsupported symmetry type: {symmetry_type}")

            # Analyze the graph after the cut
            if nx.is_connected(graph_copy):
                logging.debug(f"Cut on {symmetry_type.value} {obj} results in a connected graph")
            else:
                components = list(nx.connected_components(graph_copy))
                logging.debug(f"Cut on {symmetry_type.value} {obj} results in {len(components)} components")

                # Get the two largest components
                components = sorted(components, key=len, reverse=True)[:2]

                # Create subgraphs from the components
                subgraph1 = graph.subgraph(components[0])
                subgraph2 = graph.subgraph(components[1])

                # Check if the two subgraphs are isomorphic
                is_isomorphic = self.check_isomorphism(subgraph1, subgraph2)
                if is_isomorphic:
                    logging.debug("Two subgraphs are isomorphic")
                    return True
                logging.debug("Two subgraphs are not isomorphic")

        return False
