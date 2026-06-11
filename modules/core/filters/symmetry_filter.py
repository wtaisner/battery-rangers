"""Module for filtering molecules based on symmetry properties."""

import logging
from collections import defaultdict
from typing import Any

import networkx as nx
from networkx import faster_could_be_isomorphic
from networkx.algorithms.cycles import simple_cycles
from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.features.symmetries import AvailableSymmetry
from modules.core.features.utils import mol2graph
from modules.core.filters.generic_filter import GenericMoleculeFilter
from modules.core.graph_visualization import plot_nx_graphs

logger = logging.getLogger(__name__)  # __name__ ensures the logger is specific to this module
logger.setLevel(logging.INFO)


class SymmetryFilter(GenericMoleculeFilter):
    """Check if any of the defined symmetries are present in the molecule."""

    def __init__(
        self,
        min_isomorphic_nodes: int = 8,
        types_to_check: list[AvailableSymmetry] | None = None,
    ):
        """Args:
        min_isomorphic_nodes (int): The minimum number of nodes an isomorphic
            component must have to be counted/considered a valid symmetry.
            Defaults to 3 (ignores single atoms and diatomic fragments).
        types_to_check (list[AvailableSymmetry]): List of symmetry types to check. If None, defaults to AvailableSymmetry.WHOLE_RING, EDGE, NODE and RING_NODES

        """
        self.min_isomorphic_nodes = min_isomorphic_nodes
        if types_to_check is None:
            self.types_to_check = [
                AvailableSymmetry.WHOLE_RING,
                AvailableSymmetry.EDGE,
                AvailableSymmetry.RING_NODES,
                AvailableSymmetry.NODE,
            ]
        else:
            self.types_to_check = types_to_check

    def apply(self, molecules: list[Mol], **kwargs) -> list[Mol] | tuple[list[Mol], list[dict]]:  # type: ignore[override]
        """Filters molecules based on specified symmetry types and threshold.
        Optionally returns detailed branch counts per molecule.

        Args:
            molecules (list[Mol]): A list of RDKit Mol objects to filter.
            **kwargs: Additional keyword arguments. Supports:
                plot_symmetry_checks (bool): Generate plots for disconnected cuts. (Default: False)
                return_branch_counts (bool): If True, return tuple with symmetric mols
                                             and list of branch count dicts. (Default: False)

        Returns:
            Union[list[Mol], tuple[list[Mol], list[dict]]]:
                - If return_branch_counts is False (default): List of symmetric Mol objects.
                - If return_branch_counts is True: Tuple containing:
                    - List of symmetric Mol objects.
                    - List of dictionaries, one for each input molecule, detailing the
                      counts of distinct isomorphic branches found above the threshold.

        """
        mol_graphs: list[tuple[nx.Graph, Mol]] = []
        # Basic conversion and filtering of empty graphs
        for i, mol in enumerate(molecules):
            try:
                graph = mol2graph(mol)
                if graph.number_of_nodes() > 0:
                    mol_graphs.append((graph, mol))
                else:
                    logger.debug("Mol %d -> empty graph. Skipping.", i)
            except ValueError as e:
                logger.error("Graph conversion failed for Mol %d: %s. Skipping.", i, e)
                continue

        symmetrical_molecules_list: list[Mol] = []
        molecule_branch_summaries: list[dict] = []  # To store summary for each input mol

        plot_checks: bool = kwargs.get("plot_symmetry_checks", False)
        return_branch_counts: bool = kwargs.get("return_branch_counts", False)

        for i, (mol_graph, original_mol) in enumerate(mol_graphs):
            mol_identifier = f"Mol {i + 1}"

            # Stores ONE representative subgraph for each distinct isomorphic shape
            distinct_isomorphic_representatives: list[nx.Graph] = []
            # Stores details for EVERY VALID symmetric cut found
            all_valid_symmetric_origins: list[tuple[int, AvailableSymmetry, Any, int, int]] = []

            if not self.types_to_check:
                logger.warning(
                    "No symmetry types configured for %s. Skipping analysis.",
                    mol_identifier,
                )
                molecule_branch_summaries.append({})  # Add empty summary for this mol
                continue

            for symmetry_type_enum in self.types_to_check:
                try:
                    symmetrical = self._check_symmetry(
                        mol_graph,
                        symmetry_type=symmetry_type_enum,
                        distinct_rep_list=distinct_isomorphic_representatives,
                        all_origins_details_list=all_valid_symmetric_origins,
                        plot_checks=plot_checks,
                        mol_identifier=mol_identifier,
                    )
                    if symmetrical:
                        break
                except ValueError as e:
                    logger.error(
                        "Error in _check_symmetry for %s, type %s: %s",
                        mol_identifier,
                        symmetry_type_enum.name,
                        e,
                        exc_info=True,
                    )

            # --- Summarize and Store results for the current molecule ---
            molecule_summary = {}  # Summary dict for *this* molecule
            if not all_valid_symmetric_origins:
                logger.debug(
                    "Result for %s: No valid symmetric cuts found (above threshold).",
                    mol_identifier,
                )
            else:
                _, sym_type_enum, cut_obj, num_in_group, total_comps = all_valid_symmetric_origins[0]

                # Prepare summary dict for return/storage
                molecule_summary = {
                    "symmetry_type": sym_type_enum.name,
                    "cut_object": cut_obj,
                    "num_in_group": num_in_group,
                    "total_components": total_comps,
                }

                symmetrical_molecules_list.append(original_mol)

            molecule_branch_summaries.append(molecule_summary)

        # --- Return based on flag ---
        if return_branch_counts:
            return symmetrical_molecules_list, molecule_branch_summaries
        return symmetrical_molecules_list

    def _find_or_add_distinct_representative(self, subgraph_to_check: nx.Graph, distinct_rep_list: list[nx.Graph]) -> int:
        """Checks if subgraph_to_check is isomorphic to any graph in distinct_rep_list.

        Args:
            subgraph_to_check (nx.Graph): The subgraph to check for isomorphism.
            distinct_rep_list (list[nx.Graph]): List of distinct representative graphs.

        Returns:
            int: Index of the distinct representative graph in the list.
                    If yes, returns the index of the first match.
                    If no, appends subgraph_to_check to the list and returns its new index.

        """
        for i, stored_graph in enumerate(distinct_rep_list):
            if self.check_isomorphism(subgraph_to_check, stored_graph):
                return i  # Found match at index i
        # No match found, add it as a new distinct representative
        distinct_rep_list.append(subgraph_to_check)
        return len(distinct_rep_list) - 1  # Return its new index

    def check_isomorphism(self, graph1: nx.Graph, graph2: nx.Graph) -> bool:
        """Check whether two graphs are isomorphic using networkx GraphMatcher.

        Args:
            graph1 (nx.Graph): The first graph to compare.
            graph2 (nx.Graph): The second graph to compare.

        Returns:
            bool: True if the graphs are isomorphic, False otherwise.

        """
        if graph1.number_of_nodes() != graph2.number_of_nodes():
            return False
        if graph1.number_of_edges() != graph2.number_of_edges():
            return False

        if graph1.number_of_nodes() < self.min_isomorphic_nodes or graph2.number_of_nodes() < self.min_isomorphic_nodes:
            return False

        deg_seq1 = sorted([d for n, d in graph1.degree()])
        deg_seq2 = sorted([d for n, d in graph2.degree()])
        if deg_seq1 != deg_seq2:
            return False

        try:
            # gm = GraphMatcher(graph1, graph2)
            # return gm.is_isomorphic()
            return faster_could_be_isomorphic(graph1, graph2)
        except ValueError as e:
            logger.error(f"Isomorphism check failed: {e}")
            return False  # Treat error as non-isomorphic

    @staticmethod
    def find_all_cycles_of_len(graph: nx.Graph, length: int = 6) -> list:
        """Find all cycles of a given length in a graph.

        Args:
            graph (nx.Graph): The input graph.
            length (int): The length of cycles to find. Default is 6.

        Returns:
            list: A list of cycles found in the graph.

        """
        # Find all cycles of length in the graph
        cycles = list(simple_cycles(graph))

        if length is None:
            cycles_of_len = list(cycles)
        else:
            cycles_of_len = [cycle for cycle in cycles if len(cycle) == length]

        return cycles_of_len

    def _generate_candidate_set(
        self,
        symmetry_type: AvailableSymmetry,  # Argument is now Enum member
        graph: nx.Graph,
    ) -> set:
        """Generate candidates based on the Enum member.

        For each symmetry type, generate a set of candidates to check for symmetry.

        Args:
            symmetry_type (AvailableSymmetry): The type of symmetry to check for.
            graph (nx.Graph): The input graph.

        Returns:
            set: A set of candidates for the specified symmetry type.

        """
        nodes = set(graph.nodes())
        edges = set(graph.edges())
        candidates = set()

        if symmetry_type == AvailableSymmetry.NODE:
            candidates = nodes
        elif symmetry_type == AvailableSymmetry.EDGE:
            for u, v in edges:
                candidates.add(tuple(sorted((u, v))))
        elif symmetry_type in (
            AvailableSymmetry.RING_NODES,
            AvailableSymmetry.WHOLE_RING,
        ):
            all_cycles = self.find_all_cycles_of_len(graph)
            if symmetry_type == AvailableSymmetry.RING_NODES:
                even_cycles = [c for c in all_cycles if len(c) > 3 and len(c) % 2 == 0]
                for cycle in even_cycles:
                    half_len = len(cycle) // 2
                    for i in range(half_len):
                        candidates.add(tuple(sorted((cycle[i], cycle[i + half_len]))))
            elif symmetry_type == AvailableSymmetry.WHOLE_RING:
                candidates = {frozenset(cycle) for cycle in all_cycles if len(cycle) >= 3}
        else:
            raise ValueError(f"Unsupported symmetry type enum member: {symmetry_type}")
        return candidates

    def _check_symmetry(
        self,
        graph: nx.Graph,
        symmetry_type: AvailableSymmetry,
        distinct_rep_list: list[nx.Graph],
        all_origins_details_list: list[tuple[int, AvailableSymmetry, Any, int, int]],
        plot_checks: bool = False,
        mol_identifier: str = "",
    ) -> bool:
        """Checks cuts for a symmetry type. If a valid isomorphic group (above
        threshold) is found, records details, optionally plots, and returns True.

        Args:
            graph: The input graph.
            symmetry_type: The symmetry type to check.
            distinct_rep_list: List storing unique graph shapes found so far.
            all_origins_details_list: List to store details of valid symmetries.
            plot_checks: Flag to enable plotting.
            mol_identifier: Identifier for logging/plotting.

        Returns:
            True if a valid symmetry was found for this type, False otherwise.

        """
        try:
            candidates = self._generate_candidate_set(symmetry_type, graph)
        except ValueError as e:
            logger.error("Cannot generate candidates for %s: %s", symmetry_type.name, e)
            return False
        if not candidates or graph.number_of_nodes() == 0:
            logger.debug("No candidates or empty graph for %s.", symmetry_type.name)
            return False

        pos_original = self._get_layout(graph, plot_checks, mol_identifier, symmetry_type)

        for obj in candidates:
            cut_info = self._perform_cut(graph, symmetry_type, obj)
            if not cut_info:  # Cut failed or element didn't exist
                continue

            graph_copy = cut_info["graph_copy"]
            nodes_cut = cut_info["nodes_cut"]
            edges_cut = cut_info["edges_cut"]

            # check if the graph is null
            if graph_copy.number_of_nodes() == 0:
                return False

            if not nx.is_connected(graph_copy):
                analysis_result = self._analyze_components(graph, graph_copy, distinct_rep_list)

                if analysis_result["valid_iso_group_found"]:
                    # --- VALID SYMMETRY FOUND ---
                    self._record_valid_symmetry(all_origins_details_list, analysis_result, symmetry_type, obj)
                    if plot_checks and pos_original is not None:
                        self._plot_symmetry_cut(
                            graph=graph,
                            pos_original=pos_original,
                            symmetry_type=symmetry_type,
                            cut_obj=obj,
                            nodes_cut=nodes_cut,
                            edges_cut=edges_cut,
                            analysis_result=analysis_result,
                            mol_identifier=mol_identifier,
                            plot_suffix="VALID Isomorphic Group(s)",  # Indicate success
                        )
                    return True

                if plot_checks and pos_original is not None:
                    # Plot disconnected cuts even if not validly symmetric
                    plot_suffix = "Isomorphic Group(s) Below Threshold" if analysis_result["any_iso_group_found"] else "No Isomorphic Groups Found"
                    self._plot_symmetry_cut(
                        graph=graph,
                        pos_original=pos_original,
                        symmetry_type=symmetry_type,
                        cut_obj=obj,
                        nodes_cut=nodes_cut,
                        edges_cut=edges_cut,
                        analysis_result=analysis_result,
                        mol_identifier=mol_identifier,
                        plot_suffix=plot_suffix,
                    )

        return False

    def _get_layout(
        self,
        graph: nx.Graph,
        plot_checks: bool,
        mol_identifier: str,
        symmetry_type: AvailableSymmetry,
    ) -> dict[Any, tuple[float, float]] | None:
        """Calculates graph layout if plotting is enabled and possible."""
        pos_original = None
        if plot_checks:
            try:
                if graph.number_of_nodes() > 0:
                    pos_original = nx.kamada_kawai_layout(graph)
                else:
                    pos_original = {}  # Empty layout for empty graph
            except ValueError as e:
                logger.warning(
                    "Layout failed for %s (%s): %s. Disabling plots for this check.",
                    mol_identifier,
                    symmetry_type.name,
                    e,
                )
        return pos_original

    # pylint: disable=too-many-return-statements, too-many-branches
    def _perform_cut(self, graph: nx.Graph, symmetry_type: AvailableSymmetry, obj: Any) -> dict | None:
        """Performs the cut operation on a copy of the graph based on symmetry type.

        Returns:
            A dictionary containing {'graph_copy', 'nodes_cut', 'edges_cut'} if successful,
            None otherwise.

        """
        graph_copy = graph.copy()
        nodes_cut: set[Any] = set()
        edges_cut: set[frozenset] = set()  # Store edges as frozensets

        try:
            if symmetry_type == AvailableSymmetry.NODE:
                if obj not in graph_copy:
                    return None  # Node doesn't exist
                graph_copy.remove_node(obj)
                nodes_cut.add(obj)
            elif symmetry_type == AvailableSymmetry.EDGE:
                u, v = obj  # Assumes obj is a sorted tuple
                if not graph_copy.has_edge(u, v):
                    return None  # Edge doesn't exist
                graph_copy.remove_edge(u, v)
                edges_cut.add(frozenset({u, v}))
            elif symmetry_type == AvailableSymmetry.RING_NODES:
                u, v = obj  # Assumes obj is a sorted tuple
                if u not in graph_copy or v not in graph_copy:
                    return None  # Node(s) don't exist
                graph_copy.remove_node(u)
                graph_copy.remove_node(v)
                nodes_cut.add(u)
                nodes_cut.add(v)
            elif symmetry_type == AvailableSymmetry.WHOLE_RING:
                ring_nodes = obj  # Assumes obj is a frozenset
                if not ring_nodes.issubset(graph_copy.nodes()):
                    return None  # Node(s) don't exist
                graph_copy.remove_nodes_from(ring_nodes)
                nodes_cut.update(ring_nodes)
                # Identify internal edges from original graph
                for u_r in ring_nodes:
                    for v_r in ring_nodes:
                        if u_r < v_r and graph.has_edge(u_r, v_r):
                            edges_cut.add(frozenset({u_r, v_r}))
            else:
                logger.error("Unhandled symmetry type in _perform_cut: %s", symmetry_type)
                return None  # Unknown type

            return {
                "graph_copy": graph_copy,
                "nodes_cut": nodes_cut,
                "edges_cut": edges_cut,
            }

        except (nx.NetworkXError, KeyError) as e:
            logger.debug(
                "Skipping candidate %s (%s): cannot modify graph. %s",
                obj,
                symmetry_type.name,
                e,
            )
            return None

    def _analyze_components(
        self,
        original_graph: nx.Graph,
        graph_after_cut: nx.Graph,
        distinct_rep_list: list[nx.Graph],
    ) -> dict[str, Any]:
        """Analyzes components of the graph after a cut, checks for isomorphism,
        and applies the node threshold.

        Returns:
            A dictionary containing analysis results:
            'all_components_nodes': List of node sets for all components (sorted by size).
            'total_components_found': Total number of components.
            'isomorphism_groups': Dict mapping rep_idx to list of component indices.
            'any_iso_group_found': Boolean indicating if any group size > 1 exists.
            'valid_iso_group_found': Boolean indicating if any group size > 1 meets threshold.
            'valid_iso_details': List of tuples for valid groups: [(rep_idx, num_in_group), ...]

        """
        all_components_nodes = sorted(list(nx.connected_components(graph_after_cut)), key=len, reverse=True)
        total_components_found = len(all_components_nodes)
        result = {
            "all_components_nodes": all_components_nodes,
            "total_components_found": total_components_found,
            "isomorphism_groups": {},
            "any_iso_group_found": False,
            "valid_iso_group_found": False,
            "valid_iso_details": [],  # Store details of groups meeting threshold
        }

        if total_components_found < 2:
            return result  # Nothing to compare

        component_subgraphs = [original_graph.subgraph(comp_nodes).copy() for comp_nodes in all_components_nodes]
        isomorphism_groups = defaultdict(list)  # Use defaultdict for easy grouping

        for comp_idx, subg in enumerate(component_subgraphs):
            rep_idx = self._find_or_add_distinct_representative(subg, distinct_rep_list)
            isomorphism_groups[rep_idx].append(comp_idx)

        result["isomorphism_groups"] = dict(isomorphism_groups)  # Convert back for return value

        # Check groups and apply threshold
        for rep_idx, group_indices in isomorphism_groups.items():
            if len(group_indices) > 1:  # Found an isomorphic group
                result["any_iso_group_found"] = True
                if rep_idx < len(distinct_rep_list):
                    representative_graph = distinct_rep_list[rep_idx]
                    node_count = representative_graph.number_of_nodes()
                    if node_count >= self.min_isomorphic_nodes:
                        # VALID group found
                        result["valid_iso_group_found"] = True
                        num_components_in_group = len(group_indices)
                        result["valid_iso_details"].append((rep_idx, num_components_in_group))
                    # else: group is below threshold (do nothing extra here)
                else:
                    logger.error(
                        "Logic Error: Invalid representative index %d in _analyze_components.",
                        rep_idx,
                    )

        return result

    def _record_valid_symmetry(
        self,
        all_origins_details_list: list,
        analysis_result: dict,
        symmetry_type: AvailableSymmetry,
        cut_obj: Any,
    ) -> None:
        """Adds details of valid isomorphic groups found to the main list."""
        total_components = analysis_result["total_components_found"]
        for rep_idx, num_in_group in analysis_result["valid_iso_details"]:
            all_origins_details_list.append((rep_idx, symmetry_type, cut_obj, num_in_group, total_components))

    # pylint: disable=too-many-branches
    def _plot_symmetry_cut(
        self,
        graph: nx.Graph,
        pos_original: dict,
        symmetry_type: AvailableSymmetry,
        cut_obj: Any,
        nodes_cut: set,
        edges_cut: set,
        analysis_result: dict,
        mol_identifier: str,
        plot_suffix: str,
    ) -> None:
        """Handles the plotting logic for a given cut.

        Args:
            graph (nx.Graph): The original graph.
            pos_original (dict): The layout positions of the original graph.
            symmetry_type (AvailableSymmetry): The symmetry type being checked.
            cut_obj (Any): The object representing the cut.
            nodes_cut (set): Set of nodes that were cut.
            edges_cut (set): Set of edges that were cut.
            analysis_result (dict): Analysis results from _analyze_components.
            mol_identifier (str): Identifier for logging/plotting.
            plot_suffix (str): Suffix for the plot title.

        """
        color_subgraph1 = "mediumseagreen"
        color_subgraph2 = "cornflowerblue"
        color_subgraph3_plus = "darkorange"
        color_cut_element = "red"
        color_default_node = "lightgrey"
        color_default_edge = "silver"

        # Prepare colors
        node_colors_final: list[str] = []
        edge_colors_final: list[str] = []
        all_components_nodes = analysis_result["all_components_nodes"]
        total_components_found = analysis_result["total_components_found"]
        comp1_nodes = all_components_nodes[0] if total_components_found > 0 else set()
        comp2_nodes = all_components_nodes[1] if total_components_found > 1 else set()
        comp3plus_nodes = set().union(*all_components_nodes[2:]) if total_components_found > 2 else set()

        for node in graph.nodes():
            if node in nodes_cut:
                node_colors_final.append(color_cut_element)
            elif node in comp1_nodes:
                node_colors_final.append(color_subgraph1)
            elif node in comp2_nodes:
                node_colors_final.append(color_subgraph2)
            elif node in comp3plus_nodes:
                node_colors_final.append(color_subgraph3_plus)
            else:
                node_colors_final.append(color_default_node)

        for u, v in graph.edges():
            edge_frozenset = frozenset({u, v})
            # pylint: disable=too-many-boolean-expressions
            if edge_frozenset in edges_cut:
                edge_colors_final.append(color_cut_element)
            elif u in comp1_nodes and v in comp1_nodes:
                edge_colors_final.append(color_subgraph1)
            elif u in comp2_nodes and v in comp2_nodes:
                edge_colors_final.append(color_subgraph2)
            elif (u in comp3plus_nodes and v in comp3plus_nodes) or (u in comp3plus_nodes and v in nodes_cut) or (u in nodes_cut and v in comp3plus_nodes):
                edge_colors_final.append(color_subgraph3_plus)
            else:
                edge_colors_final.append(color_default_edge)

        obj_str = str(cut_obj)
        if isinstance(cut_obj, (set, frozenset)):
            obj_str = f"{{{','.join(map(str, sorted(list(cut_obj))))}}}"
        elif isinstance(cut_obj, tuple) and len(cut_obj) > 0 and isinstance(cut_obj[0], tuple):
            obj_str = f"({cut_obj[0]},{cut_obj[1]})"

        plot_title = f"{symmetry_type.name} Check on {mol_identifier} - Cut: {obj_str}\nTotal Comps:{total_components_found} | C1(G):{len(comp1_nodes)} C2(B):{len(comp2_nodes)}{f' C3+(O):{len(comp3plus_nodes)}' if comp3plus_nodes else ''} Cut(R) - ({plot_suffix})"

        try:
            plot_nx_graphs(
                graphs=[graph],
                titles=[plot_title],
                pos_list=[pos_original],
                node_color_list=[node_colors_final],
                edge_color_list=[edge_colors_final],
                figsize_per_plot=(8, 7),
                show_plot=True,
                filename=None,  # Pass filename to save
            )
        except ValueError as e:
            logger.error("Error during plotting for %s, cut %s: %s", mol_identifier, obj_str, e)


if __name__ == "__main__":
    symmetry_filter = SymmetryFilter()

    molecules = [
        ## rings are connected by a single bond - passes
        "Nc1ccc2c(c1)C(=O)c1ccc(Nc3nc(Nc4ccc5c(c4)C(=O)c4ccc(N)cc4C5=O)nc(Nc4ccc5c(c4)C(=O)c4ccc(N)cc4C5=O)n3)cc1C2=O",
        "Clc1nc(Cl)nc(-c2ccc(-c3cc(-c4ccc(-c5nc(Cl)nc(Cl)n5)cc4)cc(-c4ccc(-c5nc(Cl)nc(Cl)n5)cc4)c3)cc2)n1",
        ## shouldn't be symmetrical?? - passes
        "Nc1ccc(-c2nc(-c3ccc(N)nc3)nc(-c3ccc(/N=C/c4cc(/C=N/c5ccc(-c6nc(-c7ccc(N)nc7)nc(-c7ccc(N)nc7)n6)cn5)c(O)c(/C=N/c5ccc(-c6nc(-c7ccc(N)nc7)nc(-c7ccc(N)nc7)n6)cn5)c4)nc3)n2)cn1",
        ## shouldn't be symmetrical?? - no pass
        "O=Cc1ccc(N(c2ccc(C=O)cc2)c2ccc(/C=N/c3ccc(-n4c5ccc(/N=C/c6ccc(N(c7ccc(C=O)cc7)c7ccc(C=O)cc7)cc6)cc5c5cc(/N=C/c6ccc(N(c7ccc(C=O)cc7)c7ccc(C=O)cc7)cc6)ccc54)cc3)cc2)cc1",
        ## rings are connected by having a common edge - not passes
        "Nc1cc2nc3cc4nc5c6nc7cc8nc9cc(N)c(N)cc9nc8cc7nc6c6nc7cc8nc9cc(N)c(N)cc9nc8cc7nc6c5nc4cc3nc2cc1N",
        "Nc1cc2nc3c4nc5cc(N)c(N)cc5nc4c4nc5cc(N)c(N)cc5nc4c3nc2cc1N",
        "NC1=C(N)C(=O)C2N=c3c(c4c(c5c3=NC3C(=O)C(N)=C(N)C(=O)C3N=5)=NC3C(=O)C(N)=C(N)C(=O)C3N=4)=NC2C1=O",
        ## C2H has no graph symmetry - not passes
        "Nc1ccc(-c2ccc3[nH]c(-c4cc(-c5nc6cc(-c7ccc(N)c(N)c7)ccc6[nH]5)cc(N5C(=O)c6ccc7c8c(ccc(c68)C5=O)C(=O)N(c5cc(-c6nc8cc(-c9ccc(N)c(N)c9)ccc8[nH]6)cc(-c6nc8cc(-c9ccc(N)c(N)c9)ccc8[nH]6)c5)C7=O)c4)nc3c2)cc1N",
        ## wild
        "Brc1ccc2c(c1)c1cc(Br)ccc1c1nc3cc4nc5c6ccc(-c7cc(-c8ccc9c(c8)c8ccccc8c8nc%10cc%11nc%12c%13ccc(Br)cc%13c%13cc(Br)ccc%13c%12nc%11cc%10nc98)c8ccc9c(-c%10ccc%11c(c%10)c%10ccccc%10c%10nc%12cc%13nc%14c%15ccc(Br)cc%15c%15cc(Br)ccc%15c%14nc%13cc%12nc%11%10)cc(-c%10ccc%11c(c%10)c%10ccccc%10c%10nc%12cc%13nc%14c%15ccc(Br)cc%15c%15cc(Br)ccc%15c%14nc%13cc%12nc%11%10)c%10c9c8c7CC%10)cc6c6ccccc6c5nc4cc3nc21",
        ## should pass - but doesn't
        "Brc1ccc2c(c1)c1cc(-c3ccc(-c4cc(-c5ccc(-c6ccc7c(c6)c6cc(Br)ccc6c6nc8cc9nc%10c%11ccc(Br)cc%11c%11cc(Br)ccc%11c%10nc9cc8nc76)cc5)cc(-c5ccc(-c6ccc7c(c6)c6cc(Br)ccc6c6nc8cc9nc%10c%11ccc(Br)cc%11c%11cc(Br)ccc%11c%10nc9cc8nc76)cc5)n4)cc3)ccc1c1cc3nc4cc5nc6c7ccc(Br)cc7c7cc(Br)ccc7c6nc5cc4nc3cc21",
        ## very long computation under GraphMatcher - but should pass
        "O=S1(=O)c2ccc(Br)cc2S(=O)(=O)c2cc(-c3ccc(-c4ccc(C(c5ccc(-c6ccc(-c7ccc8c(c7)S(=O)(=O)C7C=CC(Br)=CC7S8(=O)=O)cc6)cc5)C(c5ccc(-c6ccc(-c7ccc8c(c7)S(=O)(=O)C7C=CC(Br)=CC7S8(=O)=O)cc6)cc5)c5ccc(-c6ccc(-c7ccc8c(c7)S(=O)(=O)C7C=C(Br)C=CC7S8(=O)=O)cc6)cc5)cc4)cc3)ccc21",
    ]

    molecules = [
        # "N#C/C=C/c1nc(/C=C/C#N)nc(/C=C/C#N)n1", # won't pass with threshold
        ## doesn't pass (C1)
        "CCc1ccc(NCc2cc(O)c(CNc3ccc(CCc4ccc(CNc5ccc(N(c6ccc(NCc7ccc(CCc8ccc(NCc9cc(O)c(CNc%10ccc(CC)cc%10)cc9O)cc8)cc7O)cc6)c6ccc(NCc7ccc(CCc8ccc(NCc9cc(O)c(CNc%10ccc(CC)cc%10)cc9O)cc8)cc7O)cc6)cc5)c(O)c4)cc3)cc2O)cc1",
        ## will probably pass on node
        "O=Cc1ccc(/C=N/c2ccc(Oc3ccc(/N=C/c4ccc(C=O)cc4)cc3)cc2)cc1",
        "O=CC1=C(O)C(C=O)=C(O)C(=CNc2ccc(N(c3ccc(NC=C4C(=O)C(C=O)=C(O)C(C=O)=C4O)cc3)c3ccc(NC=C4C(=O)C(C=O)=C(O)C(C=O)=C4O)cc3)cc2)C1=O",
    ]

    symmetrical = symmetry_filter.apply([Chem.MolFromSmiles(mol) for mol in molecules], plot_symmetry_checks=True)
    print([Chem.MolToSmiles(mol) for mol in symmetrical])
