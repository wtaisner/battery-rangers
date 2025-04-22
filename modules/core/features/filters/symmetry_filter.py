"""Module for filtering molecules based on symmetry properties."""
import logging
import math
from collections import defaultdict
from typing import Any, Sequence

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
from networkx.algorithms.isomorphism import GraphMatcher
from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.features.filters.generic_filter import GenericMoleculeFilter
from modules.core.features.symmetries import AvailableSymmetry
from modules.core.features.utils import get_graph_from_molecule

logger = logging.getLogger(__name__)  # __name__ ensures the logger is specific to this module
logger.setLevel(logging.DEBUG)


# pylint: disable=too-many-branches, too-many-locals, too-many-statements
def plot_nx_graphs(
    graphs: nx.Graph | Sequence[nx.Graph],
    titles: str | Sequence[str] | None = None,
    pos_list: Sequence[dict[Any, tuple[float, float]]] | None = None,  # Pass positions
    node_color_list: Sequence[Sequence[str] | str] | None = None,  # Pass specific node colors
    edge_color_list: Sequence[Sequence[str] | str] | None = None,  # Pass specific edge colors
    figsize_per_plot: tuple[int, int] = (6, 5),
    node_options: dict[str, Any] | None = None,
    edge_options: dict[str, Any] | None = None,
    label_options: dict[str, Any] | None = None,
    filename: str | None = None,
    show_plot: bool = True,
) -> None:
    """
    Plots one or more NetworkX graphs using Matplotlib, arranging multiple
    graphs in a grid. Uses Kamada-Kawai layout by default if positions are not provided.
    Supports passing pre-calculated positions and specific node/edge colors for highlighting.

    Args:
        graphs: A single NetworkX graph or a sequence (list/tuple) of graphs.
        titles: A single title or a sequence of titles.
        pos_list: Optional. A sequence of position dictionaries (node -> (x, y)),
                  one for each graph. If provided, layout calculation is skipped.
                  Crucial for keeping layouts consistent across related plots.
        node_color_list: Optional. Sequence where each item is a list of node colors
                         for the corresponding graph, or a single color string.
                         Overrides node_options['node_color'].
        edge_color_list: Optional. Sequence where each item is a list of edge colors
                         for the corresponding graph, or a single color string.
                         Overrides edge_options['edge_color'].
        figsize_per_plot: Approximate (width, height) in inches for each subplot.
        node_options: Default dictionary of args for nx.draw_networkx_nodes().
        edge_options: Default dictionary of args for nx.draw_networkx_edges().
        label_options: Default dictionary of args for nx.draw_networkx_labels().
        filename: If provided, saves the plot to this file.
        show_plot: If True, calls plt.show().
    """
    # --- Input Handling ---
    single_graph_input = not isinstance(graphs, (list, tuple))
    if single_graph_input:
        graphs = [graphs]
        if isinstance(titles, str):
            titles = [titles]
        elif titles is not None and not isinstance(titles, (list, tuple)):
            print("Warning: 'titles' type mismatch. Ignoring.")
            titles = None
        # Handle single pos dict, color list etc. if needed, but typically list expected
        if pos_list is not None and isinstance(pos_list, dict):
            pos_list = [pos_list]
        if node_color_list is not None and not isinstance(node_color_list[0], (list, tuple)):
            node_color_list = [node_color_list]  # Wrap single list/str
        if edge_color_list is not None and not isinstance(edge_color_list[0], (list, tuple)):
            edge_color_list = [edge_color_list]  # Wrap single list/str

    num_graphs = len(graphs)
    if num_graphs == 0:
        print("No graphs provided.")
        return None

    # Validate list lengths
    if titles is not None and len(titles) != num_graphs:
        print("Warning: Title count != graph count. Using defaults.")
        titles = None
    if pos_list is not None and len(pos_list) != num_graphs:
        print("Warning: pos_list count != graph count. Ignoring pos_list.")
        pos_list = None
    if node_color_list is not None and len(node_color_list) != num_graphs:
        print("Warning: node_color_list count != graph count. Ignoring node_color_list.")
        node_color_list = None
    if edge_color_list is not None and len(edge_color_list) != num_graphs:
        print("Warning: edge_color_list count != graph count. Ignoring edge_color_list.")
        edge_color_list = None

    # --- Grid and Figure Setup ---
    cols = max(1, num_graphs if num_graphs <= 3 else int(math.ceil(math.sqrt(num_graphs))))  # Prefer horizontal for 3 plots
    rows = int(math.ceil(num_graphs / cols))
    total_figsize = (cols * figsize_per_plot[0], rows * figsize_per_plot[1])
    _, axes = plt.subplots(rows, cols, figsize=total_figsize, squeeze=False)
    axes_flat = axes.flatten()

    # --- Default Drawing Options ---
    default_node_opts = {"node_size": 400, "node_color": "skyblue"}
    default_edge_opts = {"edge_color": "gray", "width": 1.0}
    default_label_opts = {"font_size": 9, "font_color": "black"}

    # Merge user options with defaults (important for non-color options)
    merged_node_opts = {**default_node_opts, **(node_options or {})}
    merged_edge_opts = {**default_edge_opts, **(edge_options or {})}
    merged_label_opts = {**default_label_opts, **(label_options or {})}

    # --- Loop through graphs and Plot ---
    for i, g in enumerate(graphs):
        ax = axes_flat[i]
        current_title = titles[i] if titles else f"Graph {i + 1}"

        if g is None or not isinstance(g, (nx.Graph, nx.DiGraph)):
            print(f"Warning: Item {i} is not a valid NX graph. Skipping.")
            ax.set_title(f"{current_title} (Invalid Graph)")
            ax.axis("off")
            continue

        pos = None
        layout_error = None

        # --- Position Calculation ---
        if pos_list and pos_list[i] is not None:
            pos = pos_list[i]
            # Ensure positions cover all nodes in the current graph G
            missing_nodes = set(g.nodes()) - set(pos.keys())
            if missing_nodes:
                print(f"Warning: Graph {i + 1} nodes missing from pos_list: {missing_nodes}. Layout might be incomplete.")
                # Optional: try to compute layout just for missing? Or filter pos?
                # Filter pos to only include nodes present in G
                pos = {node: p for node, p in pos.items() if node in g.nodes()}

        if pos is None:  # Calculate layout if not provided
            try:
                pos = nx.kamada_kawai_layout(g)
            except ValueError as e:
                logger.error(f"Warning: Kamada-Kawai layout failed for graph {i + 1}. Error: {e}")

        if pos is None:
            ax.set_title(f"{current_title} (Layout Error: {layout_error or 'Unknown'})")
            ax.axis("off")
            continue

        # --- Determine Colors for this subplot ---
        current_node_colors = merged_node_opts.get("node_color", "skyblue")  # Default
        if node_color_list and node_color_list[i] is not None:
            current_node_colors = node_color_list[i]
            if isinstance(current_node_colors, list) and len(current_node_colors) != g.number_of_nodes():
                print(f"Warning: node_color_list[{i}] length mismatch for graph {i + 1}. Using default color.")
                current_node_colors = merged_node_opts.get("node_color", "skyblue")

        current_edge_colors = merged_edge_opts.get("edge_color", "gray")  # Default
        if edge_color_list and edge_color_list[i] is not None:
            current_edge_colors = edge_color_list[i]
            if isinstance(current_edge_colors, list) and len(current_edge_colors) != g.number_of_edges():
                print(f"Warning: edge_color_list[{i}] length mismatch for graph {i + 1}. Using default color.")
                current_edge_colors = merged_edge_opts.get("edge_color", "gray")

        # --- Drawing ---
        # Draw nodes, passing specific colors if available
        node_draw_opts = {**merged_node_opts, "node_color": current_node_colors}  # Start with merged defaults
        nx.draw_networkx_nodes(g, pos, ax=ax, **node_draw_opts)

        # Draw edges, passing specific colors if available
        edge_draw_opts = {**merged_edge_opts, "edge_color": current_edge_colors}  # Start with merged defaults
        nx.draw_networkx_edges(g, pos, ax=ax, **edge_draw_opts)

        # Draw labels (usually doesn't need color override list)
        nx.draw_networkx_labels(g, pos, ax=ax, **merged_label_opts)
        # ---------------

        ax.set_title(current_title)
        ax.axis("off")

    # --- Cleanup and Display ---
    for i in range(num_graphs, len(axes_flat)):
        axes_flat[i].axis("off")
    plt.tight_layout()
    if filename:
        try:
            plt.savefig(filename)
            print(f"Plot saved to {filename}")
        except ValueError as e:
            logger.error(f"Error saving plot: {e}")
    if show_plot:
        plt.show()
    return None


class SymmetryFilter(GenericMoleculeFilter):
    """Check if any of the defined symmetries are present in the molecule."""

    def __init__(self, min_isomorphic_nodes: int = 6, types_to_check: list[AvailableSymmetry] | None = None):
        """
        Args:
            min_isomorphic_nodes (int): The minimum number of nodes an isomorphic
                component must have to be counted/considered a valid symmetry.
                Defaults to 3 (ignores single atoms and diatomic fragments).
            types_to_check (list[AvailableSymmetry]): List of symmetry types to check. If None, defaults to AvailableSymmetry.WHOLE_RING and RING_NODES
        """
        self.min_isomorphic_nodes = min_isomorphic_nodes
        if types_to_check is None:
            self.types_to_check = [AvailableSymmetry.WHOLE_RING, AvailableSymmetry.RING_NODES]
        else:
            self.types_to_check = types_to_check

    def apply(self, molecules: list[Mol], **kwargs) -> list[Mol] | tuple[list[Mol], list[dict]]:
        """
        Filters molecules based on specified symmetry types and threshold.
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
                graph = get_graph_from_molecule(mol)
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

        logger.info("--- Starting Symmetry Analysis ---")
        log_types_names = sorted([t.name for t in self.types_to_check])
        logger.info("Checking symmetry types: %s", log_types_names)

        for i, (mol_graph, original_mol) in enumerate(mol_graphs):
            mol_identifier = f"Mol {i + 1}"
            logger.info("Analyzing %s...", mol_identifier)

            # Stores ONE representative subgraph for each distinct isomorphic shape
            distinct_isomorphic_representatives: list[nx.Graph] = []
            # Stores details for EVERY VALID symmetric cut found
            all_valid_symmetric_origins: list[tuple[int, AvailableSymmetry, Any, int, int]] = []

            if not self.types_to_check:
                logger.warning("No symmetry types configured for %s. Skipping analysis.", mol_identifier)
                molecule_branch_summaries.append({})  # Add empty summary for this mol
                continue

            for symmetry_type_enum in self.types_to_check:
                logger.debug("Checking %s for %s...", symmetry_type_enum.name, mol_identifier)
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
                    logger.error("Error in _check_symmetry for %s, type %s: %s", mol_identifier, symmetry_type_enum.name, e, exc_info=True)

            # --- Summarize and Store results for the current molecule ---
            molecule_summary = {}  # Summary dict for *this* molecule
            if not all_valid_symmetric_origins:
                logger.info("Result for %s: No valid symmetric cuts found (above threshold).", mol_identifier)
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

    @staticmethod
    def _find_or_add_distinct_representative(subgraph_to_check: nx.Graph, distinct_rep_list: list[nx.Graph]) -> int:
        """
        Checks if subgraph_to_check is isomorphic to any graph in distinct_rep_list.

        Args:
            subgraph_to_check (nx.Graph): The subgraph to check for isomorphism.
            distinct_rep_list (list[nx.Graph]): List of distinct representative graphs.
        Returns:
            int: Index of the distinct representative graph in the list.
                    If yes, returns the index of the first match.
                    If no, appends subgraph_to_check to the list and returns its new index.
        """
        for i, stored_graph in enumerate(distinct_rep_list):
            if SymmetryFilter.check_isomorphism(subgraph_to_check, stored_graph):
                return i  # Found match at index i
        # No match found, add it as a new distinct representative
        distinct_rep_list.append(subgraph_to_check)
        return len(distinct_rep_list) - 1  # Return its new index

    @staticmethod
    def check_isomorphism(graph1: nx.Graph, graph2: nx.Graph) -> bool:
        """
        Check whether two graphs are isomorphic using networkx GraphMatcher.

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

        deg_seq1 = sorted([d for n, d in graph1.degree()])
        deg_seq2 = sorted([d for n, d in graph2.degree()])
        if deg_seq1 != deg_seq2:
            return False

        try:
            gm = GraphMatcher(graph1, graph2)
            return gm.is_isomorphic()
        except ValueError as e:
            logger.error(f"Isomorphism check failed: {e}")
            return False  # Treat error as non-isomorphic

    @staticmethod
    def find_all_cycles_of_len(graph: nx.Graph, length: int = 6) -> list:
        """
        Find all cycles of a given length in a graph.

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

        logger.debug(f"All cycles of length {length}: {cycles_of_len}")

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
        elif symmetry_type in (AvailableSymmetry.RING_NODES, AvailableSymmetry.WHOLE_RING):
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
        """
        Checks cuts for a symmetry type. If a valid isomorphic group (above
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

    def _get_layout(self, graph: nx.Graph, plot_checks: bool, mol_identifier: str, symmetry_type: AvailableSymmetry) -> dict[Any, tuple[float, float]] | None:
        """Calculates graph layout if plotting is enabled and possible."""
        pos_original = None
        if plot_checks:
            try:
                if graph.number_of_nodes() > 0:
                    pos_original = nx.kamada_kawai_layout(graph)
                else:
                    pos_original = {}  # Empty layout for empty graph
            except ValueError as e:
                logger.warning("Layout failed for %s (%s): %s. Disabling plots for this check.", mol_identifier, symmetry_type.name, e)
        return pos_original

    # pylint: disable=too-many-return-statements
    def _perform_cut(self, graph: nx.Graph, symmetry_type: AvailableSymmetry, obj: Any) -> dict | None:
        """
        Performs the cut operation on a copy of the graph based on symmetry type.

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

            return {"graph_copy": graph_copy, "nodes_cut": nodes_cut, "edges_cut": edges_cut}

        except (nx.NetworkXError, KeyError) as e:
            logger.debug("Skipping candidate %s (%s): cannot modify graph. %s", obj, symmetry_type.name, e)
            return None

    def _analyze_components(self, original_graph: nx.Graph, graph_after_cut: nx.Graph, distinct_rep_list: list[nx.Graph]) -> dict[str, Any]:
        """
        Analyzes components of the graph after a cut, checks for isomorphism,
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
                    logger.error("Logic Error: Invalid representative index %d in _analyze_components.", rep_idx)

        return result

    def _record_valid_symmetry(self, all_origins_details_list: list, analysis_result: dict, symmetry_type: AvailableSymmetry, cut_obj: Any) -> None:
        """Adds details of valid isomorphic groups found to the main list."""
        total_components = analysis_result["total_components_found"]
        for rep_idx, num_in_group in analysis_result["valid_iso_details"]:
            all_origins_details_list.append((rep_idx, symmetry_type, cut_obj, num_in_group, total_components))

    def _plot_symmetry_cut(
        self, graph: nx.Graph, pos_original: dict, symmetry_type: AvailableSymmetry, cut_obj: Any, nodes_cut: set, edges_cut: set, analysis_result: dict, mol_identifier: str, plot_suffix: str
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

        plot_title = (
            f"{symmetry_type.name} Check on {mol_identifier} - Cut: {obj_str}\n"
            f"Total Comps:{total_components_found} | C1(G):{len(comp1_nodes)} C2(B):{len(comp2_nodes)}"
            f"{f' C3+(O):{len(comp3plus_nodes)}' if comp3plus_nodes else ''} Cut(R)"
            f" - ({plot_suffix})"
        )

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
    ]

    symmetrical = symmetry_filter.apply([Chem.MolFromSmiles(mol) for mol in molecules], plot_symmetry_checks=True)
    print([Chem.MolToSmiles(mol) for mol in symmetrical])
