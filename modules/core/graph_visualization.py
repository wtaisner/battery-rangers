"""A set of utility functions to visualize a graph"""

import math
from typing import Any, Sequence

import matplotlib.pyplot as plt
import networkx as nx


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
                print(f"Warning: Kamada-Kawai layout failed for graph {i + 1}. Error: {e}")

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
            print(f"Error saving plot: {e}")
    if show_plot:
        plt.show()
    return None
