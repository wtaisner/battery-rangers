import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import wandb

    return pd, plt, sns, wandb


@app.cell
def _(pd, wandb):
    def extract_runs(target_tag: str):
        api = wandb.Api()

        runs = api.runs("witold_taisner/molecule-generation")

        summary_list, config_list, name_list, tags = [], [], [], []
        for run in runs:
            if not run.tags:
                continue  # skip runs without tags
            elif target_tag not in run.tags:
                continue

            # remove target_tag from tags
            tags_to_append = [tag for tag in run.tags if tag != target_tag]
            tags.append(tags_to_append)
            summary_list.append(run.summary._json_dict)

            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith("_")}
            )

            # .name is the human-readable name of the run.
            name_list.append(run.name)

        runs_df = pd.DataFrame(
            {
                "summary": summary_list,
                "config": config_list,
                "name": name_list,
                "tags": tags,
            }
        )
        return runs_df

    return (extract_runs,)


@app.cell
def _(extract_runs):
    node_df = extract_runs("structure")
    node_df
    return (node_df,)


@app.cell
def _(extract_runs):
    substrate_df = extract_runs("NR")
    # remove rows in which "node" is present in name column
    df_filtered = substrate_df[~substrate_df["name"].str.contains("node", na=False)]
    df_filtered = df_filtered[~df_filtered["name"].str.contains("passing", na=False)]
    return (df_filtered,)


@app.cell
def _(df_filtered, node_df, pd):
    runs_df = pd.concat([node_df, df_filtered])
    runs_df
    return (runs_df,)


@app.cell
def _(pd, runs_df):
    # unpack summary column into separate columns
    summary_df = runs_df["summary"].apply(pd.Series)
    config_df = runs_df["config"].apply(pd.Series)
    final_df = pd.concat(
        [runs_df["name"], runs_df["tags"], summary_df, config_df], axis=1
    )
    # make tags to be a single string
    # final_df["tags"] = final_df["tags"].apply(lambda x: " ".join(x))
    # extract new column model_name from first word from name column

    final_df["model_name"] = final_df["name"].str.split(r"[_-]").str[0]

    final_df["tags"] = final_df["tags"].str.join(" ")

    final_df["VUCS"] = (
        final_df["validity_mean"]
        * final_df["uniqueness_mean"]
        * final_df["percent_passing_filters_mean"]
        * 100
    )

    final_df["model_name"].unique(), final_df["tags"].unique()
    return (final_df,)


@app.cell
def _(final_df):
    final_df
    return


@app.cell
def _(final_df, pd):
    pct_metrics = [
        "validity",
        "uniqueness",
        "novelty_wrt_reference_set",
        "novelty_wrt_training_set",
        "percent_passing_filters",
        "internal_diversity",
    ]

    metric_bases = [
        col.replace("_mean", "")
        for col in final_df.columns
        if col.endswith("_mean") and col.replace("_mean", "_std") in final_df.columns
    ]

    metadata_cols = ["name", "tags", "molecule_type", "VUCS"]
    output_df = final_df[metadata_cols].copy()

    for metric in metric_bases:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        mean_val = pd.to_numeric(final_df[mean_col], errors="coerce")
        std_val = pd.to_numeric(final_df[std_col], errors="coerce")

        if metric in pct_metrics:
            # PERCENTAGE LOGIC:
            output_df[metric] = (
                (mean_val * 100).map("{:.1f}".format)
                + "("
                + (std_val * 1000).map("{:.0f}".format)
                + ")"
            )
        else:
            # ABSOLUTE LOGIC (FCD, #circles):
            output_df[metric] = (
                mean_val.map("{:.1f}".format)
                + "("
                + (std_val * 10).map("{:.0f}".format)
                + ")"
            )

    output_df.drop(columns=["num_valid_molecules"], inplace=True)
    # reorder columns to  name, tags, molecule type, validity, uniqueness, internal diversity, circles, fcd, novelty_wrt_reference, passing filters
    output_df = output_df[
        [
            "name",
            "tags",
            "molecule_type",
            "validity",
            "uniqueness",
            "internal_diversity",
            "#circles",
            "fcd",
            "novelty_wrt_reference_set",
            "percent_passing_filters",
            "VUCS",
        ]
    ]
    output_df
    return (output_df,)


@app.cell(hide_code=True)
def _(final_df, output_df, plt, sns):
    # correlation matrix for metrics columns per each molecule_type
    import numpy as np

    metrics_cols = [
        col for col in final_df.columns if "_mean" in col and "novelty" not in col
    ]

    metrics_cols.remove("num_valid_molecules_mean")
    metrics_cols.append("VUCS")

    for mol_type in output_df["molecule_type"].unique():
        subset = final_df[final_df["molecule_type"] == mol_type]
        corr = subset[metrics_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Correlation Matrix for {mol_type}")
        plt.show()
    return (np,)


@app.cell(hide_code=True)
def _(final_df):
    def _():
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from matplotlib.lines import Line2D
        from scipy.stats import pearsonr

        # 1. Configuration
        grid_metrics = ["internal_diversity", "#circles", "fcd"]
        mean_cols = [f"{m}_mean" for m in grid_metrics]
        filter_col = "percent_passing_filters_mean"
        display_names = ["Internal Diversity", "#Circles", "FCD"]

        mol_types = final_df["molecule_type"].unique()
        type_a_raw, type_b_raw = mol_types[0], mol_types[1]
        display_name_map = {type_a_raw: "Substrate", type_b_raw: "Lattice node"}

        # 2. Define visible Gradients
        cmap_a = mcolors.LinearSegmentedColormap.from_list(
            "sub_grad", ["#BDC9CE", "#4A6572"]
        )
        cmap_b = mcolors.LinearSegmentedColormap.from_list(
            "node_grad", ["#EBD7D1", "#A35D47"]
        )

        type_cmaps = {type_a_raw: cmap_a, type_b_raw: cmap_b}
        type_base_colors = {type_a_raw: "#4A6572", type_b_raw: "#A35D47"}

        num_vars = len(mean_cols)
        fig, axes = plt.subplots(
            num_vars, num_vars, figsize=(num_vars * 5.5, num_vars * 5.5)
        )

        # 3. Logarithmic Scaling
        vmin, vmax = 0.001, 0.5
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        intensity_cmap = mcolors.LinearSegmentedColormap.from_list(
            "intensity", ["#f0f0f0", "#111111"]
        )

        for i in range(num_vars):
            for j in range(num_vars):
                ax = axes[i, j]
                ax.tick_params(axis="both", which="major", labelsize=16)

                if i == j:
                    sns.boxplot(
                        data=final_df,
                        x="molecule_type",
                        y=mean_cols[i],
                        ax=ax,
                        palette=type_base_colors,
                        hue="molecule_type",
                        legend=False,
                        width=0.5,
                        showfliers=False,
                    )
                    sns.stripplot(
                        data=final_df,
                        x="molecule_type",
                        y=mean_cols[i],
                        ax=ax,
                        color=".1",
                        size=5,
                        alpha=0.4,
                        dodge=True,
                    )
                    ax.set_xticklabels(
                        [display_name_map[type_a_raw], display_name_map[type_b_raw]]
                    )
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                else:
                    current_type = type_a_raw if i > j else type_b_raw
                    subset = final_df[final_df["molecule_type"] == current_type].copy()
                    f_vals = subset[filter_col].clip(lower=vmin, upper=vmax)
                    point_colors = [
                        type_cmaps[current_type](norm(val)) for val in f_vals
                    ]
                    point_sizes = (subset[filter_col] * 600) + 100

                    ax.scatter(
                        subset[mean_cols[j]],
                        subset[mean_cols[i]],
                        c=point_colors,
                        s=point_sizes,
                        alpha=0.85,
                        edgecolors="0.2",
                        linewidth=1.0,
                    )

                    r, _ = pearsonr(subset[mean_cols[j]], subset[mean_cols[i]])
                    ax.text(
                        0.05,
                        0.95,
                        f"r = {r:.2f}",
                        transform=ax.transAxes,
                        fontsize=18,
                        fontweight="bold",
                        verticalalignment="top",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white",
                            alpha=0.8,
                            edgecolor="none",
                        ),
                    )

                if j == 0:
                    ax.set_ylabel(
                        display_names[i], fontweight="bold", fontsize=22, labelpad=15
                    )
                if i == num_vars - 1:
                    ax.set_xlabel(
                        display_names[j], fontweight="bold", fontsize=22, labelpad=15
                    )

        # --- REFINED UNIFIED LEGEND ---
        type_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Substrate",
                markerfacecolor="#4A6572",
                markersize=14,
                markeredgecolor="0.2",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Lattice node",
                markerfacecolor="#A35D47",
                markersize=14,
                markeredgecolor="0.2",
            ),
        ]

        size_values = [0.01, 0.1, 0.5]
        size_and_color_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=intensity_cmap(norm(v)),
                markeredgecolor="0.2",
                markersize=np.sqrt((v * 600) + 100),
                label=f"{int(v * 100)}%",
            )
            for v in size_values
        ]

        # CSR header handle
        csr_header = [Line2D([0], [0], color="none", label="CSR")]
        all_handles = (
            type_handles
            + [Line2D([0], [0], color="none", label="")]
            + csr_header
            + size_and_color_handles
        )

        leg = fig.legend(
            handles=all_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title="Representation",
            title_fontproperties={"weight": "bold", "size": 20},
            fontsize=16,  # Standard font size for all items
            frameon=True,
            borderpad=1.2,
            labelspacing=1.2,
        )

        # --- ALIGN CSR TO THE LEFT MARGIN ---
        # Legend labels are usually indented by (handlelength + handletextpad).
        # We move CSR to the left to align it with the "R" in "Representation".
        for t in leg.get_texts():
            if t.get_text() == "CSR":
                t.set_fontweight("bold")
                t.set_fontsize(16)  # Explicitly match the other labels
                # Shift left by -35 points (approx the width of the markers/padding)
                t.set_position((-35, 0))

        plt.tight_layout()
        plt.savefig(
            "diversity_quality_matrix.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )
        plt.show()

    _()
    return


@app.cell
def _(final_df):
    def _():
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.ticker import PercentFormatter
        from scipy.stats import gaussian_kde

        def plot_frontier(
            df: pd.DataFrame, name_col: str | None = None, show_validity: bool = False
        ) -> None:
            """
            Discovery Frontier Analysis Visualization:
            - Y-axis label updated to 'Constraint Satisfaction Rate'.
            - Arrow style '->' with thick shaft.
            - Spacing optimized to reduce empty gaps.
            - Points colored by molecule type.
            """
            mol_types = ["substrate", "node"]
            mol_colors = {"substrate": "#4A6572", "node": "#A35D47"}

            # df['VUCS'] = (df['validity_mean'] * df['uniqueness_mean'] * df['percent_passing_filters_mean']) * 100

            if show_validity:
                fig_w, fig_h = 24, 22
                gs = GridSpec(
                    5,
                    5,
                    hspace=0,
                    wspace=0,
                    height_ratios=[0.3, 5, 0.8, 0.3, 5],
                    width_ratios=[6, 0.4, 0.8, 6, 0.4],
                    right=0.98,
                    top=0.95,
                    bottom=0.1,
                )
            else:
                fig_w, fig_h = 24, 11
                gs = GridSpec(
                    2,
                    5,
                    hspace=0,
                    wspace=0,
                    height_ratios=[0.3, 5],
                    width_ratios=[6, 0.4, 0.8, 6, 0.4],
                    right=0.98,
                    top=0.95,
                    bottom=0.25,
                )

            iso_levels = [1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 40.0]
            x_mesh, y_mesh = np.meshgrid(
                np.linspace(0.001, 1.0, 300), np.linspace(0.001, 0.5, 300)
            )
            Z_mesh = (x_mesh * y_mesh) * 100

            def to_plain(text) -> str:
                return str(text).replace("_", " ")

            def get_model_family(name):
                n = str(name).upper()
                if "REINVENT" in n:
                    return "REINVENT"
                if "MOL-AIR" in n:
                    return "Mol-AIR"
                if "MOLMIM" in n:
                    return "MolMIM"
                return "Other"

            fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)

            for m_idx, m_type in enumerate(mol_types):
                subset = df[df["molecule_type"] == m_type].copy()
                if subset.empty:
                    continue
                m_color = mol_colors[m_type]

                for c_idx in range(2):
                    if not show_validity and m_idx != c_idx:
                        continue

                    if show_validity:
                        r_rug, r_main = (
                            (0 if m_idx == 0 else 3),
                            (1 if m_idx == 0 else 4),
                        )
                        c_main, c_rug = (
                            (0 if c_idx == 0 else 3),
                            (1 if c_idx == 0 else 4),
                        )
                        x_var = "uniqueness_mean" if c_idx == 0 else "validity_mean"
                        x_label = "Uniqueness" if c_idx == 0 else "Validity"
                    else:
                        r_rug, r_main = 0, 1
                        c_main, c_rug = (
                            (0 if m_idx == 0 else 3),
                            (1 if m_idx == 0 else 4),
                        )
                        x_var = "uniqueness_mean"
                        x_label = "Uniqueness"

                    ax_main = fig.add_subplot(gs[r_main, c_main])
                    ax_rug_x = fig.add_subplot(gs[r_rug, c_main])
                    ax_rug_y = fig.add_subplot(gs[r_main, c_rug])

                    ax_main.tick_params(labelbottom=True, labelsize=20)
                    ax_main.set_xlabel(
                        x_label, fontsize=28, fontweight="bold", labelpad=25
                    )

                    contours = ax_main.contour(
                        x_mesh,
                        y_mesh,
                        Z_mesh,
                        levels=iso_levels,
                        colors="darkgray",
                        alpha=0.4,
                        linestyles="--",
                    )
                    fmt = {l: f"{l:.1f}%" for l in iso_levels}
                    ax_main.clabel(
                        contours, inline=True, fontsize=18, fmt=fmt, colors="black"
                    )

                    arrow = FancyArrowPatch(
                        (0.15, 0.075),
                        (0.82, 0.41),
                        arrowstyle="->,head_width=2.5,head_length=3.0",
                        color="darkgray",
                        alpha=0.15,
                        linewidth=40,
                        zorder=1,
                        mutation_scale=20,
                    )
                    ax_main.add_patch(arrow)

                    ax_main.text(
                        0.48,
                        0.24,
                        "VUCS",
                        color="gray",
                        fontsize=28,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        rotation=26.57,
                        alpha=0.6,
                        zorder=2,
                    )

                    ax_main.scatter(
                        subset[x_var],
                        subset["percent_passing_filters_mean"],
                        color=m_color,
                        s=200,
                        edgecolors="black",
                        linewidths=0.8,
                        zorder=10,
                    )

                    if name_col:
                        subset["model_fam"] = subset[name_col].apply(get_model_family)
                        model_list = ["REINVENT", "Mol-AIR", "MolMIM"]
                        for stack_idx, fam in enumerate(model_list):
                            f_sub = subset[subset["model_fam"] == fam]
                            if f_sub.empty:
                                continue
                            row = f_sub.loc[f_sub["VUCS"].idxmax()]
                            ha = "right" if row[x_var] > 0.5 else "left"
                            x_off = -15 if ha == "right" else 15
                            ax_main.annotate(
                                to_plain(row[name_col]),
                                xy=(row[x_var], row["percent_passing_filters_mean"]),
                                xytext=(x_off, 12 + (stack_idx * 18)),
                                textcoords="offset points",
                                fontsize=18,
                                fontweight="bold",
                                ha=ha,
                                zorder=12,
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    fc="white",
                                    ec="gray",
                                    alpha=0.85,
                                ),
                                arrowprops=dict(
                                    arrowstyle="->", color="black", alpha=0.3
                                ),
                            )

                    for ax_r, data_vec, orient in [
                        (ax_rug_x, subset[x_var], "h"),
                        (ax_rug_y, subset["percent_passing_filters_mean"], "v"),
                    ]:
                        if len(data_vec.unique()) > 1:
                            kde = gaussian_kde(data_vec)
                            supp = np.linspace(0, 1 if orient == "h" else 0.5, 200)
                            dens = kde(supp)
                            dens /= dens.max()
                            if orient == "h":
                                ax_r.fill_between(
                                    supp, 0, dens, color=m_color, alpha=0.2, lw=0
                                )
                            else:
                                ax_r.fill_betweenx(
                                    supp, 0, dens, color=m_color, alpha=0.2, lw=0
                                )

                        if orient == "h":
                            ax_r.vlines(
                                data_vec, 0, 1, color=m_color, lw=1.5, alpha=0.8
                            )
                        else:
                            ax_r.hlines(
                                data_vec, 0, 1, color=m_color, lw=1.5, alpha=0.8
                            )
                        ax_r.axis("off")
                        ax_r.set_xlim(0, 1) if orient == "h" else ax_r.set_ylim(0, 0.5)

                    ax_main.set_xlim(0, 1)
                    ax_main.set_ylim(0, 0.5)
                    ax_main.xaxis.set_major_formatter(PercentFormatter(1.0))
                    ax_main.yaxis.set_major_formatter(PercentFormatter(1.0))
                    ax_main.tick_params(axis="y", labelsize=20)
                    ax_main.grid(True, linestyle=":", alpha=0.3)

                    if c_idx == 0:
                        ax_main.set_ylabel(
                            "Constraint Satisfaction Rate",
                            fontsize=28,
                            fontweight="bold",
                            labelpad=25,
                        )
                    else:
                        plt.step(ax_main.get_yticklabels(), visible=False)

            filename = "VUCS_frontier" + ("_validity" if show_validity else "") + ".svg"
            plt.savefig(filename, format="svg", transparent=True, bbox_inches="tight")
            plt.show()

        plot_frontier(final_df, name_col="name", show_validity=False)

    _()
    return


@app.cell
def _(final_df):
    def _():
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
        from matplotlib.ticker import PercentFormatter
        from scipy.stats import gaussian_kde

        def plot_frontier(
            df: pd.DataFrame,
            name_col: str | None = None,
            show_validity: bool = False,
            jitter: float = 0.012,
        ) -> None:
            """
            Discovery Frontier Analysis Visualization:
            - Rug plots aggregated per molecule type using original colors.
            - Points colored by model name for left plot, unified for right plot.
            - Base version = Hollow, Fine-Tuning = Filled.
            - Vanilla = Solid edge line, ChEMBL35 = Dashed edge line.
            - RL = Triangle shape, Seed = Inner star.
            - Legend organized into two non-overlapping rows at the bottom.
            """
            mol_types = ["substrate", "node"]

            # Colors for the marginal rug/KDE plots (original unified colors)
            mol_colors = {"substrate": "#4A6572", "node": "#A35D47"}

            # Colors for the scatter points on the left plot
            model_colors = {
                "REINVENT": "#4A6572",  # Original slate blue/gray
                "MolMIM": "#E68A5C",  # Rust orange
                "Mol-AIR": "#7EB26D",  # Muted green
            }

            # Increased bottom margin to accommodate the two-row legend setup comfortably
            bottom_margin = 0.20 if show_validity else 0.28

            if show_validity:
                fig_w, fig_h = 24, 22
                gs = GridSpec(
                    5,
                    5,
                    hspace=0,
                    wspace=0,
                    height_ratios=[0.3, 5, 0.8, 0.3, 5],
                    width_ratios=[6, 0.4, 0.8, 6, 0.4],
                    right=0.98,
                    top=0.95,
                    bottom=bottom_margin,
                )
            else:
                fig_w, fig_h = 24, 11
                gs = GridSpec(
                    2,
                    5,
                    hspace=0,
                    wspace=0,
                    height_ratios=[0.3, 5],
                    width_ratios=[6, 0.4, 0.8, 6, 0.4],
                    right=0.98,
                    top=0.95,
                    bottom=bottom_margin,
                )

            iso_levels = [1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 40.0]
            x_mesh, y_mesh = np.meshgrid(
                np.linspace(0.001, 1.0, 300), np.linspace(0.001, 0.5, 300)
            )
            Z_mesh = (x_mesh * y_mesh) * 100

            # Reproducible random number generator for jitter
            rng = np.random.default_rng(42)

            def to_plain(text) -> str:
                return str(text).replace("_", " ")

            def get_model_family(name):
                n = str(name).upper()
                if "MOL-AIR" in n or "MOL_AIR" in n:
                    return "Mol-AIR"
                if "MOLMIM" in n:
                    return "MolMIM"
                return "REINVENT"

            fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)

            # Store the last axis to use for legend proxy artists later
            last_ax_main = None

            for m_idx, m_type in enumerate(mol_types):
                subset = df[df["molecule_type"] == m_type].copy()
                if subset.empty:
                    continue

                # The unified color for the rug plot for this molecule type
                m_color = mol_colors[m_type]

                # Pre-calculate features from names
                if name_col:
                    subset["model_fam"] = subset[name_col].apply(get_model_family)
                    subset["ft"] = subset[name_col].apply(
                        lambda x: "ft" in str(x).lower()
                    )
                    subset["rl"] = subset[name_col].apply(
                        lambda x: "rl" in str(x).lower()
                    )
                    subset["seed"] = subset[name_col].apply(
                        lambda x: "seed" in str(x).lower()
                    )
                    subset["chembl35"] = subset[name_col].apply(
                        lambda x: "chembl35" in str(x).lower()
                    )

                for c_idx in range(2):
                    if not show_validity and m_idx != c_idx:
                        continue

                    if show_validity:
                        r_rug, r_main = (
                            (0 if m_idx == 0 else 3),
                            (1 if m_idx == 0 else 4),
                        )
                        c_main, c_rug = (
                            (0 if c_idx == 0 else 3),
                            (1 if c_idx == 0 else 4),
                        )
                        x_var = "uniqueness_mean" if c_idx == 0 else "validity_mean"
                        x_label = "Uniqueness" if c_idx == 0 else "Validity"
                    else:
                        r_rug, r_main = 0, 1
                        c_main, c_rug = (
                            (0 if m_idx == 0 else 3),
                            (1 if m_idx == 0 else 4),
                        )
                        x_var = "uniqueness_mean"
                        x_label = "Uniqueness"

                    # --- JITTER & BOUNDARY LOGIC ---
                    if jitter > 0:
                        x_noise = rng.uniform(-jitter, jitter, size=len(subset))
                        y_noise = rng.uniform(-jitter, jitter, size=len(subset))

                        # Clip to bounds slightly inside the axis lines to prevent marker overlap with axes
                        subset["plot_x"] = (subset[x_var] + x_noise).clip(0.015, 0.985)
                        subset["plot_y"] = (
                            subset["percent_passing_filters_mean"] + y_noise
                        ).clip(0.005, 0.495)
                    else:
                        subset["plot_x"] = subset[x_var]
                        subset["plot_y"] = subset["percent_passing_filters_mean"]

                    ax_main = fig.add_subplot(gs[r_main, c_main])
                    last_ax_main = ax_main

                    ax_rug_x = fig.add_subplot(gs[r_rug, c_main])
                    ax_rug_y = fig.add_subplot(gs[r_main, c_rug])

                    ax_main.tick_params(labelbottom=True, labelsize=20)
                    ax_main.set_xlabel(
                        x_label, fontsize=28, fontweight="bold", labelpad=25
                    )

                    contours = ax_main.contour(
                        x_mesh,
                        y_mesh,
                        Z_mesh,
                        levels=iso_levels,
                        colors="darkgray",
                        alpha=0.4,
                        linestyles="--",
                    )
                    fmt = {l: f"{l:.1f}%" for l in iso_levels}
                    ax_main.clabel(
                        contours, inline=True, fontsize=18, fmt=fmt, colors="black"
                    )

                    # Group by ALL combinatorial features to plot efficiently
                    for (
                        model,
                        has_ft,
                        has_rl,
                        has_seed,
                        is_c35,
                    ), group in subset.groupby(
                        ["model_fam", "ft", "rl", "seed", "chembl35"]
                    ):
                        # Resolve base color
                        color_val = (
                            model_colors.get(model, "#808080")
                            if m_type == "substrate"
                            else mol_colors["node"]
                        )

                        # Resolve Visual Grammar
                        m_shape = "^" if has_rl else "o"
                        m_face = (
                            color_val if has_ft else "white"
                        )  # Solid for FT, Hollow for Base
                        m_edge = (
                            "black" if has_ft else color_val
                        )  # Contrast edge for FT, Colored edge for Base
                        m_ls = (
                            "--" if is_c35 else "-"
                        )  # Dashed for ChEMBL35, Solid for Vanilla
                        m_lw = 2.5  # Thick enough for dashes to be highly visible
                        m_size = 280 if has_rl else 220

                        # 1. Plot Base Geometry
                        ax_main.scatter(
                            group["plot_x"],
                            group["plot_y"],
                            facecolor=m_face,
                            edgecolor=m_edge,
                            marker=m_shape,
                            s=m_size,
                            linewidth=m_lw,
                            linestyle=m_ls,
                            zorder=10,
                        )

                        # 2. Plot Seed Overlay
                        if has_seed:
                            ax_main.scatter(
                                group["plot_x"],
                                group["plot_y"],
                                color="black",
                                edgecolor="white",
                                linewidth=0.8,
                                marker="*",
                                s=120,
                                zorder=11,
                            )

                    # Peak Annotations (pointed to the jittered location)
                    if name_col:
                        model_list = ["REINVENT", "Mol-AIR", "MolMIM"]
                        for stack_idx, fam in enumerate(model_list):
                            f_sub = subset[subset["model_fam"] == fam]
                            if f_sub.empty:
                                continue
                            if "VUCS" in f_sub.columns:
                                row = f_sub.loc[f_sub["VUCS"].idxmax()]
                                ha = "right" if row["plot_x"] > 0.5 else "left"
                                x_off = -15 if ha == "right" else 15
                                ax_main.annotate(
                                    to_plain(row[name_col]),
                                    xy=(row["plot_x"], row["plot_y"]),
                                    xytext=(x_off, 12 + (stack_idx * 18)),
                                    textcoords="offset points",
                                    fontsize=16,
                                    fontweight="bold",
                                    ha=ha,
                                    zorder=12,
                                    bbox=dict(
                                        boxstyle="round,pad=0.2",
                                        fc="white",
                                        ec="gray",
                                        alpha=0.85,
                                    ),
                                    arrowprops=dict(
                                        arrowstyle="->", color="black", alpha=0.3
                                    ),
                                )

                    # KDE and Rug Plots aggregated by molecule type
                    for ax_r, data_col, orient in [
                        (ax_rug_x, x_var, "h"),
                        (ax_rug_y, "percent_passing_filters_mean", "v"),
                    ]:
                        data_vec = subset[data_col]

                        if len(data_vec.unique()) > 1:
                            kde = gaussian_kde(data_vec)
                            supp = np.linspace(0, 1 if orient == "h" else 0.5, 200)
                            dens = kde(supp)
                            dens /= dens.max()
                            if orient == "h":
                                ax_r.fill_between(
                                    supp, 0, dens, color=m_color, alpha=0.2, lw=0
                                )
                            else:
                                ax_r.fill_betweenx(
                                    supp, 0, dens, color=m_color, alpha=0.2, lw=0
                                )

                        if orient == "h":
                            ax_r.vlines(
                                data_vec, 0, 1, color=m_color, lw=1.5, alpha=0.8
                            )
                        else:
                            ax_r.hlines(
                                data_vec, 0, 1, color=m_color, lw=1.5, alpha=0.8
                            )

                        ax_r.axis("off")
                        ax_r.set_xlim(0, 1) if orient == "h" else ax_r.set_ylim(0, 0.5)

                    ax_main.set_xlim(0, 1)
                    ax_main.set_ylim(0, 0.5)
                    ax_main.xaxis.set_major_formatter(PercentFormatter(1.0))
                    ax_main.yaxis.set_major_formatter(PercentFormatter(1.0))
                    ax_main.tick_params(axis="y", labelsize=20)
                    ax_main.grid(True, linestyle=":", alpha=0.3)

                    if c_idx == 0:
                        ax_main.set_ylabel(
                            "Constraint Satisfaction Rate",
                            fontsize=28,
                            fontweight="bold",
                            labelpad=25,
                        )
                    else:
                        plt.step(ax_main.get_yticklabels(), visible=False)

            # --- Figure-Level Legend Placement ---
            # 1. Model Colors
            color_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=m,
                    markerfacecolor=c,
                    markersize=14,
                    markeredgecolor="black",
                )
                for m, c in model_colors.items()
            ]

            # 2. Pre-training (Cleaned up labels without parentheses)
            sc_vanilla = last_ax_main.scatter(
                [],
                [],
                facecolor="white",
                edgecolor="black",
                marker="o",
                s=150,
                linewidth=2.5,
                linestyle="-",
                label="Vanilla",
            )
            sc_c35 = last_ax_main.scatter(
                [],
                [],
                facecolor="white",
                edgecolor="black",
                marker="o",
                s=150,
                linewidth=2.5,
                linestyle="--",
                label="ChEMBL35",
            )
            pretrain_handles = [sc_vanilla, sc_c35]

            # 3. Combinable Methods
            comp_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor="#808080",
                    markeredgewidth=2.5,
                    markersize=14,
                    label="Base (Hollow)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#808080",
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    markersize=14,
                    label="+ FT (Solid Fill)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor="#808080",
                    markeredgecolor="black",
                    markersize=14,
                    label="+ RL (Triangle Shape)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    markerfacecolor="black",
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    markersize=14,
                    label="+ Seed (Inner Star)",
                ),
            ]

            # Draw Legends on the overall figure, separated into two rows

            # Row 1 (Top Row of Legend): Model Colors & Pre-training
            fig.legend(
                handles=color_handles,
                title="Model Colors",
                loc="lower center",
                bbox_to_anchor=(0.33, 0.12),
                ncol=3,
                fontsize=16,
                title_fontsize=18,
                framealpha=0.9,
            )

            fig.legend(
                handles=pretrain_handles,
                title="Pre-training",
                loc="lower center",
                bbox_to_anchor=(0.67, 0.12),
                ncol=2,
                fontsize=16,
                title_fontsize=18,
                framealpha=0.9,
            )

            # Row 2 (Bottom Row of Legend): Combinable Methods
            fig.legend(
                handles=comp_handles,
                title="Combinable Methods",
                loc="lower center",
                bbox_to_anchor=(0.50, 0.02),
                ncol=4,
                fontsize=16,
                title_fontsize=18,
                framealpha=0.9,
            )

            filename = "VUCS_frontier" + ("_validity" if show_validity else "") + ".svg"
            plt.savefig(filename, format="svg", transparent=True, bbox_inches="tight")
            plt.show()

        plot_frontier(final_df, name_col="name", show_validity=False, jitter=0.01)

    _()
    return


@app.cell
def _(final_df, np):
    import plotly.express as px
    import plotly.graph_objects as go

    # import numpy as np

    def _():
        # 1. Clean Tags: Remove "NR" and handle formatting
        df_plot = final_df.copy()

        def clean_tags(t):
            if not isinstance(t, str):
                return "Standard"
            # Remove "NR" specifically (using word boundaries to avoid catching strings like "INR")
            import re

            cleaned = re.sub(r"\bNR\b", "", t)
            # Clean up commas and whitespace left behind
            cleaned = cleaned.replace(",,", ",").strip(", ")
            return cleaned if cleaned != "" else "Standard"

        df_plot["tags_clean"] = df_plot["tags"].apply(clean_tags)

        # 2. Define a consistent mapping of Tags to Shapes
        # This ensures 'vanilla' is ALWAYS the same shape across different models
        unique_tags = sorted(df_plot["tags_clean"].unique())
        # Plotly symbols: circle, diamond, square, x, cross, triangle-up, pentagon, etc.
        symbols_list = [
            "circle",
            "diamond",
            "square",
            "x",
            "cross",
            "triangle-up",
            "star",
            "hexagram",
        ]
        symbol_map = {
            tag: symbols_list[i % len(symbols_list)]
            for i, tag in enumerate(unique_tags)
        }

        # 3. Add Jitter for the stripplot effect
        df_plot["jitter"] = np.random.uniform(-0.15, 0.15, len(df_plot))

        metrics = [
            # "validity_mean",
            # "uniqueness_mean",
            "percent_passing_filters_mean",
            # "internal_diversity_mean",
            # "fcd_mean",
            # "#circles_mean",
            "potential_yield",
        ]

        for metric in metrics:
            clean_label = metric.replace("_mean", "").replace("_", " ").title()

            # 4. Create the scatter plot
            fig = px.scatter(
                df_plot,
                x="jitter",
                y=metric,
                color="model_name",
                symbol="tags_clean",  # Maps to shapes
                symbol_map=symbol_map,  # FORCES consistency across models
                facet_col="molecule_type",
                hover_name="run_name",
                hover_data={
                    "jitter": False,
                    "model_name": True,
                    "tags_clean": True,
                    metric: ":.4f",
                },
                title=f"{clean_label} Distribution",
                labels={
                    metric: clean_label,
                    "model_name": "Model",
                    "tags_clean": "Tag",
                },
                template="plotly_white",
            )

            # 5. Add one overall Boxplot per facet (molecule_type)
            mol_types = sorted(df_plot["molecule_type"].unique())
            for i, mol_type in enumerate(mol_types):
                facet_data = df_plot[df_plot["molecule_type"] == mol_type]

                fig.add_trace(
                    go.Box(
                        y=facet_data[metric],
                        x=[0] * len(facet_data),
                        name="Overall Distribution",
                        marker_color="rgba(180, 180, 180, 0.3)",
                        fillcolor="rgba(200, 200, 200, 0.1)",
                        line=dict(color="rgba(100, 100, 100, 0.5)", width=1),
                        boxpoints=False,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=i + 1,
                )

            # 6. Styling Polish
            fig.update_traces(
                marker=dict(size=10, opacity=0.8, line=dict(width=0.5, color="white"))
            )

            # Clean up facet headers
            fig.for_each_annotation(
                lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>")
            )

            # Move legend to the side and format
            fig.update_layout(
                legend_title_text="<b>Model, Tag</b>",
                xaxis_showticklabels=False,
                xaxis_title=None,
                xaxis2_showticklabels=False,
                xaxis2_title=None,
                margin=dict(l=50, r=50, t=80, b=50),
            )

            fig.show()

    _()
    return


if __name__ == "__main__":
    app.run()
