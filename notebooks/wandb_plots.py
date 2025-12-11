import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    import wandb

    return pd, plt, sns, wandb


@app.cell
def _(pd, wandb):
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("witold_taisner/molecule-generation")

    summary_list, config_list, name_list, tags = [], [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files

        if not run.tags:
            continue  # skip runs without tags

        tags.append(run.tags)
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({"summary": summary_list, "config": config_list, "name": name_list, "tags": tags})
    return (runs_df,)


@app.cell
def _(pd, runs_df):
    # unpack summary column into separate columns
    summary_df = runs_df["summary"].apply(pd.Series)
    config_df = runs_df["config"].apply(pd.Series)
    final_df = pd.concat([runs_df["name"], runs_df["tags"], summary_df, config_df], axis=1)
    # make tags to be a single string
    # final_df["tags"] = final_df["tags"].apply(lambda x: " ".join(x))
    # extract new column model_name from first word from name column

    final_df["model_name"] = final_df["name"].str.split(r"[_-]").str[0]

    final_df["tags"] = final_df["tags"].str.join(" ")

    final_df["model_name"].unique(), final_df["tags"].unique()
    return (final_df,)


@app.cell
def _(final_df):
    final_df
    return


@app.cell
def _(final_df, plt, sns):
    # 1. Create a combined column to act as the plot grouping
    # This flattens the matrix into a list of existing combinations only
    final_df["plot_group"] = final_df["molecule_type"] + " | " + final_df["model_name"]

    sns.set_context("paper")  # Or "paper", "notebook", "poster"

    # Optional: Sort values so the plots appear in a logical order (e.g., all Nodes first)
    df = final_df.sort_values(by=["molecule_type", "model_name"])

    # 2. Create the FacetGrid using col_wrap
    # col_wrap=3 means it will put 3 plots per row, then start a new line
    g = sns.FacetGrid(
        df, col="plot_group", col_wrap=2, height=6, aspect=1.5, sharex=False, sharey=True  # Increased from 5 to 6 (Height of each individual plot)  # Increased from 1.3 to 1.5 (Makes each plot wider)
    )

    # 3. Map the Bars
    g.map_dataframe(sns.barplot, x="tags", y="percent_passing_filters_mean", hue="tags", dodge=False, palette="viridis", alpha=0.8, capsize=0)

    # 4. Map the Custom Error Bars
    g.map(plt.errorbar, "tags", "percent_passing_filters_mean", "percent_passing_filters_std", fmt="none", ecolor="black", capsize=5, elinewidth=1.5)

    # 5. Set Axis Limits (The Fix)
    # (0, None) forces the bottom to 0, but lets the top expand automatically
    g.set(ylim=(0, None))

    # 6. Formatting
    g.set_axis_labels("Tags", "Percent Passing Filter (Mean)")
    g.set_titles("{col_name}")

    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
