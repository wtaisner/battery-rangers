import marimo

__generated_with = "0.15.5"
app = marimo.App(width="full")


@app.cell
def _():
    import glob
    import os

    import marimo as mo
    import pandas as pd
    from rdkit import Chem

    return glob, os, pd


@app.cell
def _(glob, os, pd):
    path = "data/sampling/reinvent/substrate/*/*.csv"
    files = glob.glob(path)

    len(files)

    # read sample from all files
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df["file"] = os.path.basename(file)
        dfs.append(df.sample(1000, random_state=42))

    df = pd.concat(dfs, ignore_index=True)
    df.shape, df.columns
    return (df,)


@app.cell
def _(df):
    from modules.core.enums import MoleculeType
    from modules.core.molecule_filter import MoleculeFilter

    molecule_filter = MoleculeFilter(molecule_type=MoleculeType.SUBSTRATE)

    detailed_results = molecule_filter.apply_against_all_filters(df["SMILES"].tolist()[:10000])
    return (detailed_results,)


@app.cell
def _(detailed_results, pd):
    records = []
    for smiles, filter_outcomes in detailed_results.items():
        # Invert the boolean: True if the molecule failed (result is False)
        record = {filter_name: not passed for filter_name, passed in filter_outcomes.items()}
        records.append(record)

    # Create the DataFrame
    df_for_upset = pd.DataFrame.from_records(records).drop(columns=["Mol Conversion"]).dropna()
    return (df_for_upset,)


@app.cell
def _(pd):
    df_for_upset = pd.read_csv("wandb_export_2025-09-19T12_34_53.792+02_00.csv")
    return (df_for_upset,)


@app.cell
def _(df_for_upset):
    df_for_upset
    return


@app.cell
def _(df_for_upset):
    from matplotlib import pyplot as plt
    from upsetplot import plot

    plot(df_for_upset.groupby(list(df_for_upset.columns)).size())
    plt.show()
    return


if __name__ == "__main__":
    app.run()
