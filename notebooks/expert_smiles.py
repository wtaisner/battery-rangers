import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full", sql_output="pandas")


@app.cell
def _():
    import math
    import sys

    import lets_plot as lp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import selfies as sf
    from rdkit import Chem, RDLogger
    from sklearn.model_selection import train_test_split

    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
    return Chem, math, mo, np, pd, plt, sf, sys, train_test_split


@app.cell
def _(sys):
    sys.path.append("../")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 15.09.2025 - data reading / parsing
    """
    )
    return


@app.cell
def _(Chem, pd):
    df = pd.read_excel(
        "data/raw/data_battery_materials_PRISTINE_CORRECTED_15.09.xlsx",
        sheet_name="Sheet1",
    )

    df = df[["smiles (substrate)", "SMILES to be used (molecules with a node)", "capacitance_max"]]
    df.rename(columns={"smiles (substrate)": "smiles_substrate", "SMILES to be used (molecules with a node)": "smiles_node"}, inplace=True)

    def canonicalize_smiles(smiles_string):
        """
        Attempts to canonicalize a SMILES string.
        Returns the canonical SMILES or pd.NA on failure.
        """
        if pd.isna(smiles_string) or not isinstance(smiles_string, str) or smiles_string.strip() == "":
            return pd.NA

        try:
            mol = Chem.MolFromSmiles(smiles_string)
            if mol is None:
                print(f"Invalid SMILES string provided: '{smiles_string}'")
                return pd.NA
            return Chem.MolToSmiles(mol)
        except Exception as e:
            print(f"Error processing SMILES string '{smiles_string}': {e}")
            return pd.NA

    smiles_columns = [col for col in df.columns if "smiles" in col]
    for col_name in smiles_columns:
        df[col_name] = df[col_name].apply(canonicalize_smiles)

    df.head(), df.shape
    return df, smiles_columns


@app.cell
def _(df):
    print(df.isna().sum())
    na_rows = df[df.isna().any(axis=1)]
    na_rows
    return


@app.cell
def _(df, smiles_columns):
    # look for duplicates in all smiles columns
    def find_duplicates(df, smiles_columns):
        duplicates = {}
        for col in smiles_columns:
            dupes = df[df.duplicated(subset=[col], keep=False)]
            if not dupes.empty:
                duplicates[col] = dupes
        return duplicates

    duplicates = find_duplicates(df, smiles_columns)
    duplicates
    return


@app.cell
def _(df):
    print(f"Shape before deduplication and NA removal: {df.shape}")
    # drop duplicates and NA based on smiles_substrate or smiles_to_be_used with a node
    df.dropna(subset=["smiles_substrate", "smiles_node"], inplace=True)
    df.drop_duplicates(subset=["smiles_substrate", "smiles_node"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Shape after deduplication and NA removal: {df.shape}")
    return


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Train-Test Split | SELFIES encoding
    """
    )
    return


@app.cell
def _(Chem, df, sf):
    df["selfies_substrate"] = df["smiles_substrate"].apply(sf.encoder)
    df["selfies_node"] = df["smiles_node"].apply(sf.encoder)

    df["molecule_substrate"] = df["smiles_substrate"].apply(Chem.MolFromSmiles)
    df["molecule_node"] = df["smiles_node"].apply(Chem.MolFromSmiles)

    df["selfies_substrate"].to_csv("data/raw/substrate/experts_substrate.slf", sep=" ", index=None, header=None)
    # df["selfies_node"].to_csv("data/raw/node/experts_node.slf", sep=" ", index=None, header=None)

    df["smiles_substrate"].rename("canon_smiles").to_csv("data/raw/substrate/experts_substrate.csv", index=None)
    df["smiles_node"].rename("canon_smiles").to_csv("data/raw/node/experts_node.csv", index=None, header=1)

    df.to_csv("data/raw/experts_15_09_25_parsed.csv", index=None)
    return


@app.cell
def _(df, train_test_split):
    X_train, X_test, _, _ = train_test_split(df, df["capacitance_max"], test_size=0.2, random_state=42)

    # save train-test sets
    X_train["selfies_substrate"].to_csv("data/raw/substrate/train_selfies.slf", sep=" ", index=None, header=None)
    X_train["selfies_node"].to_csv("data/raw/node/train_selfies.slf", sep=" ", index=None, header=None)

    X_test["selfies_substrate"].to_csv("data/raw/substrate/test_selfies.slf", sep=" ", index=None, header=None)
    X_test["selfies_node"].to_csv("data/raw/node/test_selfies.slf", sep=" ", index=None, header=None)

    # X_train["smiles_substrate"].to_csv("data/raw/substrate/train_substrate.smi", index=None, sep=" ", header=None)
    # X_test["smiles_substrate"].to_csv("data/raw/substrate/test_substrate.smi", index=None, sep=" ", header=None)

    # X_train["smiles_node"].to_csv("data/raw/node/chembl35_train_node.smi", index=None, sep=" ", header=None)
    # X_test["smiles_node"].to_csv("data/raw/node/chembl35_test_node.smi", index=None, sep=" ", header=None)

    X_train["smiles_substrate"].rename("canon_smiles").to_csv("data/raw/substrate/train.csv", index=None)
    X_train["smiles_node"].rename("canon_smiles").to_csv("data/raw/node/train.csv", index=None)

    X_test["smiles_substrate"].rename("canon_smiles").to_csv("data/raw/substrate/test.csv", index=None)
    X_test["smiles_node"].rename("canon_smiles").to_csv("data/raw/node/test.csv", index=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## SMARTS - CTF filtering
    """
    )
    return


@app.cell
def _():
    import os

    os.getcwd()
    return


@app.cell
def _(Chem, pd):
    from modules.core.filters.smarts_filter import InclusionRule, SMARTSFilter

    ctf_filter = SMARTSFilter(rules=[InclusionRule(["C#N"], 2)])
    smarts_filter = SMARTSFilter()

    data = pd.read_excel("data/ctfy-wygenerowane-capacity-20-11-2025_with_filters.xlsx")

    # check how many columns from last three have "no" in all three
    last_three = data.columns[-3:]
    data["all_no"] = data[last_three].apply(lambda x: all(v == "no" for v in x), axis=1)
    print(data["all_no"].value_counts())

    data["passed_smarts"] = data["smiles"].apply(lambda x: smarts_filter.apply([Chem.MolFromSmiles(x)]))

    # count non empty passed_smarts
    data["passed_smarts_count"] = data["passed_smarts"].apply(lambda x: len(x) if x is not None else 0)
    data["passed_smarts_bool"] = data["passed_smarts_count"] > 0
    data["passed_smarts_bool"].value_counts()
    return (ctf_filter,)


@app.cell
def _(ctf_filter, df):
    print(f"Total molecules: {len(df)}")
    filtered = ctf_filter.apply(df["molecule_substrate"])
    print(f"CTF only: {len(filtered)}")

    for idx, mol in enumerate(df["molecule_substrate"]):
        filtered = ctf_filter.apply([mol])
        if not filtered:
            df.at[idx, "CTF"] = False
        else:
            df.at[idx, "CTF"] = True

    ctfs = df[df["CTF"] == True]
    ctfs.head()
    return (ctfs,)


@app.cell
def _(ctfs, train_test_split):
    X_ctf_train, X_ctf_test, _, _ = train_test_split(ctfs, ctfs["capacitance_max"], test_size=0.2, random_state=42)

    # save train-test sets
    X_ctf_train["selfies_substrate"].to_csv("data/raw/substrate/ctf_train_selfies.slf", sep=" ", index=None, header=None)
    X_ctf_test["selfies_substrate"].to_csv("data/raw/substrate/ctf_test_selfies.slf", sep=" ", index=None, header=None)
    X_ctf_train["smiles_substrate"].rename("canon_smiles").to_csv("data/raw/substrate/ctf_train.csv", index=None)
    X_ctf_test["smiles_substrate"].rename("canon_smiles").to_csv("data/raw/substrate/ctf_test.csv", index=None)

    # save .smi for REINVENT pre-trained on chembl35
    X_ctf_train["smiles_substrate"].to_csv("data/raw/substrate/ctf_train_chembl35.smi", index=None, sep=" ", header=None)
    X_ctf_test["smiles_substrate"].to_csv("data/raw/substrate/ctf_test_chembl35.smi", index=None, sep=" ", header=None)

    X_ctf_train["smiles_node"].to_csv("data/raw/node/ctf_train_chembl35.smi", index=None, sep=" ", header=None)
    X_ctf_test["smiles_node"].to_csv("data/raw/node/ctf_test_chembl35.smi", index=None, sep=" ", header=None)

    # save for nodes
    X_ctf_train["selfies_node"].to_csv("data/raw/node/ctf_train_selfies.slf", sep=" ", index=None, header=None)
    X_ctf_test["selfies_node"].to_csv("data/raw/node/ctf_test_selfies.slf", sep=" ", index=None, header=None)
    X_ctf_train["smiles_node"].rename("canon_smiles").to_csv("data/raw/node/ctf_train.csv", index=None)
    X_ctf_test["smiles_node"].rename("canon_smiles").to_csv("data/raw/node/ctf_test.csv", index=None)

    # save full ctf dataset
    ctfs.to_csv("data/raw/experts_15_09_25_ctf_filtered.csv", index=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Filters
    """
    )
    return


@app.cell
def _(pd):
    working_df = pd.read_csv("data/raw/experts_15_09_25_ctf_filtered.csv")
    working_df.head()
    return (working_df,)


@app.cell
def _(Chem, working_df):
    # compute mean number of atoms for substrates and nodes
    working_df["molecule_substrate"] = working_df["smiles_substrate"].apply(Chem.MolFromSmiles)
    working_df["molecule_node"] = working_df["smiles_node"].apply(Chem.MolFromSmiles)
    mean_atoms_substrate = working_df["molecule_substrate"].apply(lambda x: x.GetNumAtoms() if x is not None else 0).mean()
    mean_atoms_node = working_df["molecule_node"].apply(lambda x: x.GetNumAtoms() if x is not None else 0).mean()
    mean_atoms_substrate, mean_atoms_node
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Flatness
    """
    )
    return


@app.cell
def _():
    from modules.core.filters.flatness_filter import FlatnessFilter

    flatness_filter = FlatnessFilter()
    return (flatness_filter,)


@app.cell
def _(flatness_filter, working_df):
    _, flatness_substrate = flatness_filter.apply(working_df["molecule_substrate"].tolist(), return_flatness=True)
    _, flatness_node = flatness_filter.apply(working_df["molecule_node"].tolist(), return_flatness=True)
    return flatness_node, flatness_substrate


@app.cell(hide_code=True)
def _(math, np, plt):
    def plot_flatness(flatness_list, name):
        flatness = [f[0] for f in flatness_list if f[0] is not None and not math.isnan(f[0])]

        # boxplot of flatness
        plt.figure(figsize=(5, 5))
        plt.boxplot(flatness, vert=True)
        # add median, mean, min, max, quantiles
        # and add values to the plot
        plt.plot([1], [np.mean(flatness)], "ro", label=f"Mean: {np.mean(flatness):.2f}")
        plt.plot([1], [np.median(flatness)], "bo", label=f"Median: {np.median(flatness):.2f}")
        plt.plot([1], [np.min(flatness)], "go", label=f"Min: {np.min(flatness):.2f}")
        plt.plot([1], [np.max(flatness)], "yo", label=f"Max: {np.max(flatness):.2f}")
        plt.plot([1], [np.quantile(flatness, 0.25)], "co", label=f"1Q: {np.quantile(flatness, 0.25):.2f}")
        plt.plot([1], [np.quantile(flatness, 0.75)], "mo", label=f"3Q: {np.quantile(flatness, 0.75):.2f}")
        plt.legend()
        plt.xticks([1], ["Flatness"])
        plt.ylabel("Flatness")
        plt.title(f"Flatness of molecules {name}")
        plt.show()

    return (plot_flatness,)


@app.cell
def _(flatness_node, flatness_substrate, plot_flatness):
    plot_flatness(flatness_substrate, "substrate"), plot_flatness(flatness_node, "node")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Conjugation & steric hindrance
    """
    )
    return


@app.cell
def _():
    from modules.core.filters.conjugation_filter import ConjugationFilter
    from modules.core.filters.steric_hindrance_filter import StericHindranceFilter

    conjugation_filter = ConjugationFilter()
    steric_hindrance_filter = StericHindranceFilter()
    return conjugation_filter, steric_hindrance_filter


@app.cell
def _(conjugation_filter, steric_hindrance_filter, working_df):
    def is_molecule_xyz(mol_object, xyz_set):
        if mol_object is None:
            return False
        return mol_object in xyz_set

    for col in ["substrate", "node"]:
        all_mol_objects_from_df = [mol for mol in working_df[f"molecule_{col}"].tolist() if mol is not None]

        conjugated_mol_objects_list = conjugation_filter.apply(all_mol_objects_from_df)
        steric_mol_objects_list = steric_hindrance_filter.apply(all_mol_objects_from_df)

        print(f"# conjugated molecules {col}: {len(conjugated_mol_objects_list)}")
        print(f"# steric molecules {col}: {len(steric_mol_objects_list)}")

        conjugated_set = set(conjugated_mol_objects_list)
        steric_set = set(steric_mol_objects_list)

        working_df[f"conjugated_{col}"] = working_df[f"molecule_{col}"].apply(lambda x: is_molecule_xyz(x, conjugated_set))
        working_df[f"steric_{col}"] = working_df[f"molecule_{col}"].apply(lambda x: is_molecule_xyz(x, steric_set))
    return


@app.cell
def _(working_df):
    # check non-conjugated substrates
    working_df[not working_df["conjugated_substrate"]]["smiles_substrate"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Symmetry (CSM)
    """
    )
    return


@app.cell
def _(working_df):
    from tqdm import tqdm

    from modules.core.features.csm_runner import CSMRunner

    runner = CSMRunner()

    measures = []
    measures_normalized = []

    for sml in tqdm(working_df["smiles_substrate"]):
        results = runner.analyze_molecule(sml, ["c2", "c3", "c4"], exact=False)
        print(results)
        if results is None:
            continue
        else:
            measures.append(results.lowest_csm)
            measures_normalized.append(results.lowest_csm_normalized)
    return measures, measures_normalized


@app.cell
def _(measures, measures_normalized, np, plt):
    scores = [m[1] for m in measures]
    scores_normalized = [m[1] for m in measures_normalized]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # histogram of scores
    axes[0].hist(scores, bins=30)
    axes[0].set_xlabel("CSM Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(
        f"CSM: min={min(scores):.2f}, max={max(scores):.2f}, mean={np.mean(scores):.2f},\n median={np.median(scores):.2f}, 1Q={np.quantile(scores, 0.25):.2f}, 3Q={np.quantile(scores, 0.75):.2f}"
    )
    # histogram of normalized scores
    axes[1].hist(scores_normalized, bins=30)
    axes[1].set_xlabel("Normalized CSM Score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(
        f"Normalized CSM: min={min(scores_normalized):.2f}, max={max(scores_normalized):.2f}, mean={np.mean(scores_normalized):.2f}, \nmedian={np.median(scores_normalized):.2f}, 1Q={np.quantile(scores_normalized, 0.25):.2f}, 3Q={np.quantile(scores_normalized, 0.75):.2f}"
    )
    plt.show()
    return (scores,)


@app.cell
def _(plt, scores):
    from modules.generation.utils import score_value_exponential

    new_scores = [score_value_exponential(s, min_val=1e-10, max_val=0.2, decay_rate=0.1) for s in scores]

    # histogram of scores
    plt.figure(figsize=(5, 5))
    plt.hist(new_scores, bins=30)
    plt.xlabel("Normalized Score")
    plt.ylabel("Frequency")
    plt.title("Histogram of Norm Scores")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Summary
    """
    )
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, plt):
    # show capacities based on CTF column
    import seaborn as sns

    plt.figure(figsize=(3, 5))
    sns.boxplot(x="CTF", y="capacitance_max", data=df)
    plt.xlabel("CTF")
    plt.ylabel("Max Capacitance (F/g)")
    plt.title("Max Capacitance vs CTF")
    plt.show()

    # print basic statistics of capacitance_max based on CTF column
    df.groupby("CTF")["capacitance_max"].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Property evaluator
    """
    )
    return


@app.cell
def _():
    # from modules.generation.property_evaluator import PropertyEvaluator

    # evaluator = PropertyEvaluator()

    # for sml in df_1["smiles_substrate"]:
    #     evaluator.evaluate(sml)
    return


@app.cell
def _():
    #
    return


if __name__ == "__main__":
    app.run()
