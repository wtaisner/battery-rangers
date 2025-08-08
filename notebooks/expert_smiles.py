import marimo

__generated_with = "0.14.15"
app = marimo.App(width="full", sql_output="pandas")


@app.cell
def _():
    import math

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
    return Chem, math, mo, np, pd, plt, sf, train_test_split


@app.cell
def _(mo):
    mo.md(r"""# 20.06.2025 - data reading / parsing""")
    return


@app.cell
def _(Chem, pd):
    df = pd.read_excel("data/raw/data_battery_materials_PRISTINE_CORRECTED_20_06_25.xlsx", sheet_name="Sheet1")
    print(df.shape)
    # edit column names: to lower, replace spaces with underscores, remove special characters
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_").str.replace("(", "").str.replace(")", "").str.replace("/", "_")

    def canonicalize_smiles_with_location(smiles_string, row_index, col_name):
        """
        Attempts to canonicalize a SMILES string.
        Prints an error with location if it fails.
        Returns canonical SMILES or None on failure.
        """
        if pd.isna(smiles_string) or not isinstance(smiles_string, str) or smiles_string.strip() == "":
            return None  # Or pd.NA, or "" depending on how you want to represent it

        try:
            mol = Chem.MolFromSmiles(smiles_string)
            if mol is None:  # RDKit can return None for invalid SMILES without raising an exception
                print(f"Niepoprawny SMILES w wierszu {row_index + 2}, kolumna '{col_name}': '{smiles_string}'")
                return None
            return Chem.MolToSmiles(mol)
        except Exception as e:
            print(f"ERROR: Exception processing SMILES at row {row_index + 2}, column '{col_name}': '{smiles_string}'. Details: {e}")
            return None

    # Identify SMILES columns
    smiles_columns = [col for col in df.columns if "smiles" in col]

    smiles_columns.remove("smiles_pore_side")

    for col_name in smiles_columns:
        for index, smiles_value in df[col_name].items():
            canonical_form = canonicalize_smiles_with_location(smiles_value, index, col_name)
            df.at[index, col_name] = canonical_form
    return df, smiles_columns


@app.cell
def _(df, smiles_columns):
    df[smiles_columns].isna().sum()
    # print all rows in which at least one is na
    na_rows = df[df[smiles_columns].isna().any(axis=1)]
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Train-Test Split | SELFIES encoding""")
    return


@app.cell
def _(Chem, df, sf):
    df_1 = df[["smiles_substrate", "smiles_to_be_used_molecules_with_a_node", "capacitance_max"]].dropna()
    df_1.rename(columns={"smiles_to_be_used_molecules_with_a_node": "smiles_node"}, inplace=True)

    df_1["selfies_substrate"] = df_1["smiles_substrate"].apply(sf.encoder)
    df_1["selfies_node"] = df_1["smiles_node"].apply(sf.encoder)

    df_1["molecule_substrate"] = df_1["smiles_substrate"].apply(Chem.MolFromSmiles)
    df_1["molecule_node"] = df_1["smiles_node"].apply(Chem.MolFromSmiles)

    df_1["selfies_substrate"].to_csv("data/raw/substrate/experts_substrate.slf", sep=" ", index=None, header=None)
    # df_1["selfies_node"].to_csv("data/raw/node/experts_node.slf", sep=" ", index=None, header=None)

    df_1["smiles_substrate"].rename("canon_smiles").to_csv("data/raw/substrate/experts_substrate.csv", index=None)
    df_1["smiles_node"].rename("canon_smiles").to_csv("data/raw/node/experts_node.csv", index=None, header=1)

    df_1.to_csv("data/raw/new_experts_merged.csv", index=None)
    return (df_1,)


@app.cell
def _(df_1, train_test_split):
    # train_test split on entire df_1
    X_train, X_test, _, _ = train_test_split(df_1, df_1["capacitance_max"], test_size=0.2, random_state=42)
    return X_test, X_train


@app.cell
def _(X_test, X_train):
    # save train-test sets
    X_train["selfies_substrate"].to_csv("data/raw/substrate/train_selfies.slf", sep=" ", index=None, header=None)
    # X_train["selfies_node"].to_csv("data/raw/node/train_selfies.slf", sep=" ", index=None, header=None)

    X_test["selfies_substrate"].to_csv("data/raw/substrate/test_selfies.slf", sep=" ", index=None, header=None)
    # X_test["selfies_node"].to_csv("data/raw/node/test_selfies.slf", sep=" ", index=None, header=None)

    X_train["smiles_substrate"].to_csv("data/raw/substrate/chembl35_train_substrate.smi", index=None, sep=" ", header=None)
    X_test["smiles_substrate"].to_csv("data/raw/substrate/chembl35_test_substrate.smi", index=None, sep=" ", header=None)

    X_train["smiles_node"].to_csv("data/raw/node/chembl35_train_node.smi", index=None, sep=" ", header=None)
    X_test["smiles_node"].to_csv("data/raw/node/chembl35_test_node.smi", index=None, sep=" ", header=None)

    X_train["smiles_substrate"].rename("canon_smiles").to_csv("data/raw/substrate/train_substrate.csv", index=None)
    X_train["smiles_node"].rename("canon_smiles").to_csv("data/raw/node/train_node.csv", index=None)

    X_test["smiles_substrate"].rename("canon_smiles").to_csv("data/raw/substrate/test_substrate.csv", index=None)
    X_test["smiles_node"].rename("canon_smiles").to_csv("data/raw/node/test_node.csv", index=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Filters/criteria
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
def _(df_1, flatness_filter):
    _, flatness_substrate = flatness_filter.apply(df_1["molecule_substrate"].tolist(), return_flatness=True)
    _, flatness_node = flatness_filter.apply(df_1["molecule_node"].tolist(), return_flatness=True)
    return flatness_node, flatness_substrate


@app.cell(hide_code=True)
def _(math, np, plt):
    def plot_flatness(flatness_list, name):
        flatness = [f[0] for f in flatness_list if not math.isnan(f[0])]

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


@app.cell
def _(mo):
    mo.md(r"""## SMARTS""")
    return


@app.cell
def _():
    from modules.core.filters.smarts_filter import SMARTSFilter

    smarts_filter = SMARTSFilter()
    return (smarts_filter,)


@app.cell
def _(df_1, smarts_filter):
    # filtered = smarts_filter.apply(df_1["molecule_substrate"])
    # len(filtered)

    for idx, mol in enumerate(df_1["molecule_substrate"]):
        filtered = smarts_filter.apply([mol])
        if not filtered:
            print(idx)
            print("---")
    return


@app.cell
def _():
    from modules.core.filters.conjugation_filter import ConjugationFilter
    from modules.core.filters.steric_hindrance_filter import StericHindranceFilter

    # from modules.core.filters.symmetry_filter import SymmetryFilter
    # from modules.core.filters.point_group_symmetry_filter import PointGroupSymmetryFilter
    # symmetry_filter = SymmetryFilter()
    conjugation_filter = ConjugationFilter()
    steric_hindrance_filter = StericHindranceFilter()
    # point_group_symmetry_filter = PointGroupSymmetryFilter()
    return (conjugation_filter,)


@app.cell
def _(conjugation_filter, df_1):
    def is_molecule_xyz(mol_object, xyz_set):
        if mol_object is None:
            return False
        return mol_object in xyz_set

    for col in ["substrate", "node"]:
        all_mol_objects_from_df = [mol for mol in df_1[f"molecule_{col}"].tolist() if mol is not None]

        # symmetrical_mol_objects_list = symmetry_filter.apply(all_mol_objects_from_df)
        conjugated_mol_objects_list = conjugation_filter.apply(all_mol_objects_from_df)
        # steric_mol_objects_list = steric_hindrance_filter.apply(all_mol_objects_from_df)
        # point_group_symmetry_mol_objects_list = point_group_symmetry_filter.apply(all_mol_objects_from_df)

        # print(f"# symmetrical molecules {col}: {len(symmetrical_mol_objects_list)}")
        print(f"# conjugated molecules {col}: {len(conjugated_mol_objects_list)}")
        # print(f"# steric molecules {col}: {len(steric_mol_objects_list)}")
        # print(f"# point group symmetry molecules {col}: {len(point_group_symmetry_mol_objects_list)}")

        # symmetrical_set = set(symmetrical_mol_objects_list)
        conjugated_set = set(conjugated_mol_objects_list)
        # steric_set = set(steric_mol_objects_list)
        # point_group_symmetry_set = set(point_group_symmetry_mol_objects_list)

        # df_1[f"point_group_symmetry_{col}"] = df_1[f"molecule_{col}"].apply(lambda x: is_molecule_xyz(x, point_group_symmetry_set))
        # df_1[f"symmetrical_{col}"] = df_1[f"molecule_{col}"].apply(lambda x: is_molecule_xyz(x, symmetrical_set))
        df_1[f"conjugated_{col}"] = df_1[f"molecule_{col}"].apply(lambda x: is_molecule_xyz(x, conjugated_set))
        # df_1[f"steric_{col}"] = df_1[f"molecule_{col}"].apply(lambda x: is_molecule_xyz(x, steric_set))
    return


@app.cell
def _(df_1):
    # check non-conjugated substrates
    df_1[df_1["conjugated_substrate"] == False]["smiles_substrate"]
    return


@app.cell
def _(df_1):
    from tqdm import tqdm

    from modules.core.features.csm_runner import CSMRunner

    runner = CSMRunner()

    measures = []

    print(df_1.columns)

    for sml in tqdm(df_1["smiles_to_be_used_molecules_with_a_node"]):
        results = runner.analyze_molecule(sml, ["c2", "c3", "c4"], exact=False)
        # print(results)
        if results is None:
            continue
        else:
            measures.append(results.lowest_csm)
    return (measures,)


@app.cell
def _(measures, plt):
    scores = [m[1] for m in measures]

    # histogram of scores
    plt.figure(figsize=(5, 5))
    plt.hist(scores, bins=10, edgecolor="black")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Histogram of Scores")
    plt.show()
    return (scores,)


@app.cell
def _(np, scores):
    np.median(scores)
    return


@app.cell
def _(plt, scores):
    from modules.generation.utils import score_value_exponential

    new_scores = [score_value_exponential(s, min_val=1e-10, max_val=5.0, decay_rate=0.1) for s in scores]

    # histogram of scores
    plt.figure(figsize=(5, 5))
    plt.hist(new_scores, bins=10, edgecolor="black")
    plt.xlabel("Normalized Score")
    plt.ylabel("Frequency")
    plt.title("Histogram of Norm Scores")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Summary""")
    return


@app.cell
def _(df_1):
    df_1
    return


@app.cell
def _(mo):
    mo.md(r"""## Property evaluator""")
    return


@app.cell
def _():
    from modules.generation.property_evaluator import PropertyEvaluator

    evaluator = PropertyEvaluator()

    # for sml in df_1["smiles_substrate"]:
    #     evaluator.evaluate(sml)
    return


@app.cell
def _():
    #
    return


if __name__ == "__main__":
    app.run()
