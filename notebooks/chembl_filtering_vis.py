import marimo

__generated_with = "0.17.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Screening large chemical databases for candidate molecules
    **Author:** Witold Taisner, 3826

    ## Motivation
    Materials discovery is the scientific endeavor of identifying and designing new materials with novel or enhanced properties that can drive technological innovation. This process is fundamental to advancements across a vast range of fields, including renewable energy, electronics, medicine, and sustainable manufacturing. Historically, the discovery of new materials often relied on laborious trial-and-error experimentation. However, the field is undergoing a significant transformation, with modern approaches leveraging computational simulations, artificial intelligence (AI), and large-scale data analysis to accelerate the discovery timeline. This shift towards data-driven and computational methods allows scientists to screen thousands of potential material compositions and predict their properties before synthesizing them in a lab, making the process more efficient and targeted

    ###ChEMBL: A Valuable Resource for Material Discovery

    ChEMBL is a large, open-access, and manually curated database of bioactive molecules with drug-like properties, it contains a wealth of information on chemical compounds and their biological activities. While its primary focus is on drug discovery, ChEMBL's extensive and well-structured data is increasingly valuable for materials science.

    The database can be instrumental in the search for new functional organic materials. Researchers can mine ChEMBL for molecules with specific structural motifs, physicochemical properties, or recorded activities that might be relevant to a material's desired function. Its data is formatted to be suitable for computational analysis and machine learning, which aligns with the modern, data-driven paradigm of materials discovery.

    For instance, I am looking for conjugated covalent triazine frameworks (CTFs) and use ChEMBL to perform substructure searches. CTFs are a class of porous organic materials built from aromatic 1,3,5-triazine rings, known for their exceptional stability and semiconducting properties. These characteristics make them promising candidates for applications as supercapacitors. By querying ChEMBL for molecules containing the triazine core and specific connectivity patterns, a researcher could identify potential building blocks or fragments for designing new CTFs with tailored properties.

    Furthermore, ChEMBL is publicly available via https://www.ebi.ac.uk/chembl/visualise/ . For the sake of computational efficiency, a reasonable subset of 100k molecules is used to make computations feasible, with random seed fixed for reproducibility. Lastly, custom implementation of substructure search (SMARTSFilter) and conjugation estimation (ConjugationFilter) are employed to identify relevant molecules and are available via: implementations of both filters are available via: https://github.com/wtaisner/battery-rangers/tree/main/modules/core/filters. For the sake of readabilty, implementation is not included.

    The report has been generated as Marimo nodebook and apart from resulting HTML, .py source file is provided as well. In order to run it:
    ```python
    marimo run visualization_Witold_Taisner_3826.py
    ```

    Marimo can be installed via pip or uv, e.g. `uv add marimo`
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw

    from modules.core.filters.conjugation_filter import ConjugationFilter
    from modules.core.filters.smarts_filter import SMARTSFilter

    ctf_filter = SMARTSFilter()
    conjugation_filter = ConjugationFilter()
    return (
        AllChem,
        Chem,
        Draw,
        conjugation_filter,
        ctf_filter,
        mo,
        pd,
        plt,
        sns,
    )


@app.cell
def _(pd):
    df = pd.read_parquet("data/chembl_35_sqlite/chembl_35.parquet").sample(100000, random_state=42)
    df.rename(columns={"smiles": "canonical_smiles"}, inplace=True)
    return (df,)


@app.cell
def _(Chem, conjugation_filter, ctf_filter, df):
    df["molecule"] = df["canonical_smiles"].apply(Chem.MolFromSmiles)
    df["CTF"] = df["molecule"].apply(lambda x: True if len(ctf_filter.apply([x])) > 0 else False)
    df["conjugated"] = df["molecule"].apply(lambda x: True if len(conjugation_filter.apply([x])) > 0 else False)
    return


@app.cell
def _(df):
    # we can discard unnecessary columns to make the dataframe cleaner.
    df.drop(
        columns=["molregno", "molfile", "standard_inchi", "standard_inchi_key", "selfies", "molecule", "cx_most_apka", "cx_most_bpka"],
        inplace=True,
        errors="ignore",
    )
    df.head()
    return


@app.cell
def _(df):
    df["CTF"].value_counts(), df["conjugated"].value_counts()
    return


@app.cell
def _(df):
    # both CTF and conjugated
    df[(df["CTF"] & df["conjugated"])].shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can see that the number of CTFs in the data is highly limited, however a significant portion of them is conjugated. Let's visualize one of the CTF molecules in 3D to better understand its structure."""
    )
    return


@app.cell
def _(AllChem, Chem, Draw, plt):
    def plot_molecule_grid(smiles_string: str, title: str = ""):
        """
        Generates and displays a grid of molecule visualizations.
        Left: 2D molecule graph.
        Right: 3D scatter plot of atom conformations.

        Args:
            smiles_string (str): The SMILES representation of the molecule.
            title (str, optional): The main title for the figure.
        """
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            print("Error: Invalid SMILES string.")
            return

        mol = Chem.AddHs(mol)

        conf_id = AllChem.EmbedMolecule(mol, randomSeed=42)  # Use a seed for reproducibility
        if conf_id == -1:
            print("Error: Conformer generation failed.")
            return

        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=0)
        except Exception as e:
            print(f"Warning: Could not optimize molecule geometry. Error: {e}")

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        fig.suptitle(title if title else f"Molecule: {smiles_string}", fontsize=16)

        img = Draw.MolToImage(mol, size=(300, 300))
        ax1.imshow(img)
        ax1.set_title("2D Structure")
        ax1.axis("off")
        conformer = mol.GetConformer(0)
        positions = conformer.GetPositions()

        atom_colors = {
            "C": "black",
            "H": "lightgray",
            "N": "blue",
            "O": "red",
            "S": "yellow",
            "F": "green",
            "Cl": "lime",
            "Br": "darkred",
            "I": "purple",
            "P": "orange",
        }

        for i, atom in enumerate(mol.GetAtoms()):
            x, y, z = positions[i]
            symbol = atom.GetSymbol()
            color = atom_colors.get(symbol, "gray")  # Default to gray
            ax2.scatter([x], [y], [z], s=150, c=color, alpha=0.9, edgecolors="w", linewidth=0.5)

        for bond in mol.GetBonds():
            start_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            pos_start = positions[start_atom_idx]
            pos_end = positions[end_atom_idx]
            ax2.plot([pos_start[0], pos_end[0]], [pos_start[1], pos_end[1]], [pos_start[2], pos_end[2]], color="dimgray", linewidth=2, zorder=-1)

        ax2.set_title("3D Conformation (Atoms)")
        ax2.set_xlabel("X (Å)")
        ax2.set_ylabel("Y (Å)")
        ax2.set_zlabel("Z (Å)")
        ax2.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return (plot_molecule_grid,)


@app.cell
def _(plot_molecule_grid, working_df):
    # plot 3D conformation of a random CTF conjugated molecule
    random_ctf_smiles = working_df[(working_df["CTF"] & working_df["conjugated"])]["canonical_smiles"].sample(100, random_state=42).values[-1]
    plot_molecule_grid(random_ctf_smiles, title=f"3D Conformation of {random_ctf_smiles}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Furthermore, we can check for correlation between numeric columns, with some visible high values between properties directly dependend on number of atoms, i.e. molecular weight, etc., which is to be expected."""
    )
    return


@app.cell
def _(df):
    working_df = df[["canonical_smiles", "CTF", "conjugated", "mw_freebase", "alogp", "psa", "molecular_species", "full_mwt", "aromatic_rings", "heavy_atoms", "qed_weighted"]]
    numeric_cols = working_df.select_dtypes(include=["number"]).columns
    return numeric_cols, working_df


@app.cell
def _(numeric_cols, plt, sns, working_df):
    # select numeric columns
    correlation_matrix = working_df[numeric_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation matrix between numeric features")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We want to take a look at how some features correspond directly to conjugation or CTF.""")
    return


@app.cell
def _(sns, working_df):
    sns.jointplot(data=working_df, x="qed_weighted", y="heavy_atoms", hue="conjugated"), sns.jointplot(data=working_df, x="psa", y="alogp", hue="CTF")
    return


@app.cell
def _(numeric_cols, plt, sns, working_df):
    # pairplot of numeric columns + CTF
    from sklearn.model_selection import train_test_split

    # we want to make sure that CTF will be represented equally in the sample
    _, sample_df = train_test_split(working_df, test_size=1000, random_state=23, stratify=working_df["CTF"])
    sns.pairplot(sample_df, vars=numeric_cols, hue="CTF")
    plt.show()
    return (sample_df,)


@app.cell
def _(sample_df, sns):
    sns.violinplot(data=sample_df, x="aromatic_rings", y="alogp", hue="CTF", split=True, gap=0.1, inner="quart")
    return


@app.cell
def _(sample_df, sns):
    sns.violinplot(data=sample_df, x="aromatic_rings", y="alogp", hue="conjugated", split=True, gap=0.1, inner="quart")
    return


@app.cell
def _(numeric_cols, plt, sns, working_df):
    sns.pairplot(working_df.sample(1000, random_state=42), vars=numeric_cols, hue="conjugated")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Lastly, we can use an UpSet plot to visualize the intersections between different molecular species, conjugation status, and CTF classification. This visualization helps us understand how these categories overlap and interact within our dataset. The UpSet plot is particularly useful for identifying commonalities and unique features among the various groups of molecules, providing insights that can guide further analysis and material discovery efforts."""
    )
    return


@app.cell
def _(pd, working_df):
    from upsetplot import UpSet, from_indicators

    # adjust the data for upset plot
    molecular_species_dummies = pd.get_dummies(working_df["molecular_species"], prefix="species").astype(bool)
    upset_df = pd.concat([working_df[["CTF", "conjugated"]], molecular_species_dummies], axis=1)

    upset_data = upset_df.set_index(list(upset_df.columns))
    upset_plot_data = from_indicators(upset_df)

    UpSet(upset_plot_data, subset_size="count").plot()["matrix"]
    return


if __name__ == "__main__":
    app.run()
