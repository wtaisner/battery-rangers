import marimo

__generated_with = "0.13.1"
app = marimo.App(width="full", sql_output="pandas")


@app.cell
def _():
    import lets_plot as lp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import selfies as sf
    from rdkit import Chem

    return Chem, lp, mo, np, pd, plt, sf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Expert SMILES -> SELFIES""")
    return


@app.cell
def _(pd):
    df = pd.read_excel("data/raw/experts_merged.xlsx", sheet_name="Sheet1")
    return (df,)


@app.cell
def _(df):
    # check for duplicates
    duplicates = df[df.duplicated(subset=["canon_smiles"], keep=False)]
    duplicates = duplicates.sort_values(by=["canon_smiles"])
    duplicates
    return


@app.cell
def _(Chem, df):
    print(df.shape)
    df_1 = df[df["SMILES to be used"].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    df_1.drop(columns=["smiles", "canon_smiles", "molecule", "Komentarz"], inplace=True)
    df_1.columns = ["dataset", "capacity_max", "smiles"]
    df_1["canon_smiles"] = df_1["smiles"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    df_1 = df_1.drop_duplicates(subset=["canon_smiles"], keep="first")
    df_1 = df_1[df_1["dataset"].apply(lambda x: "expert" in x)]
    print(df_1.shape)
    df_1.head()
    return (df_1,)


@app.cell
def _(Chem, df_1, sf):
    df_1["selfies"] = df_1["canon_smiles"].apply(sf.encoder)
    df_1["selfies"].to_csv("data/raw/experts_merged.slf", sep=" ", index=None, header=None)
    df_1["canon_smiles"].to_csv("data/raw/experts_merged.smi", sep=" ", index=None, header=None)
    df_1["molecule"] = df_1["canon_smiles"].apply(Chem.MolFromSmiles)
    df_1.to_csv("data/raw/experts_merged.csv", index=None)
    df_1.head(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Flatness""")
    return


@app.cell
def _(df_1):
    from modules.core.filters.flatness_filter import FlatnessFilter

    flatness_filter = FlatnessFilter()
    filtered = flatness_filter.apply(df_1["molecule"].tolist(), return_flatness=True)
    return (filtered,)


@app.cell
def _(filtered, np, plt):
    flatness = [f[0] for f in filtered]

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
    plt.title("Flatness of molecules")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Symmetry""")
    return


@app.cell
def _(Chem, pd):
    smiles_1 = pd.read_csv("data/raw/experts_merged.smi", sep=" ", header=None)
    smiles_1.columns = ["smiles"]
    smiles_1["smiles"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    smiles_1.shape
    return (smiles_1,)


@app.cell
def _(smiles_1):
    smiles_2 = smiles_1.drop_duplicates(subset=["smiles"], keep="first")
    smiles_2.shape
    return (smiles_2,)


@app.cell
def _(Chem, lp, np, pd, smiles_2):
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    from rdkit.Chem import Draw
    from tqdm import tqdm

    from modules.core.features.utils import get_pymatgen_molecule_from_smiles
    from modules.core.filters import SymmetryFilter

    symmetry_filter = SymmetryFilter()

    def molecule_to_dataframe(molecule):
        """Extracts X, Y coordinates and symbols into a pandas DataFrame."""
        data = []
        for site in molecule:
            data.append({"x": site.coords[0], "y": site.coords[1], "symbol": site.specie.symbol, "z": site.coords[2]})
        return pd.DataFrame(data)

    def create_molecule_plot(df, title, color_map):
        """Creates a lets_plot scatter plot for a molecule DataFrame."""
        unique_symbols = df["symbol"].unique()
        current_plot_color_map = {sym: color_map.get(sym, color_map.get("DEFAULT", "gray")) for sym in unique_symbols}
        plot = (
            lp.ggplot(df, lp.aes(x="x", y="y", z="z"))
            + lp.geom_point(lp.aes(color="symbol"), size=3, alpha=0.8)
            + lp.scale_color_manual(name="Element", values=current_plot_color_map)
            + lp.coord_fixed(ratio=1)
            + lp.ggtitle(title)
            + lp.xlab("X Coordinate (Å)")
            + lp.ylab("Y Coordinate (Å)")
            + lp.ggsize(width=4, height=4)
            + lp.theme_minimal()
        )
        return plot

    def create_image_plot(pil_image, title=""):
        """Creates a lets_plot ggplot object displaying a PIL image."""
        img_array = np.array(pil_image)
        plot = lp.ggplot() + lp.geom_imshow(image_data=img_array) + lp.ggtitle(title) + lp.ggsize(width=2, height=2) + lp.theme_void()
        return plot

    element_colors = {
        "H": "lightgray",
        "C": "black",
        "N": "blue",
        "O": "red",
        "F": "lightgreen",
        "Cl": "green",
        "Br": "darkblue",
        "I": "purple",
        "S": "yellow",
        "P": "orange",
        "Si": "darkcyan",
        "B": "pink",
        "DEFAULT": "gray",
    }
    plots_to_show = []
    og_point_groups = []
    for index, row_1 in tqdm(smiles_2.iterrows(), total=smiles_2.shape[0], desc="Processing SMILES"):
        smiles_str = row_1["smiles"]
        try:
            molecule_original = get_pymatgen_molecule_from_smiles(smiles_str)
            rdkit_mol = Chem.MolFromSmiles(smiles_str)
            symmetrical = symmetry_filter.apply([rdkit_mol])
            if len(symmetrical) == 0:
                passed_symmetry = "no"
            else:
                passed_symmetry = "yes"
            df_original = molecule_to_dataframe(molecule_original)
            pga = PointGroupAnalyzer(molecule_original)
            og_point_group = pga.get_pointgroup()
            og_point_groups.append(str(og_point_group))
            title_orig = f"Graph: {passed_symmetry} Point: {og_point_group}"
            plot_original = create_molecule_plot(df_original, title_orig, element_colors)
            img_rdkit_2d = Draw.MolToImage(rdkit_mol)
            plot_rdkit_2d = create_image_plot(img_rdkit_2d)
            grid_plot = lp.gggrid([plot_rdkit_2d, plot_original], ncol=2, fit=True)
            plots_to_show.append((smiles_str, grid_plot))
        except ValueError as ve:
            print(f"Error processing SMILES '{smiles_str}': {ve}")
            continue
        except IndexError as ie:
            print(f"Index error processing SMILES '{smiles_str}': {ie}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred processing SMILES '{smiles_str}': {e}")
            continue
    return og_point_groups, plots_to_show


@app.cell
def _(plots_to_show):
    plots_to_show

    # optional for more compact view / plots in the loop, as plot.show() won't work
    # mo.vstack([plot for sml, plot in plots_to_show])
    return


@app.cell
def _(og_point_groups):
    from collections import Counter

    Counter(og_point_groups).most_common(10)
    return


if __name__ == "__main__":
    app.run()
