import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Sampling-based set extension data filtering & split
    This notebook assumes that
    1. data is already sampled
    2. the `scripts/sampling-based-database-retrieval.py` code had already been run, meaning that there exists a file with all unique, canon SMILES sampled.
    """
    )
    return


@app.cell
def _():
    import re

    import marimo as mo
    import pandas as pd
    from sklearn.model_selection import train_test_split

    prior = "vanilla"
    # prior = "ChEMBL35"

    molecule_type = "node"
    # molecule_type = "substrate"

    # optional suffix
    suffix = "_structure_reward"

    path = f"data/discovered/molecules_passing_filters_{molecule_type}{suffix}.smi"
    return mo, path, pd, prior, re, train_test_split


@app.cell
def _(path, pd):
    sampled_df = pd.read_csv("node-sampling-based-properties.csv")
    sampled_df.head(), sampled_df.shape

    passing_filters = (
        sampled_df.query("smarts_filter >= 0.99")
        .query("conjugation_filter >= 0.99")
        .query("flatness <= 5.0")  # TODO: change for substrate to 4.0
        .query("normalized_csm <= 0.2")
        .query("steric_hindrance >= 0.99")
    )

    passing_filters["canon_smiles"].to_csv(path, index=False, header=False)

    passing_filters.head(), passing_filters.shape
    return (passing_filters,)


@app.cell
def _(passing_filters, path, prior, re, train_test_split):
    if prior == "ChEMBL35":
        allowed_tokens = {
            "[NH2+]",
            "9",
            "(",
            "8",
            "S",
            "[S@]",
            "[SH]",
            "Br",
            "[CH2-]",
            "[SH2]",
            "[cH-]",
            "[n+]",
            "#",
            "[N@@+]",
            "[S-]",
            "%10",
            "[C-]",
            "[N@+]",
            "[NH-]",
            "[S@@]",
            "3",
            "4",
            ")",
            "6",
            "Cl",
            "-",
            "[C+]",
            "/",
            "%12",
            "=",
            "I",
            "[c+]",
            "[n-]",
            "\\",
            "7",
            "[O]",
            "1",
            "[o+]",
            "[N@@]",
            "[N-]",
            "[O+]",
            "[C]",
            "[C@H]",
            "[CH]",
            "[NH+]",
            "[O-]",
            "[s+]",
            "[SH+]",
            "F",
            "O",
            "n",
            "[S+]",
            "$",
            "[N@]",
            "[CH-]",
            "c",
            "[S@@+]",
            "s",
            "C",
            "[C@@H]",
            "[N+]",
            "%11",
            "^",
            "[C@@]",
            "o",
            "2",
            "N",
            "[S@+]",
            "[CH2]",
            "[C@]",
            "[c-]",
            "5",
            "[nH]",
        }
    elif prior == "vanilla":
        allowed_tokens = {
            "1",
            "6",
            "O",
            "$",
            "=",
            "^",
            "[N+]",
            "[nH]",
            "2",
            "Cl",
            "F",
            "Br",
            "4",
            "o",
            "[N-]",
            "[O-]",
            "5",
            "8",
            "N",
            "S",
            "[n+]",
            "%10",
            "C",
            "s",
            "n",
            "-",
            "9",
            "3",
            "7",
            "c",
            "#",
            "[S+]",
            "(",
            ")",
        }
    else:
        raise ValueError(f"Unknown prior: {prior}")

    def reinvent_tokenize(smiles):
        """
        The exact regex used by REINVENT4/reinvent-models SMILESTokenizer.
        This treats [bracketed_atoms], Cl, Br, and %nn as single tokens.
        """
        # Note: the order of the regex components matters!
        pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smiles)]
        return tokens

    def is_valid_reinvent_molecule(smiles, vocabulary):
        """Checks if every token in the SMILES exists in the model's vocabulary."""
        tokens = reinvent_tokenize(smiles)
        for t in tokens:
            if t not in vocabulary:
                # print(f"Rejected due to token: {t}") # Debugging
                return False
        return True

    # 1. Load Data
    print(f"Total molecules: {len(passing_filters)}")

    # 2. Filter using REINVENT4-style tokenization
    df = passing_filters[passing_filters["canon_smiles"].apply(lambda x: is_valid_reinvent_molecule(x, allowed_tokens))]
    print(f"Molecules after filtering: {len(df)}")

    # 3. Split and Save
    if not df.empty:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df["canon_smiles"].to_csv(path.replace(".smi", f"_{prior}_train.smi"), index=False, header=False)
        test_df["canon_smiles"].to_csv(path.replace(".smi", f"_{prior}_test.smi"), index=False, header=False)
        print("Train shape:", train_df.shape)
        print("Test shape:", test_df.shape)
    return


if __name__ == "__main__":
    app.run()
