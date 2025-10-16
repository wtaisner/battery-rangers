import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re

    import lets_plot as lp
    import marimo as mo
    import pandas as pd

    return json, lp, mo, pd, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # MolAIR training data

    **DISCLAIMER** See `notebooks/chembl.py` for information on how to obtain `selfies_config.json`.
    """
    )
    return


@app.cell
def _(json, pd):
    with open("models/mol_air/vanilla/vocab.json", "r") as f_1:
        config = json.load(f_1)
    selfies = pd.read_csv("data/raw/substrate/train_selfies.slf", sep=" ", header=None)
    selfies.columns = ["selfies"]
    selfies.shape, config
    return config, selfies


@app.cell
def _(pd, re, selfies):
    lengths = []
    set_of_tokens = set()
    vanilla_vocab_compliant = []
    for selfie in selfies["selfies"]:
        tokens = re.findall(r"\[[^\]]+\]", selfie)
        set_of_tokens.update(tokens)
        lengths.append(len(tokens))
        if len(tokens) <= 75:
            vanilla_vocab_compliant.append(selfie)

    pd.DataFrame(lengths, columns=["lengths"]).describe()
    return lengths, vanilla_vocab_compliant


@app.cell
def _(lengths, lp):
    plot = lp.ggplot({"lengths": lengths}, lp.aes(x=lengths)) + lp.geom_histogram()

    plot
    return


@app.cell
def _(pd, vanilla_vocab_compliant):
    pd.DataFrame(vanilla_vocab_compliant).to_csv("data/raw/substrate/train_substrate_vanilla_vocab_compliant.slf", sep=" ", index=None, header=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# CHEMBL35 vocab compliance""")
    return


@app.cell
def _(config, pd, re):
    chembl_selfies = pd.read_csv("data/chembl_35_sqlite/chembl_35_selfies.txt", sep=" ", header=None)
    chembl_selfies.columns = ["selfies"]
    print(chembl_selfies.shape)

    def check_vocab_compliance(selfies: str, vocab: dict) -> bool:
        """
        Check if the given SELFIES string is compliant with the provided vocabulary.

        Args:
            selfies (str): The SELFIES string to check.
            vocab (dict): The vocabulary dictionary containing valid tokens.

        Returns:
            bool: True if the SELFIES string is compliant, False otherwise.
        """
        # Extract tokens from the SELFIES string
        tokens = re.findall(r"\[[^\]]+\]", selfies)

        # Check if all tokens are in the vocabulary
        return all(token in vocab for token in tokens)

    chembl_selfies["vocab_compliant"] = chembl_selfies["selfies"].apply(lambda x: check_vocab_compliance(x, config["vocabulary"]))
    return (chembl_selfies,)


@app.cell
def _(chembl_selfies, re):
    chembl_selfies["num_tokens"] = chembl_selfies["selfies"].apply(lambda x: len(re.findall(r"\[[^\]]+\]", x)))
    # describe num_tokens only for vocab compliant selfies
    chembl_selfies[chembl_selfies["vocab_compliant"]]["num_tokens"].describe()
    return


@app.cell
def _(chembl_selfies, lp):
    # plot distribution of num_tokens and lengths
    lp.ggplot({"num_tokens": chembl_selfies[chembl_selfies["vocab_compliant"]]["num_tokens"]}, lp.aes(x="num_tokens")) + lp.geom_histogram()
    return


@app.cell
def _(chembl_selfies):
    # filter lengths <= 311
    chembl_selfies[chembl_selfies["vocab_compliant"] & (chembl_selfies["num_tokens"] <= 311)].to_csv("data/chembl_35_sqlite/chembl_35_selfies_voc_compliant.txt", sep=" ", index=None, header=None)
    return


if __name__ == "__main__":
    app.run()
