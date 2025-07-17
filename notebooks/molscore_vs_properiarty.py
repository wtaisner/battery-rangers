import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from moleval.metrics.metrics import GetMetrics
    from molscore import MockGenerator

    from modules.generation.evaluation import MoleculeGenerationEvaluator

    generated_smiles_example = [
        "CCO",  # duplicate of training / generated
        "CCC",  # novel, valid
        "c1ccccc1",  # duplicate of training
        "invalid-smiles-string",  # invalid
        "CC(=O)O",  # duplicate of training
        "CCO",  # duplicate of training / generated
    ]

    training_smiles_example = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic Acid
        "C",  # Methane
        "CC",  # Ethane
        "c1ccccc1",  # Benzene (make one generated mol non-novel)
        "CC(C)C",  # Isobutane
    ]

    test_smiles_example = ["CCOC", "c1ccccn1"]

    # Reference set (more diverse, drug-like subset)
    reference_smiles_example = [
        "O=C=O",  # Carbon Dioxide
        "c1ccc(C(=O)O)cc1",  # Benzoic Acid (canonical)
        "Nc1ccccc1",  # Aniline
    ]
    return GetMetrics, MockGenerator, MoleculeGenerationEvaluator, mo


@app.cell
def _(MockGenerator):
    mg = MockGenerator()
    GEN_SMILES = mg.sample(50)
    TRAIN_SMILES = mg.sample(500)
    TEST_SMILES = mg.sample(20)
    TARGET_SMILES = mg.sample(20)
    return GEN_SMILES, TARGET_SMILES, TEST_SMILES, TRAIN_SMILES


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Molscore""")
    return


@app.cell
def _(GEN_SMILES, GetMetrics, TARGET_SMILES, TEST_SMILES, TRAIN_SMILES):
    MetricEngine = GetMetrics(
        n_jobs=1,
        device="cpu",
        batch_size=512,
        test=TEST_SMILES,
        train=TRAIN_SMILES,
        target=TARGET_SMILES,
    )
    metrics = MetricEngine.calculate(
        GEN_SMILES,
        calc_valid=True,
        calc_unique=True,
        # unique_k=10000,
        se_k=1000,
        sp_k=1000,
        properties=True,
    )

    print(metrics.keys())

    # print only a set of keys
    for key in metrics.keys():
        if key in [
            "Validity",
            "Novelty",
            "IntDiv1",  # should be the same as custom
            # "IntDiv2", # ChemGAN paper, p = 2 in custom?
            # "FCD_test",
            "FCD_target",
            "Uniqueness",
        ]:
            print(f"{key}: {metrics[key]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Custom""")
    return


@app.cell
def _(GEN_SMILES, MoleculeGenerationEvaluator, TARGET_SMILES, TRAIN_SMILES):
    evaluator = MoleculeGenerationEvaluator(
        generated_smiles=GEN_SMILES,
        training_smiles=TRAIN_SMILES,
        reference_smiles=TARGET_SMILES,
        n_jobs=4,  # Adjust based on CPU cores
        device="cpu",  # Use the determined device
    )

    # --- Evaluate All Metrics ---

    all_results = evaluator.evaluate(log_wandb=False, log_examples=False)
    all_results
    return


if __name__ == "__main__":
    app.run()
