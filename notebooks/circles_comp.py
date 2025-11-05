import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from modules.generation.evaluation import MoleculeGenerationEvaluator

    return MoleculeGenerationEvaluator, pd


@app.cell
def _(pd):
    generated_smiles = pd.read_csv("modules/bionemo/data/outputs/molrl/MolMIM_test_seed_molecules_100_epochs_lr_0_0005_bs_200.csv")["canon_smiles"].tolist()
    len(generated_smiles)
    return (generated_smiles,)


@app.cell
def _(MoleculeGenerationEvaluator, generated_smiles):
    generated_smiles_example = [
        "CCO",  # duplicate of training / generated
        "CCC",  # novel, valid
        "c1ccccc1",  # duplicate of training
        # "invalid-smiles-string",  # invalid
        "CC(=O)O",  # duplicate of training
        "CCO",  # duplicate of training / generated
        "N#CC1=CC=C(C#N)C=C1",
    ]

    training_smiles_example = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic Acid
        "C",  # Methane
        "CC",  # Ethane
        "c1ccccc1",  # Benzene (make one generated mol non-novel)
        "CC(C)C",  # Isobutane
    ]

    # Reference set (more diverse, drug-like subset)
    reference_smiles_example = [
        "O=C=O",  # Carbon Dioxide
        "c1ccc(C(=O)O)cc1",  # Benzoic Acid (canonical)
        "Nc1ccccc1",  # Aniline
    ]

    evaluator = MoleculeGenerationEvaluator(
        generated_smiles=generated_smiles,
        # training_smiles=training_smiles_example,
        # reference_smiles=reference_smiles_example,
        n_jobs=4,  # Adjust based on CPU cores
        device="cpu",  # Use the determined device
    )

    circles_value, org_indices = evaluator.calculate_circles_metric(distance_threshold=0.75)
    return evaluator, org_indices


@app.cell
def _(Chem, evaluator, org_indices):
    # plot the NxN grid of most diverse molecules based on results, evaluator.valid_generated_smiles_canon, and org_indices list
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw

    diverse_mols = [Chem.MolFromSmiles(evaluator.valid_generated_smiles_canon[i]) for i in org_indices]
    img = Draw.MolsToGridImage(diverse_mols[:50], molsPerRow=10)
    img
    return


@app.cell
def _():
    import random

    import more_itertools as mit
    import numpy as np
    import rdkit
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    from tqdm import tqdm
    from tqdm.contrib.concurrent import process_map

    def get_circles(args, silent=True):
        # 'sim_mat_func' has been removed from the arguments
        vecs, t = args

        circs = []
        for vec in tqdm(vecs, disable=silent):
            if len(circs) > 0:
                # Instead of using a passed-in function, we calculate the distances
                # directly inside the worker. This is the core of the fix.
                # We find the single closest similarity and convert it to a distance.
                min_dist = 1.0 - max(DataStructs.TanimotoSimilarity(vec, c) for c in circs)

                # The original logic of your algorithm is preserved
                if min_dist <= t:
                    continue
            circs.append(vec)
        return circs

    class NCircles:
        def __init__(self, vectorizer=None, threshold=0.75, *args, **kargs):
            self.vectorizer = vectorizer
            self.t = threshold
            self.vecs = []

        def measure(self, mols=[], is_vec=False, n_chunk=64):
            if is_vec:
                vecs = mols
            else:
                vecs = self.vectorizer(mols)

            for i in range(3):
                vecs_list = [list(c) for c in mit.divide(n_chunk // (2**i), vecs)]

                # --- MODIFICATION 2: The Arguments for process_map ---
                # We no longer include the non-pickleable 'self.sim_mat_func'
                args = zip(vecs_list, [self.t] * len(vecs_list))

                # Now this call is safe, as it only sends pickleable objects
                circs_list = process_map(get_circles, args)

                vecs = [c for ls in circs_list for c in ls]
                random.shuffle(vecs)

            # The final serial call also needs its arguments adjusted
            vecs = get_circles((vecs, self.t), silent=False)
            return len(vecs), vecs

        def update(self, mols=[], is_vec=False):
            if is_vec:
                vecs = mols
            else:
                vecs = self.vectorizer(mols)
            for vec in tqdm(vecs):
                if len(self.vecs) > 0:
                    dists = 1.0 - self.sim_mat_func([vec], self.vecs)
                    if dists.min() <= self.t:
                        continue
                self.vecs.append(vec)

        def report(self):
            return len(self.vecs)

    return Chem, GetMorganFingerprintAsBitVect, NCircles


@app.cell
def _(Chem, GetMorganFingerprintAsBitVect, NCircles, evaluator):
    circles = NCircles(vectorizer=lambda x: [GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in x], threshold=0.75)
    circles.measure(evaluator.valid_generated_smiles_canon, False, len(evaluator.valid_generated_smiles_canon))
    return


if __name__ == "__main__":
    app.run()
