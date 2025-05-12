"""Script to evaluate molecule generation metrics."""
import argparse

import pandas as pd

from modules.generation.evaluation import MoleculeGenerationEvaluator

parser = argparse.ArgumentParser(description="Evaluate molecule generation metrics.")
parser.add_argument(
    "--generated_smiles",
    type=str,
    required=True,
    help="Path to the generated SMILES file.",
)
parser.add_argument(
    "--training_smiles",
    type=str,
    required=True,
    help="Path to the training SMILES file.",
)
parser.add_argument(
    "--reference_smiles",
    type=str,
    required=True,
    help="Path to the reference SMILES file.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    generated_smiles = pd.read_csv(args.generated_smiles)["SMILES"].tolist()

    # check if file is .smi
    if args.training_smiles.endswith(".smi"):
        training_smiles = pd.read_csv(args.training_smiles, header=None)[0].tolist()
    else:
        training_smiles = pd.read_csv(args.training_smiles)["canonical_smiles"].tolist()

    reference_smiles = pd.read_csv(args.reference_smiles)["canon_smiles"].tolist()

    evaluator = MoleculeGenerationEvaluator(
        generated_smiles=generated_smiles,
        training_smiles=training_smiles,
        reference_smiles=reference_smiles,
        n_jobs=4,  # Adjust based on CPU cores
        device="cpu",  # Use the determined device
    )

    # --- Evaluate All Metrics ---

    all_results = evaluator.evaluate(log_wandb=True, log_examples=True)
