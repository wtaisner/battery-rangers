"""
Script to evaluate molecule generation metrics.
Accepts a single file or a glob pattern for generated SMILES.
Logs an aggregated report to Weights & Biases for multiple files.
"""
import argparse
import glob
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from upsetplot import plot

from modules.core.enums import MoleculeType

# pylint: disable=import-error
from modules.generation.evaluation import MoleculeGenerationEvaluator

parser = argparse.ArgumentParser(description="Evaluate molecule generation metrics for one or more runs.")
parser.add_argument(
    "--generated_smiles",
    type=str,
    required=True,
    help="Path or glob pattern for the generated SMILES file(s) (e.g., 'data/run.csv' or 'data/run_*.csv').",
)
parser.add_argument(
    "--training_smiles",
    type=str,
    required=False,
    help="Path to the training SMILES file.",
)
parser.add_argument(
    "--reference_smiles",
    type=str,
    required=True,
    help="Path to the reference SMILES file.",
)
parser.add_argument(
    "--run_name",
    type=str,
    required=True,
    help="Base name for the run. Used for the aggregated wandb log.",
)

parser.add_argument("--log_wandb", action="store_true", help="Enable logging of the aggregated report to Weights & Biases.")  # Makes this a flag: --log_wandb
parser.add_argument("--wandb_project", type=str, default="molecule-generation", help="Specify the wandb project name.")
parser.add_argument("--wandb_entity", type=str, default="witold_taisner", help="Specify the wandb entity (user or team).")
# argument that if provided will set molecule type to NODE, otherwise it will be set to SUBSTRATE
parser.add_argument(
    "--molecule_type",
    type=str,
    choices=[MoleculeType.SUBSTRATE.value, MoleculeType.NODE.value],
    default=MoleculeType.SUBSTRATE.value,
    help="Type of the molecule to evaluate. Defaults to SUBSTRATE.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # --- Load Constant Reference and Training Files ---
    if not args.training_smiles:
        training_smiles = None
    elif args.training_smiles.endswith(".smi"):
        training_smiles = pd.read_csv(args.training_smiles, header=None)[0].tolist()
    else:
        training_smiles = pd.read_csv(args.training_smiles)["canonical_smiles"].tolist()

    reference_smiles = pd.read_csv(args.reference_smiles)["canon_smiles"].tolist()

    generated_files = sorted(glob.glob(args.generated_smiles))
    if not generated_files:
        raise FileNotFoundError(f"No file(s) found for the given path or pattern: {args.generated_smiles}")

    print(f"Found {len(generated_files)} file(s) to evaluate.")

    all_results_list = []
    all_upset_results = {}
    for f_path in generated_files:
        print(f"\n---> Evaluating file: {os.path.basename(f_path)}")
        start_time = time.time()
        allowed_smiles_columns = ["SMILES", "canonical_smiles", "smiles", "canon_smiles"]
        # Check if the file has any of the allowed columns
        df = pd.read_csv(f_path)  # [:1000] # TODO: remove later
        found_column = None
        for col in allowed_smiles_columns:
            if col in df.columns:
                found_column = col
                break
        if found_column is None:
            raise ValueError(f"None of the expected columns {allowed_smiles_columns} found in the file: {f_path}")

        generated_smiles = df[found_column].tolist()

        evaluator = MoleculeGenerationEvaluator(
            generated_smiles=generated_smiles,
            training_smiles=training_smiles,
            reference_smiles=reference_smiles,
            n_jobs=4,
            device="cpu",
            molecule_type=MoleculeType(args.molecule_type),  # Convert string to MoleculeType enum
        )

        # --- Evaluate metrics for the current file ---
        # Wandb logging is disabled for singular reports
        results, upset_results = evaluator.evaluate(log_wandb=False, log_examples=False)

        print(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")

        all_upset_results.update(upset_results)
        all_results_list.append(results)

    # --- Display results ---
    if not all_results_list:
        print("Evaluation failed to produce any results.")
    else:
        results_df = pd.DataFrame(all_results_list)
        results_df.index = [os.path.basename(p) for p in generated_files]

        print("\n\n" + "=" * 50)
        print("          RUN METRICS")
        print("=" * 50)
        print(results_df.round(3).to_string())

        records = []
        for smiles, filter_outcomes in all_upset_results.items():
            # Invert the boolean: True if the molecule failed (result is False)
            record = {filter_name: not passed for filter_name, passed in filter_outcomes.items()}
            records.append(record)

            # Create the DataFrame
        df_for_upset = pd.DataFrame.from_records(records).dropna()

        # --- Conditional Aggregation and Logging for multiple files ---
        if len(generated_files) > 1:
            mean_metrics = results_df.mean()
            std_metrics = results_df.std()
            summary_df = pd.DataFrame({"mean": mean_metrics, "std": std_metrics})

            print("\n\n" + "=" * 50)
            print("     AGGREGATED METRICS (MEAN and STD)")
            print("=" * 50)
            print(summary_df.round(3))

            # --- Log the aggregated report to wandb if enabled ---
            if args.log_wandb:
                print("\n---> Logging aggregated report to Weights & Biases...")
                wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, config=vars(args))  # Log script arguments for reproducibility

                # Log metrics as a flat dictionary for easy plotting in wandb
                wandb_log_dict = {}
                for metric, values in summary_df.iterrows():
                    wandb_log_dict[f"{metric}_mean"] = values["mean"]
                    wandb_log_dict[f"{metric}_std"] = values["std"]
                wandb.log(wandb_log_dict)

                # Log the summary dataframe as a wandb.Table for a nice view
                summary_table_df = summary_df.reset_index().rename(columns={"index": "metric"})
                wandb_table = wandb.Table(dataframe=summary_table_df)
                wandb.log({"aggregated_metrics_summary": wandb_table})

                wandb_upset_table = wandb.Table(dataframe=df_for_upset)
                wandb.log({"upset_plot_data": wandb_upset_table})

                # log upset plot to wandb
                fig = plt.figure(figsize=(10, 6))
                print(df_for_upset.head())
                plot_dict = plot(df_for_upset.groupby(list(df_for_upset.columns)).size(), fig=fig)
                wandb.log({"upset_plot": fig})

                wandb.finish()
                print("---> Logging complete.")

        print("\n")
