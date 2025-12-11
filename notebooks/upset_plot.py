import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")


@app.cell
def _():
    import glob
    import json
    import os
    import warnings

    import marimo as mo
    import pandas as pd
    from matplotlib import pyplot as plt
    from rdkit import Chem
    from upsetplot import plot

    import wandb

    warnings.filterwarnings("ignore")
    warnings.simplefilter(action="ignore", category=FutureWarning)
    return json, os, pd, plot, plt, wandb


@app.cell
def _(json, os, pd, wandb):
    def download_wandb_table(run_id: str, entity: str = "witold_taisner", project: str = "molecule-generation") -> tuple[pd.DataFrame, str] | None:
        """
        Downloads a table artifact from W&B and returns it as a pandas DataFrame.

        You can find your entity and project from the URL of your run,
        which looks like: https://wandb.ai/<entity>/<project>/runs/<run_id>

        Args:
            run_id (str): The ID of the W&B run containing the table artifact.
            entity (str): The W&B entity (user or team) name. Default is "witold_taisner".
            project (str): The W&B project name. Default is "molecule-generation".

        Returns:
            tuple[pd.DataFrame, str]: A pandas DataFrame containing the table data and a corresponding run name, or None if an error occurred.
        """
        artifact_name = f"run-{run_id.split('/')[-1]}-upset_plot_data:v0"  # extract id from full artifact path

        try:
            api = wandb.Api()

            # Construct the full path to the artifact
            artifact_path = f"{entity}/{project}/{artifact_name}"
            print(f"Fetching artifact: {artifact_path}")

            run_name = api.run(run_id).name

            # Fetch the artifact object. The type 'run_table' is based on your screenshot.
            artifact = api.artifact(artifact_path, type="run_table")

            # Download the artifact's contents to a local directory like ./artifacts/run-bky4y8eb...
            artifact_dir = artifact.download()
            print(f"Artifact downloaded to: {artifact_dir}")

            # Find the actual .table.json file within the downloaded folder
            table_json_file = None
            for file in os.listdir(artifact_dir):
                if file.endswith(".table.json"):
                    table_json_file = os.path.join(artifact_dir, file)
                    print(f"Found table file: {table_json_file}")
                    break

            if not table_json_file:
                print(f"Error: No '.table.json' file found in the artifact directory '{artifact_dir}'.")
                return None

            # Load the data from the JSON file
            with open(table_json_file, "r") as f:
                table_data = json.load(f)

            # Create a pandas DataFrame from the table's columns and data
            df = pd.DataFrame(data=table_data["data"], columns=table_data["columns"])
            return df, run_name

        except wandb.errors.CommError as e:
            print(f"Error fetching artifact: {e}")
            print("Please double-check that your 'entity', 'project', and 'artifact_name' are correct.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    return (download_wandb_table,)


@app.cell
def _(download_wandb_table, plot, plt):
    runs_to_plot = [
        "witold_taisner/molecule-generation/15o3nk8o",
    ]

    for run in runs_to_plot:
        df_for_upset, run_name = download_wandb_table(run)

        plot(df_for_upset.groupby(list(df_for_upset.columns)).size(), show_percentages=True, sort_by="cardinality", min_subset_size="1%")
        plt.title(run_name)
        plt.show()
    return


if __name__ == "__main__":
    app.run()
