"""This script orchestrates the execution of REINVENT4 experiments based on configurations defined in TOML files.
It supports filtering experiments by name, molecule type, prior type, and recipe, and can force re-running of stages.
It generates configuration files for each stage, runs REINVENT commands, saves final configurations, updates a manifest, and calculates metrics.


To run a single, specific experiment by name:

    python scripts/reinvent_run_experiments.py --name substrate_chembl35_ft_rl_final

To run ALL experiments that use the ChEMBL pre-trained prior:

    python scripts/reinvent_run_experiments.py --prior-type chembl35

To run ALL experiments that perform the "ft_rl" recipe (for all priors and molecule types):

    python scripts/reinvent_run_experiments.py --recipe ft_rl

To run a very specific combination (the most powerful feature):

    # Run only the fine-tuning recipe, for substrates, starting from the vanilla prior.
    python scripts/reinvent_run_experiments.py --molecule-type substrate --prior-type vanilla --recipe ft
"""

import argparse
import glob
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import toml

from modules.core.enums import MoleculeType, PriorType, Recipe

# --- 2. Configuration Paths (relative to project root) ---
CONFIGS_DIR = Path("configs/reinvent")
TEMPLATE_DIR = CONFIGS_DIR / "templates"
GENERATED_CONFIG_DIR = Path("outputs/generated_configs")
MANIFEST_FILENAME = "experiment_manifest.json"

# --- 3. TOML Snippet for Inception ---
INCEPTION_TEMPLATE = """
[inception]
smiles_file = "{inception_smiles_file}"
memory_size = {inception_memory_size}
sample_size = {inception_sample_size}
"""


# --- 4. Helper Functions ---


def deep_merge(source: dict, destination: dict) -> dict:
    """Recursively merge two dictionaries."""
    for key, value in source.items():
        if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
            destination[key] = deep_merge(value, destination[key])
        else:
            destination[key] = value
    return destination


def run_command(command):
    """Executes a command and streams its output in real-time.
    Handles failures cleanly.
    """
    print(f"Executing: {' '.join(map(str, command))}")

    # Use Popen to start the process and get control over its output streams
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
    ) as proc:  # Redirect stderr to stdout  # Line-buffered
        # Read and print output line by line, in real-time
        for line in proc.stdout:
            print(line, end="")  # The 'end' prevents extra newlines

    # Check the final return code after the process has finished
    if proc.returncode != 0:
        print(f"\nERROR: Command failed with return code {proc.returncode}.")
        sys.exit(1)

    print("\nExecution successful.")


def save_config_and_update_manifest(manifest_path, record):
    """Saves the final generated config and updates the experiment manifest."""
    timestamp = record["timestamp"]
    config_to_save = record["config"]
    config_save_path = Path(record["config_save_path"])

    config_to_save["meta"] = {
        "experiment_name": record["name"],
        "generation_date": timestamp,
    }
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_save_path, "w", encoding="utf-8") as f:
        toml.dump(config_to_save, f)
    print(f"Saved final training config to {config_save_path}")

    manifest_data = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    manifest_record = {
        "experiment_name": record["name"],
        "timestamp": timestamp,
        "recipe": record["recipe"].value,
        "prior_type": record["prior_type"].value,
        "molecule_type": record["molecule_type"].value,
        "final_model_path": str(record["final_model_path"]),
        "training_config_path": str(config_save_path),
        "sampling_output_pattern": str(record["sampling_glob_pattern"]),
    }
    manifest_data.append(manifest_record)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)
    print(f"Updated manifest file at {manifest_path}")


def prepare_format_dict(d):
    """Recursively converts Python booleans to TOML-compatible lowercase strings."""
    prepared_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            prepared_dict[k] = prepare_format_dict(v)
        elif isinstance(v, bool):
            prepared_dict[k] = str(v).lower()
        else:
            prepared_dict[k] = v
    return prepared_dict


def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a nested dictionary for easy use with .format()."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# --- 5. Main Orchestration Logic ---


def main():
    """Main function to discover, filter, configure, and run experiments."""
    parser = argparse.ArgumentParser(description="Run REINVENT4 experiments. Should be run from the 'battery-rangers' project root.")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Run only the experiment with this specific name.",
    )
    parser.add_argument(
        "-m",
        "--molecule-type",
        type=str,
        choices=[e.value for e in MoleculeType],
        help="Filter by molecule type.",
    )
    parser.add_argument(
        "-p",
        "--prior-type",
        type=str,
        choices=[e.value for e in PriorType],
        help="Filter by prior type.",
    )
    parser.add_argument(
        "-r",
        "--recipe",
        type=str,
        choices=[e.value for e in Recipe],
        help="Filter by recipe.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-running of training stages even if models already exist.",
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Enable logging of the aggregated report to Weights & Biases.",
    )
    args = parser.parse_args()

    # Load templates
    try:
        with open(TEMPLATE_DIR / "tl_template.toml", "r", encoding="utf-8") as f:
            tl_template = f.read()
        with open(TEMPLATE_DIR / "rl_template.toml", "r", encoding="utf-8") as f:
            rl_template = f.read()
        with open(TEMPLATE_DIR / "sampling_template.toml", "r", encoding="utf-8") as f:
            sampling_template = f.read()
    except FileNotFoundError:
        print(f"Error: A template file was not found in '{TEMPLATE_DIR}'.")
        sys.exit(1)

    with open(CONFIGS_DIR / "base.toml", "r", encoding="utf-8") as f:
        base_config = toml.load(f)

    exp_files = glob.glob(str(CONFIGS_DIR / "experiments" / "**" / "*.toml"), recursive=True)
    experiments_to_run = []
    for exp_file in exp_files:
        with open(exp_file, "r", encoding="utf-8") as f:
            exp_specific_config = toml.load(f)
        if (
            (args.name and exp_specific_config.get("name") != args.name)
            or (args.molecule_type and exp_specific_config.get("molecule_type") != args.molecule_type)
            or (args.prior_type and exp_specific_config.get("prior_type") != args.prior_type)
            or (args.recipe and exp_specific_config.get("recipe") != args.recipe)
        ):
            continue
        final_config = deep_merge(exp_specific_config, base_config.copy())
        experiments_to_run.append(final_config)

    if not experiments_to_run:
        print("No experiments found matching your criteria. Exiting.")
        return

    print(f"Found {len(experiments_to_run)} experiment(s) to run.")

    for config in experiments_to_run:
        exp_name = config["name"]
        mol_type = MoleculeType(config["molecule_type"])
        prior_type = PriorType(config["prior_type"])
        recipe = Recipe(config["recipe"])
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rl_summary_dir = Path("outputs/rl_summaries")
        rl_summary_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}\nRunning Experiment: {exp_name}\n  (Molecule: {mol_type.name}, Prior: {prior_type.name}, Recipe: {recipe.name})\n{'=' * 60}")

        models_dir = Path(config["base_models_dir"]) / mol_type.value / prior_type.value / recipe.value
        sampling_dir = Path(config["base_sampling_dir"]) / mol_type.value / prior_type.value / recipe.value
        models_dir.mkdir(parents=True, exist_ok=True)
        sampling_dir.mkdir(parents=True, exist_ok=True)
        GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        try:
            prior_model_path = base_config["priors"][prior_type.value]
        except KeyError:
            print(f"ERROR: Prior '{prior_type.value}' not defined in [priors] section of base.toml")
            sys.exit(1)

        ft_model_path = models_dir / f"{exp_name}_ft.model"
        rl_model_path = models_dir / f"{exp_name}_rl.model"

        final_config_for_manifest = {}
        current_prior_path = prior_model_path

        # --- TL Stage ---
        if recipe in [Recipe.FT, Recipe.FT_RL, Recipe.FT_RL_INCEPTION]:
            print("\n--- Stage: Transfer Learning ---")
            if not ft_model_path.exists() or args.force_rerun:
                format_params = flatten_dict(prepare_format_dict(config))
                format_params.update(
                    {
                        "prior_model_path": str(current_prior_path),
                        "output_model_path": str(ft_model_path),
                        "seed": config["tl"]["seed"],  ## <-- FIX: Explicitly add the correct seed for this stage
                        "device": config["device"],  ## <-- FIX: Explicitly add the device
                    }
                )
                tl_config_content = tl_template.format(**format_params)

                generated_tl_config_path = GENERATED_CONFIG_DIR / f"{exp_name}_tl_{timestamp}.toml"
                with open(generated_tl_config_path, "w", encoding="utf-8") as f:
                    f.write(tl_config_content)

                run_command(["reinvent", generated_tl_config_path])
                final_config_for_manifest = toml.loads(tl_config_content)
            else:
                print(f"Skipping Transfer Learning: Model '{ft_model_path}' already exists.")
            current_prior_path = ft_model_path

        # --- RL Stage ---
        if recipe in [
            Recipe.RL,
            Recipe.RL_INCEPTION,
            Recipe.FT_RL,
            Recipe.FT_RL_INCEPTION,
        ]:
            print("\n--- Stage: Reinforcement Learning ---")
            if not rl_model_path.exists() or args.force_rerun:
                inception_block = ""
                if recipe in [Recipe.RL_INCEPTION, Recipe.FT_RL_INCEPTION]:
                    inception_params = flatten_dict(config["rl"]["inception"], "inception")
                    inception_block = INCEPTION_TEMPLATE.format(**inception_params)
                summary_path_prefix = rl_summary_dir / exp_name
                format_params = flatten_dict(prepare_format_dict(config))
                format_params.update(
                    {
                        "summary_csv_prefix": str(summary_path_prefix),
                        "prior_model_path": str(current_prior_path),
                        "agent_file": str(current_prior_path),
                        "output_model_path": str(rl_model_path),
                        "inception_block": inception_block,
                        "MoleculeTypePascalCase": mol_type.value.capitalize(),
                        "seed": config["rl"]["seed"],  ## <-- FIX: Explicitly add the correct seed for this stage
                        "device": config["device"],  ## <-- FIX: Explicitly add the device
                    }
                )
                rl_config_content = rl_template.format(**format_params)

                generated_rl_config_path = GENERATED_CONFIG_DIR / f"{exp_name}_rl_{timestamp}.toml"
                with open(generated_rl_config_path, "w", encoding="utf-8") as f:
                    f.write(rl_config_content)

                run_command(["reinvent", generated_rl_config_path])
                final_config_for_manifest = toml.loads(rl_config_content)
            else:
                print(f"Skipping Reinforcement Learning: Model '{rl_model_path}' already exists.")

        # --- Determine Final Model to Sample ---
        if recipe in [
            Recipe.RL,
            Recipe.RL_INCEPTION,
            Recipe.FT_RL,
            Recipe.FT_RL_INCEPTION,
        ]:
            model_to_sample_path = rl_model_path
        elif recipe == Recipe.FT:
            model_to_sample_path = ft_model_path
        else:  # SAMPLING_ONLY
            model_to_sample_path = Path(prior_model_path)

        final_model_path_for_manifest = model_to_sample_path

        # --- Sampling Stage ---
        print("\n--- Stage: Sampling ---")
        for i in range(1, config["sampling"]["num_files"] + 1):
            output_file_path = sampling_dir / f"{exp_name}_{i}.csv"
            format_params = flatten_dict(prepare_format_dict(config))
            format_params.update(
                {
                    "model_file": str(model_to_sample_path),
                    "output_file": str(output_file_path),
                    "device": config["device"],
                }
            )
            sampling_config_content = sampling_template.format(**format_params)

            generated_sampling_config_path = GENERATED_CONFIG_DIR / f"{exp_name}_sampling_{i}_{timestamp}.toml"
            with open(generated_sampling_config_path, "w", encoding="utf-8") as f:
                f.write(sampling_config_content)
            run_command(["reinvent", generated_sampling_config_path])

        sampling_glob_pattern = sampling_dir / f"{exp_name}_*.csv"

        # --- Save Configs and Update Manifest ---
        if recipe != Recipe.SAMPLING:
            save_config_and_update_manifest(
                manifest_path=models_dir / MANIFEST_FILENAME,
                record={
                    "name": exp_name,
                    "timestamp": timestamp,
                    "recipe": recipe,
                    "prior_type": prior_type,
                    "molecule_type": mol_type,
                    "final_model_path": final_model_path_for_manifest,
                    "config_save_path": models_dir / f"{exp_name}_config_{timestamp}.toml",
                    "config": final_config_for_manifest,
                    "sampling_glob_pattern": sampling_glob_pattern,
                },
            )

        # --- Metrics Calculation Stage ---
        print("\n--- Stage: Metrics Calculation ---")

        # 1. Determine the reference SMILES path based on molecule_type
        try:
            reference_smiles_path = config["metrics"]["reference_files"][mol_type.value]
        except KeyError:
            print(f"ERROR: No reference_files path defined for molecule_type '{mol_type.value}' in base.toml")
            sys.exit(1)

        # 2. Determine the training SMILES path based on prior_type (if it exists)
        # This uses .get() to safely return None if the key doesn't exist (e.g., for 'vanilla')
        training_files_lookup = config.get("metrics", {}).get("training_files", {})
        training_smiles_path = training_files_lookup.get(prior_type.value)

        # 3. Build the command
        metrics_cmd = [
            "python",
            "scripts/calculate_metrics.py",
            "--generated_smiles",
            str(sampling_glob_pattern),
            "--reference_smiles",
            reference_smiles_path,
            "--run_name",
            # f"{exp_name}_{timestamp}",
            f"{exp_name}",
            "--molecule_type",
            mol_type.value,
        ]

        # 4. Conditionally add the training_smiles argument
        if training_smiles_path:
            metrics_cmd.extend(["--training_smiles", training_smiles_path])
            print(f"Using training smiles for novelty calculation: {training_smiles_path}")
        else:
            print("No training smiles specified for this prior; novelty will be based on the reference set.")

        if args.log_wandb:
            metrics_cmd.append("--log_wandb")

        run_command(metrics_cmd)

        print(f"\n>>>> Experiment {exp_name} completed successfully! <<<<")


if __name__ == "__main__":
    main()
