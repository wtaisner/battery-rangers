# run_molair_custom.py

"""
Wrapper to run flexible Mol-AIR experiments.

Features:
- Conditionally runs pre-training if a 'Pretrain' section is in the config.
- Dynamically injects a list of initial SELFIES from a file via a command-line flag.
- Runs RL training.
- Performs a specified number of non-deterministic inference runs.
- **Uses a temporary file to safely pass modified configurations to the factory classes.**
"""
import argparse
import os
import tempfile

import yaml

import wandb

# pylint: disable=import-error
from modules.mol_air.train import MolRLInferenceFactory, MolRLPretrainFactory, MolRLTrainFactory


def run_experiment_stage(config: dict, stage: str, inference_runs: int = 1):
    """
    Helper function to run a specific stage (pretrain, train, inference)
    by writing the provided config to a temporary file and passing it to the factory.
    """
    # tempfile.NamedTemporaryFile creates a file that is automatically deleted on exit.
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".yaml") as temp_f:
        # Write the current state of the config dictionary to the temporary file
        yaml.dump(config, temp_f)
        # Ensure the data is written to disk before the factory tries to read it
        temp_f.flush()

        # Use the path of the temporary file with the factory's class method
        temp_config_path = temp_f.name

        if stage == "pretrain":
            print("\n----- Found 'Pretrain' section. Running Pre-training. -----")
            pretrainer = MolRLPretrainFactory.from_yaml(temp_config_path).create_pretrain()
            pretrainer.pretrain()
            pretrainer.close()
            print("----- Pre-training Finished. -----")

        elif stage == "train":
            print("\n----- Running RL Training -----")
            trainer = MolRLTrainFactory.from_yaml(temp_config_path).create_train()
            trainer.train()
            trainer.close()
            print("----- RL Training Finished -----")

        elif stage == "inference":
            if inference_runs > 0:
                print(f"\n----- Starting {inference_runs} Inference Runs -----")
                for i in range(inference_runs):
                    print(f"\n----- Running Inference: Run {i + 1}/{inference_runs} -----")
                    inference_runner = MolRLInferenceFactory.from_yaml(temp_config_path).create_inference()
                    inference_runner.inference(file_id=i)
                    inference_runner.close()


def run_molair_experiment(config_path: str, init_selfies_path: str | None, inference_runs: int) -> None:
    """
    Runs a flexible Mol-AIR experiment sequence.
    This version correctly handles the configuration dictionary format expected by the factories.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # --- Step 1: Load the full, original YAML content ---
    print(f"----- Loading Configuration from {config_path} -----")
    with open(config_path, "r", encoding="utf-8") as f:
        # This dictionary has the top-level experiment ID key. DO NOT unwrap it here.
        config_with_id = yaml.safe_load(f)

    # --- Step 2: Get the inner config for local checks and modifications ---
    try:
        experiment_id = list(config_with_id.keys())[0]
        # 'inner_config' is the dictionary that contains 'Pretrain', 'Train', etc.
        inner_config = config_with_id[experiment_id]
    except (IndexError, TypeError):
        raise ValueError(f"YAML file '{config_path}' appears to be empty or misformatted.")

    wandb.init(
        project="Mol-AIR",
        name=experiment_id,  # Use the experiment ID from the YAML as the run name.
        config=inner_config,  # Log the entire configuration for reproducibility.
        # Explicitly set the mode to 'online' to override any local settings.
        settings=wandb.Settings(mode="online"),
    )

    # Use a try...finally block to guarantee that wandb.finish() is always called,
    # even if an error occurs during one of the stages.
    success = False
    try:
        # --- Step 2: Dynamically add 'init_selfies' if provided ---
        if init_selfies_path:
            print(f"----- Injecting initial SELFIES from: {init_selfies_path} -----")
            if not os.path.exists(init_selfies_path):
                raise FileNotFoundError(f"Initial SELFIES file not found: {init_selfies_path}")

            with open(init_selfies_path, "r", encoding="utf-8") as f:
                selfies_list = [line.strip() for line in f if line.strip()]

            if "Env" not in inner_config:
                inner_config["Env"] = {}
            inner_config["Env"]["init_selfies"] = selfies_list
            print(f"Successfully loaded and set {len(selfies_list)} initial SELFIES strings.")
            # Update wandb config with the new selfies (optional but good practice)
            wandb.config.update({"Env": inner_config["Env"]}, allow_val_change=True)

        # --- Step 3: Conditionally run Pre-training ---
        if "Pretrain" in inner_config:
            run_experiment_stage(config_with_id, "pretrain")

        # --- Step 4: Run RL Training ---
        run_experiment_stage(config_with_id, "train")

        # --- Step 5: Run Inference Loop ---
        inference_run_config = config_with_id.copy()
        if "Inference" in inner_config and "seed" in inner_config["Inference"]:
            inference_run_config[experiment_id] = inner_config.copy()
            inference_run_config[experiment_id]["Inference"] = inner_config["Inference"].copy()
            del inference_run_config[experiment_id]["Inference"]["seed"]
            print("\nNote: The 'seed' from the 'Inference' section has been removed for varied runs.")

        run_experiment_stage(inference_run_config, "inference", inference_runs=inference_runs)

        # If all stages complete without error, we mark the run as successful.
        success = True
        print("\n----- All Experiments Finished -----")

    finally:
        # --- WANDB INTEGRATION: FINALIZE RUN ---
        # This block will execute whether the 'try' block succeeded or failed.
        print("----- Finalizing WandB Run -----")
        exit_code = 0 if success else 1  # 0 for success, 1 for failure
        wandb.finish(exit_code=exit_code)
        if exit_code == 1:
            print("WandB run marked as 'failed' due to an error.")
        else:
            print("WandB run finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run flexible Mol-AIR experiments (Pre-train, Train, Inference).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_path", type=str, help="Path to the master YAML configuration file.")
    parser.add_argument("--init_selfies_path", type=str, default=None, help="Path to a .slf file with initial SELFIES strings, one per line.")
    parser.add_argument("--inference_runs", type=int, default=10, help="Number of inference runs to perform after RL training.")
    args = parser.parse_args()

    run_molair_experiment(args.config_path, args.init_selfies_path, args.inference_runs)
