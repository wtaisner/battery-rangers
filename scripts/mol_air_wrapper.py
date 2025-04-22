"""Wrapper to run Mol-AIR end-to-end experiments."""
import argparse
import os

# pylint: disable=import-error
from modules.mol_air.train import MolRLInferenceFactory, MolRLPretrainFactory, MolRLTrainFactory

# export WANDB to run online
os.environ["WANDB_MODE"] = "online"


def run_molair_experiment(config_path: str) -> None:
    """
    Runs the full Mol-AIR end-to-end experiment sequence (pre-training,
    training, inference) by directly calling the appropriate classes and
    methods from the `train` module.  Assumes the configuration file is for
    an end-to-end experiment.

    Args:
        config_path: The path to the YAML configuration file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError:  If the configuration file is not valid YAML.
        Exception:  If any of the training or inference steps fail.

    Returns:
        None
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    print("----- Running Pre-training -----")
    MolRLPretrainFactory.from_yaml(config_path).create_pretrain().pretrain().close()

    # RL Training
    print("----- Running RL Training -----")
    MolRLTrainFactory.from_yaml(config_path).create_train().train().close()

    # Inference
    print("----- Running Inference -----")
    MolRLInferenceFactory.from_yaml(config_path).create_inference().inference().close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mol-AIR end-to-end experiments.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    run_molair_experiment(args.config_path)
