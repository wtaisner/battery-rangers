"""This module allows for using the REINVENT API in a Python object/script."""
import os
import time

import tomlkit


class ReinventRunner:
    """
    A class to handle REINVENT API.
    Args:
        config_file (str): Path to the preconfigured .toml file for REINVENT.
    """

    def __init__(self, config_file: str):
        with open(config_file, "r", encoding="utf-8") as f:
            self.config = tomlkit.load(f)

    def run_transfer_learning(self, output_folder: str, venv_path: str = ".reinvent_venv"):
        """
        Run the REINVENT API with transfer learning, store all outputs in the specified folder.
        Args:
            output_folder (str): The output folder to store the output files in.
            venv_path (str): The path to the virtual environment to use.
        """
        # add timestamp to the output folder
        output_folder = os.path.join(output_folder, f"{time.strftime('%Y%m%d%H%M')}")

        # update relevant config parameters
        self.config["json_out_config"] = os.path.join(output_folder, "config_tl.json")
        self.config["parameters"]["output_model_file"] = os.path.join(output_folder, "tl.model")
        self.config["tb_logdir"] = os.path.join(output_folder, "tensorboard")

        os.makedirs(output_folder, exist_ok=True)

        # dump the .toml file to the same folder
        toml_file = os.path.join(output_folder, "config_tl.toml")

        with open(toml_file, "w", encoding="utf-8") as f:
            tomlkit.dump(self.config, f)

        # run REINVENT using specified virtual environment
        os.system(f"{venv_path}/bin/python -m reinvent -l {output_folder}/log.out {toml_file}")

    def run_sampling(self, output_folder: str, venv_path: str = ".reinvent_venv"):
        """
        Run the REINVENT API with sampling, store all outputs in the specified folder.
        Args:
            output_folder (str): The output folder to store the output files in.
            venv_path (str): The path to the virtual environment to use.
        """
        # add timestamp to the output folder
        output_folder = os.path.join(output_folder, f"{time.strftime('%Y%m%d%H%M')}")

        # update relevant config parameters
        self.config["json_out_config"] = os.path.join(output_folder, "config_sampling.json")
        self.config["parameters"]["output_file"] = os.path.join(output_folder, "sampling.csv")

        os.makedirs(output_folder, exist_ok=True)
        # dump the .toml file to the same folder
        toml_file = os.path.join(output_folder, "config_tl.toml")

        with open(toml_file, "w", encoding="utf-8") as f:
            tomlkit.dump(self.config, f)

        # run REINVENT using specified virtual environment
        os.system(f"{venv_path}/bin/python -m reinvent -l {output_folder}/log.out {toml_file}")


if __name__ == "__main__":
    # CONFIG_FILE = "configs/reinvent/transfer_learning.toml"
    # runner = ReinventRunner(CONFIG_FILE)
    # runner.run_transfer_learning("outputs/transfer_learning")

    CONFIG_FILE = "configs/reinvent/sampling.toml"
    runner = ReinventRunner(CONFIG_FILE)
    runner.run_sampling("outputs/sampling")
