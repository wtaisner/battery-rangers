"""This module allows for using the REINVENT API in a Python object/script."""

import os
import time
from typing import Any, cast

import tomlkit


class ReinventRunner:
    """A class to handle REINVENT API.

    Args:
        config_file (str): Path to the preconfigured .toml file for REINVENT.

    """

    def __init__(self, config_file: str):
        with open(config_file, "r", encoding="utf-8") as f:
            self.config = tomlkit.load(f)

    def run_transfer_learning(self, output_folder: str):
        """Run the REINVENT API with transfer learning, store all outputs in the specified folder."""
        output_folder = os.path.join(output_folder, f"{time.strftime('%Y%m%d%H%M')}")

        config = cast(Any, self.config)
        config["json_out_config"] = os.path.join(output_folder, "config_tl.json")
        config["parameters"]["output_model_file"] = os.path.join(output_folder, "tl.model")
        config["tb_logdir"] = os.path.join(output_folder, "tensorboard")

        os.makedirs(output_folder, exist_ok=True)

        toml_file = os.path.join(output_folder, "config_tl.toml")

        with open(toml_file, "w", encoding="utf-8") as f:
            tomlkit.dump(self.config, f)

        os.system(f"reinvent -l {output_folder}/log.out {toml_file}")

    def run_sampling(self, output_folder: str):
        """Run the REINVENT API with sampling, store all outputs in the specified folder."""
        output_folder = os.path.join(output_folder, f"{time.strftime('%Y%m%d%H%M')}")

        config = cast(Any, self.config)
        config["json_out_config"] = os.path.join(output_folder, "config_sampling.json")
        config["parameters"]["output_file"] = os.path.join(output_folder, "sampling.csv")

        os.makedirs(output_folder, exist_ok=True)
        toml_file = os.path.join(output_folder, "config_tl.toml")

        with open(toml_file, "w", encoding="utf-8") as f:
            tomlkit.dump(self.config, f)

        os.system(f"reinvent -l {output_folder}/log.out {toml_file}")


if __name__ == "__main__":
    # CONFIG_FILE = "configs/reinvent/transfer_learning.toml"
    # runner = ReinventRunner(CONFIG_FILE)
    # runner.run_transfer_learning("outputs/transfer_learning")

    CONFIG_FILE = "configs/reinvent/sampling.toml"
    runner = ReinventRunner(CONFIG_FILE)
    runner.run_sampling("outputs/sampling")
