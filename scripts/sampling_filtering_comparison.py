"""Generate report on how many molecules are passing the filters"""
import logging
from glob import glob

import pandas as pd

from modules.core.features.molecule_filter import MoleculeFilter

# Set up the logger for the module
logger = logging.getLogger(__name__)  # __name__ ensures the logger is specific to this module
logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s")
logger.setLevel(logging.ERROR)


# paths or glob patterns
FILES_TO_EVAL = "/home/witoldt/repositories/battery-rangers/data/sampling/*/*.csv"


if __name__ == "__main__":
    experts_molecules = pd.read_csv("/home/witoldt/repositories/battery-rangers/data/processed/data_experts_1.csv")["smiles"].values

    molecule_filter = MoleculeFilter()

    files = glob(FILES_TO_EVAL)

    all_smiles_that_passed = set()

    for file in files:
        try:
            smiles = pd.read_csv(file)["SMILES"].drop_duplicates().values
        except:  # pylint: disable=bare-except
            smiles = pd.read_csv(file)["smiles"].drop_duplicates().values

        smiles_filtered = molecule_filter.apply(smiles)

        print(f" === File {file} had {len(smiles_filtered)} that passed all filters. ===")

        all_smiles_that_passed |= set(smiles_filtered)

    print(all_smiles_that_passed)
    print(f"Total number of molecules: {len(all_smiles_that_passed)}")
    print(f"Excluding experts molecules: {len(all_smiles_that_passed) - len(set(experts_molecules).intersection(all_smiles_that_passed))}")
