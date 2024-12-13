"""Class responsible for filtering molecules."""
import logging
import time

import pandas as pd
from rdkit import Chem

from modules.core.features.filters.c_n_triple_bonds_filter import CNTripleBondsFilter
from modules.core.features.filters.flatness_filter import FlatnessFilter
from modules.core.features.filters.generic_filter import GenericMoleculeFilter
from modules.core.features.filters.point_group_symmetry_filter import PointGroupSymmetryFilter
from modules.core.features.filters.steric_hindrance_filter import StericHindranceFilter
from modules.core.features.filters.symmetry_filter import SymmetryFilter
from modules.core.features.filters.xyz_pattern_filter import XYZPatternFilter

# Set up the logger for the module
logger = logging.getLogger(__name__)  # __name__ ensures the logger is specific to this module
logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s")
logger.setLevel(logging.DEBUG)


class MoleculeFilter:
    """Class responsible for filtering molecules."""

    def __init__(self, filters: list[GenericMoleculeFilter] | None = None):
        """
        Initialize the MoleculeFilter object.

        Args:
            filters (list[GenericMoleculeFilter]): The list of filters to apply. If None, the default filters are used.
        """
        if filters is None:
            self.filters = [
                SymmetryFilter(),  # Filter that leaves molecules with a specific symmetry.
                CNTripleBondsFilter(),  # Filter that leaves molecules with triple bonds.
                PointGroupSymmetryFilter(),  # Filter that leaves molecules with a specific point group symmetry.
                FlatnessFilter(),  # Filter that leaves molecules with a specific flatness.
                StericHindranceFilter(),  # Filter that leaves molecules without steric hindrance.
                XYZPatternFilter(),  # Filter that leaves molecules without X-Y-Z patterns.
            ]
        else:
            self.filters = filters

    def apply(self, smiles: list[str], **kwargs) -> list[str]:
        """
        Apply the filter to a list of SMILES strings.

        Args:
            smiles (list[str]): The list of SMILES strings to filter.
        Returns:
            list[str]: The list of SMILES strings that passed the filter.
        """
        for filter_operator in self.filters:
            logger.info(f"Applying filter: {filter_operator.__class__.__name__}")
            start_time = time.time()
            smiles = filter_operator.apply(smiles, **kwargs)
            logger.info(f"Filtering time: {time.time() - start_time:.2f} s. Remaining SMILES: {len(smiles)}")
        return smiles


if __name__ == "__main__":
    molecule_filter = MoleculeFilter()
    expert_smiles = pd.read_csv("../../../data/processed/data_experts_1.csv")["smiles"].drop_duplicates().values
    standardized_expert_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in expert_smiles]

    # generated_smiles = pd.read_csv("../../../data/sampling/reinvent_sampling_experts_1_50epochs_20000_smiles.csv")["SMILES"].drop_duplicates().values
    generated_smiles = pd.read_csv("../../../data/sampling/ak_results/filtered_smiles_50epochs_second_trial.csv")["SMILES"].drop_duplicates().values

    logger.info(f"Expert smiles: {len(standardized_expert_smiles)}")
    logger.info(f"Generated smiles: {len(generated_smiles)}")

    def set_diff(list1, list2):
        """Return the difference between two lists."""
        return list(set(list1).difference(set(list2)))

    generated_smiles = set_diff(generated_smiles, standardized_expert_smiles)
    logger.info(f"Generated smiles after removing expert smiles: {len(generated_smiles)}")

    filtered_smiles = molecule_filter.apply(generated_smiles)
    logger.info(f"Final number of smiles after filtering: {len(filtered_smiles)}")
