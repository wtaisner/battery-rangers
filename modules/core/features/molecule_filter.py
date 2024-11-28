"""Class responsible for filtering molecules."""
import time

import pandas as pd
from rdkit import Chem

from modules.core.features.filters import *  # pylint: disable=wildcard-import


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
                SingleCCBondsOutsideRingsFilter()
                # Filter that leaves molecules without single C-C bonds outside rings.
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
            print(f"Applying filter: {filter_operator.__class__.__name__}")
            start_time = time.time()
            smiles = filter_operator.apply(smiles, **kwargs)
            print(f"Filtering time: {time.time() - start_time:.2f} s. Remaining SMILES: {len(smiles)}")
        return smiles


if __name__ == "__main__":
    molecule_filter = MoleculeFilter()
    expert_smiles = pd.read_csv("/home/witold/PycharmProjects/bmd-mol-generation/data/batteries.csv")["smiles"].drop_duplicates().values
    standardized_expert_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in expert_smiles]
    generated_smiles = pd.read_csv("/home/witold/PycharmProjects/bmd-mol-generation/REINVENT4/reinvent_sampling_50epochs_10000smiles_v5.csv")["SMILES"].drop_duplicates().values

    print(f"Expert smiles: {len(standardized_expert_smiles)}")
    print(f"Generated smiles: {len(generated_smiles)}")

    def set_diff(list1, list2):
        """Return the difference between two lists."""
        return list(set(list1).difference(set(list2)))

    generated_smiles = set_diff(generated_smiles, standardized_expert_smiles)
    print(f"Generated smiles after removing expert smiles: {len(generated_smiles)}")

    filtered_smiles = molecule_filter.apply(generated_smiles)
    print(f"Filtered smiles: {len(filtered_smiles)}")
