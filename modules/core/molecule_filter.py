"""Class responsible for filtering molecules."""
import logging
import multiprocessing as mp
import time

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from tqdm import tqdm

from modules.core.enums import MoleculeType
from modules.core.filters.conjugation_filter import ConjugationFilter
from modules.core.filters.csm_symmetry_filter import CSMSymmetryFilter
from modules.core.filters.flatness_filter import FlatnessFilter
from modules.core.filters.generic_filter import GenericMoleculeFilter
from modules.core.filters.smarts_filter import SMARTSFilter
from modules.core.filters.steric_hindrance_filter import StericHindranceFilter

# Set up the logger for the module
logger = logging.getLogger(__name__)  # __name__ ensures the logger is specific to this module
logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s")
logger.setLevel(logging.DEBUG)


def smiles_to_mol_worker(smiles):
    """
    Worker function for the multiprocessing pool.
    Safely converts a single SMILES string to an RDKit Mol object.
    Returns the Mol object on success or None on failure.
    """
    if isinstance(smiles, str):
        try:
            # Core RDKit conversion
            return Chem.MolFromSmiles(smiles)
        except Exception:
            # Catch any rdkit-related errors during parsing
            return None
    # Return None if the input was not a string (e.g., None, float)
    return None


class MoleculeFilter:
    """Class responsible for filtering molecules."""

    def __init__(self, filters: list[GenericMoleculeFilter] | None = None, molecule_type: MoleculeType = MoleculeType.SUBSTRATE):
        """
        Initialize the MoleculeFilter object.

        Args:
            filters (list[GenericMoleculeFilter]): The list of filters to apply. If None, the default filters are used.
        """
        if filters is None and molecule_type == MoleculeType.SUBSTRATE:
            self.filters = [
                SMARTSFilter(),
                ConjugationFilter(),
                StericHindranceFilter(),
                FlatnessFilter(),
            ]
        elif filters is None and molecule_type == MoleculeType.NODE:
            self.filters = [
                ConjugationFilter(),
                FlatnessFilter(),
                StericHindranceFilter(),
                CSMSymmetryFilter(),
            ]
        else:
            self.filters = filters

    def apply_against_all_filters(self, molecules: list[str | Mol], **kwargs) -> dict[str, dict[str, bool]]:
        """Check each molecule against each filter and return detailed results.

        Args:
            molecules (list[str | Mol]): The list of molecules either as SMILES strings or RDKit Mol objects.
                If SMILES strings, they are converted to RDKit Mol objects.

        Returns:
            dict[str, dict[str, bool]]: A dictionary where keys are SMILES of molecules, and values are dictionaries.
                  These inner dictionaries have filter names as keys and boolean values (True/False) indicating if the molecule passed the filter.
        """
        if len(molecules) == 0:
            return {}

        logger.debug(f"Checking filters for {len(molecules)} molecules.")
        filter_results: dict[str, dict[str, bool]] = {}

        molecules_to_filter = []  # Initialize the target list

        if isinstance(molecules[0], str):
            # Use multiprocessing to convert SMILES to Mol objects
            with mp.Pool(mp.cpu_count() // 2) as pool:
                mol_objects = pool.map(smiles_to_mol_worker, molecules)

            valid_mol_smiles = []
            # The loop now populates 'molecules_to_filter' directly
            for original_smiles, mol in zip(molecules, mol_objects):
                if mol is not None:
                    molecules_to_filter.append(mol)
                    valid_mol_smiles.append(original_smiles)
                else:
                    if isinstance(original_smiles, str):
                        filter_results[original_smiles] = {"Mol Conversion": False}

            # The map is built from the newly populated 'molecules_to_filter'
            original_smiles_map = {id(mol): smiles for mol, smiles in zip(molecules_to_filter, valid_mol_smiles)}

            # The log message correctly refers to the filtered list
            logger.debug(f"Number of molecules that could be converted to RDKit Mol objects: {len(molecules_to_filter)}")

        else:  # Input is already a list of Mol objects
            # Create the SMILES map from the original Mol objects
            original_smiles_map = {id(mol): Chem.MolToSmiles(mol) for mol in molecules}

            # Create a shallow copy for filtering. The original 'molecules' list is preserved.
            molecules_to_filter = list(molecules)

        for mol in molecules_to_filter:
            original_smiles = original_smiles_map[id(mol)]
            filter_results[original_smiles] = {}  # Initialize results for this molecule

            for filter_operator in self.filters:
                filter_name = filter_operator.__class__.__name__
                start_time = time.time()
                filter_result = filter_operator.apply([mol], **kwargs)  # Apply filter to single molecule
                # time_taken = time.time() - start_time

                if not filter_result:
                    filter_results[original_smiles][filter_name] = False
                    # logger.debug(f"Molecule {original_smiles} failed filter {filter_name} in {time_taken:.2f} s.")
                else:
                    filter_results[original_smiles][filter_name] = True
                    # logger.debug(f"Molecule {original_smiles} passed filter {filter_name} in {time_taken:.2f} s.")

        logger.debug(f"Filter checking complete for {len(molecules_to_filter)} molecules.")
        return filter_results

    # pylint: disable=too-many-branches
    def apply(self, molecules: list[str | Mol], return_mols: bool = False, **kwargs) -> tuple[list[str], dict[str, list[str]]]:
        """
        Apply the filter to a list of SMILES strings.

        Args:
            molecules (list[str | Mol]): The list of molecules either as SMILES strings or RDKit Mol objects.
            If SMILES strings, they are converted to RDKit Mol objects.
            return_mols (bool): If True, return the list of RDKit Mol objects instead of SMILES strings.
        Returns:
            tuple[list[str | Mol], dict[str, list[str]]]: A tuple containing:
                - The list of SMILES strings or RDkit Mol objects that passed ALL filters, depending on return_mols.
                - A dictionary where keys are SMILES of molecules that failed at least one filter,
                  and values are lists of filter names they failed.
        """
        if len(molecules) == 0:
            return [], {}
        logging.info(f"Applying filters to {len(molecules)} molecules.")
        filter_failure_reasons: dict[str, list[str]] = {}  # Store failure reasons per molecule
        filter_total_times: dict[str, float] = {filter_operator.__class__.__name__: 0.0 for filter_operator in self.filters}  # Store total times per filter

        molecules_to_filter = []  # Initialize the target list

        if isinstance(molecules[0], str):
            # Use multiprocessing to convert SMILES to Mol objects
            with mp.Pool(mp.cpu_count() // 2) as pool:
                mol_objects = pool.map(smiles_to_mol_worker, molecules)

            valid_mol_smiles = []
            # The loop now populates 'molecules_to_filter' directly
            for original_smiles, mol in zip(molecules, mol_objects):
                if mol is not None:
                    molecules_to_filter.append(mol)
                    valid_mol_smiles.append(original_smiles)
                else:
                    if isinstance(original_smiles, str):
                        filter_failure_reasons[original_smiles] = ["Invalid SMILES"]

            # The map is built from the newly populated 'molecules_to_filter'
            original_smiles_map = {id(mol): smiles for mol, smiles in zip(molecules_to_filter, valid_mol_smiles)}

            # The log message correctly refers to the filtered list
            logger.info(f"Number of molecules that could be converted to RDKit Mol objects: {len(molecules_to_filter)}")

        else:  # Input is already a list of Mol objects
            # Create the SMILES map from the original Mol objects
            original_smiles_map = {id(mol): Chem.MolToSmiles(mol) for mol in molecules}

            # Create a shallow copy for filtering. The original 'molecules' list is preserved.
            molecules_to_filter = list(molecules)

        molecules_passed_all_filters = []
        for mol in tqdm(molecules_to_filter, desc="Filtering molecules", total=len(molecules_to_filter)):
            passed_filters_for_mol = True
            failed_filters_names = []

            for filter_operator in self.filters:
                filter_name = filter_operator.__class__.__name__
                start_time = time.time()
                filter_result = filter_operator.apply([mol], **kwargs)  # Apply filter to single molecule
                time_taken = time.time() - start_time
                filter_total_times[filter_name] += time_taken  # Accumulate time

                if not filter_result:  # Assuming filter returns empty list if molecule fails
                    passed_filters_for_mol = False
                    failed_filters_names.append(filter_name)
                    break  # No need to apply further filters if one fails

            if passed_filters_for_mol:
                molecules_passed_all_filters.append(mol)
            elif failed_filters_names:  # Record failure reasons only if there were failures
                original_smiles = original_smiles_map.get(id(mol), Chem.MolToSmiles(mol) if mol else "Unknown Mol")  # Get SMILES from map or generate if needed
                filter_failure_reasons[original_smiles] = failed_filters_names

        logger.info(f"Filtering complete. Molecules passed all filters: {len(molecules_passed_all_filters)}, Molecules failed filters: {len(filter_failure_reasons)}")

        # Log average filter times
        num_processed_molecules = len(molecules_to_filter)  # Use processed valid molecules count
        if num_processed_molecules > 0:
            for filter_name, total_time in filter_total_times.items():
                average_time = total_time / num_processed_molecules
                logger.info(f"Average time for filter {filter_name} per molecule: {average_time:.3f} s")
        else:
            logger.info("No molecules were processed, cannot calculate average filter times.")

        if return_mols:
            return molecules_passed_all_filters, filter_failure_reasons

        return [Chem.MolToSmiles(mol) for mol in molecules_passed_all_filters], filter_failure_reasons


if __name__ == "__main__":
    molecule_filter = MoleculeFilter()
    expert_smiles = pd.read_csv("data/raw/data_experts1.csv")["smiles"].drop_duplicates().values
    # standardized_expert_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in expert_smiles]

    # generated_smiles = pd.read_csv("../../../data/sampling/reinvent_sampling_experts_1_50epochs_20000_smiles.csv")["SMILES"].drop_duplicates().values
    generated_smiles = pd.read_csv("data/sampling/ak_results/filtered_smiles_50epochs_second_trial.csv")["SMILES"].drop_duplicates().values

    logger.info(f"Expert smiles: {len(expert_smiles)}")
    logger.info(f"Generated smiles: {len(generated_smiles)}")

    def set_diff(list1, list2):
        """Return the difference between two lists."""
        return list(set(list1).difference(set(list2)))

    generated_smiles = set_diff(generated_smiles, expert_smiles)
    logger.info(f"Generated smiles after removing expert smiles: {len(generated_smiles)}")

    filtered_smiles, ffr = molecule_filter.apply(generated_smiles)
    logger.info(f"Final number of smiles after filtering: {len(filtered_smiles)}")
