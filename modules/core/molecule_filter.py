"""Class responsible for filtering molecules."""
import logging
import multiprocessing as mp
import time

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
from modules.generation.property_evaluator import PropertyEvaluator

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
        except Exception as e:
            logger.error(f"Error converting SMILES '{smiles}' to Mol: {e}")
            # Catch any rdkit-related errors during parsing
            return None
    # Return None if the input was not a string (e.g., None, float)
    return None


class MoleculeFilter:
    """Class responsible for filtering molecules."""

    def __init__(self, filters: list[GenericMoleculeFilter] | None = None, molecule_type: MoleculeType = MoleculeType.SUBSTRATE, db_file: str = "modules/bionemo/data/mol_db/substrate_properties.db"):
        """
        Initialize the MoleculeFilter object.

        Args:
            filters (list[GenericMoleculeFilter]): The list of filters to apply. If None, the default filters are used.
            molecule_type (MoleculeType): The type of molecule (SUBSTRATE or NODE) to determine default filters. Defaults to SUBSTRATE.
            db_file (str): Path to the database file for molecule properties. Defaults to substrate properties database.
        """
        self.db_file = db_file if molecule_type == MoleculeType.SUBSTRATE else "modules/bionemo/data/mol_db/node_properties.db"
        self.molecule_type = molecule_type

        logger.info("Initializing PropertyEvaluator for MoleculeFilter...")
        self.evaluator = PropertyEvaluator(
            molecule_type=self.molecule_type,
            db_file=self.db_file,
        )

        if filters is None and molecule_type == MoleculeType.SUBSTRATE:
            self.filters = [
                SMARTSFilter(),
                ConjugationFilter(),
                StericHindranceFilter(),
                FlatnessFilter(),
                CSMSymmetryFilter(),
            ]
        elif filters is None and molecule_type == MoleculeType.NODE:
            self.filters = [
                SMARTSFilter(),
                ConjugationFilter(),
                FlatnessFilter(max_flatness=5.0),
                StericHindranceFilter(),
                CSMSymmetryFilter(),
            ]
            logger.info(f"Using Node representation, flatness has a threshold of {self.filters[2].max_flatness} and CSM symmetry has a threshold of {self.filters[-1].symmetry_measure_threshold}.")
        else:
            self.filters = filters

    def apply_against_all_filters(self, molecules: list[str | Mol]) -> dict[str, dict[str, bool]]:
        """
        Check each molecule against each filter.

        Strategy:
        1. Check Database.
        2. If missing, use PropertyEvaluator to calculate all properties and write to DB.
        3. Fetch from DB and apply filters.
        4. Fallback to on-the-fly calculation if DB interaction fails. In general, this shouldn't happen.

        Args:
            molecules (list[str | Mol]): The list of molecules either as SMILES strings or RDKit Mol objects.
            If SMILES strings, they are converted to RDKit Mol objects.
        """
        if not molecules:
            return {}

        logger.info(f"Checking filters for {len(molecules)} molecules.")
        filter_results: dict[str, dict[str, bool]] = {}

        molecules_to_filter = []
        if isinstance(molecules[0], str):
            with mp.Pool(mp.cpu_count() // 2) as pool:
                mol_objects = pool.map(smiles_to_mol_worker, molecules)

            for original_smiles, mol in zip(molecules, mol_objects):
                if mol is not None:
                    molecules_to_filter.append(mol)
                elif isinstance(original_smiles, str):
                    filter_results[original_smiles] = {"Mol Conversion": False}
        else:
            molecules_to_filter = list(molecules)

        for mol in tqdm(molecules_to_filter, desc="Checking filters"):
            canon_smiles = Chem.MolToSmiles(mol)
            filter_results[canon_smiles] = {}

            properties = self.evaluator.get_properties_and_cache(canon_smiles)

            if properties:
                # We have data (either from DB or fresh from RAM)
                for filter_operator in self.filters:
                    filter_name = filter_operator.__class__.__name__
                    try:
                        # Ensure the filter knows how to read from the dict
                        passed = filter_operator.filter_from_property(properties)
                        filter_results[canon_smiles][filter_name] = passed
                    except Exception as e:
                        logger.error(f"Filter {filter_name} failed on data for {canon_smiles}: {e}")
                        filter_results[canon_smiles][filter_name] = False
            else:
                logger.error(f"Could not determine properties for '{canon_smiles}'. Marking all filters as failed.")
                for filter_operator in self.filters:
                    filter_results[canon_smiles][filter_operator.__class__.__name__] = False

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
