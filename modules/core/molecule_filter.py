"""Class responsible for filtering molecules."""
import logging
import multiprocessing as mp
import time

from rdkit import Chem
from rdkit.Chem import Mol
from tqdm import tqdm

from modules.core.database import MoleculeDB
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
        self.database = MoleculeDB(self.db_file)
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
                ConjugationFilter(),
                FlatnessFilter(max_flatness=5.0),
                StericHindranceFilter(),
                CSMSymmetryFilter(),
            ]
            logger.info(f"Using Node representation, flatness has a threshold of {self.filters[1].max_flatness} and CSM symmetry has a threshold of {self.filters[3].symmetry_measure_threshold}.")
        else:
            self.filters = filters

    def apply_against_all_filters(self, molecules: list[str | Mol], **kwargs) -> dict[str, dict[str, bool]]:
        """
        Check each molecule against each filter using a "read-from-db-or-compute" strategy.

        For each molecule, it first tries to read pre-computed properties from the database
        and apply filters based on them. If the molecule is not in the database, it falls
        back to computing the filter results on-the-fly.

        NOTE: This method does NOT write new results to the database.

        Args:
            molecules (list[str | Mol]): The list of molecules either as SMILES strings or RDKit Mol objects.
            If SMILES strings, they are converted to RDKit Mol objects.
            **kwargs: Additional keyword arguments passed to filter apply methods.
        """
        if not molecules:
            return {}

        logger.info(f"Checking filters for {len(molecules)} molecules with DB fallback.")
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

        for mol in tqdm(molecules_to_filter, desc="Checking filters (DB/Compute)"):
            canon_smiles = Chem.MolToSmiles(mol)
            filter_results[canon_smiles] = {}

            # Attempt to read from the database
            db_properties = self.database.get_molecule_properties(canon_smiles)

            if db_properties:
                # --- PATH 1: CACHE HIT (Fast) ---
                # Molecule found in the database, use pre-computed properties.
                logger.debug(f"'{canon_smiles}' found in DB. Using cached properties for filtering.")
                for filter_operator in self.filters:
                    filter_name = filter_operator.__class__.__name__
                    passed = filter_operator.filter_from_property(db_properties)
                    filter_results[canon_smiles][filter_name] = passed
            else:
                # --- PATH 2: CACHE MISS (Fallback to on-the-fly computation) ---
                # Molecule not in the database, apply filters directly.
                logger.debug(f"'{canon_smiles}' not in DB. Computing filters on-the-fly.")
                for filter_operator in self.filters:
                    filter_name = filter_operator.__class__.__name__
                    try:
                        filter_pass_list = filter_operator.apply([mol], **kwargs)
                        passed = bool(filter_pass_list)  # True if the result list is not empty
                        filter_results[canon_smiles][filter_name] = passed
                    except Exception as e:
                        filter_results[canon_smiles][filter_name] = False
                        logger.error(f"Error applying filter {filter_name} to molecule {canon_smiles}: {e}")

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
