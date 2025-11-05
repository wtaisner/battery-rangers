"""
Evaluates generative models for molecule design using standard metrics.

This module provides the MoleculeGenerationEvaluator class to calculate
metrics like Validity, Uniqueness, Novelty, Internal Diversity, and FCD
for sets of generated SMILES strings.
"""

import logging
import multiprocessing as mp
from itertools import combinations

import numpy as np
from fcd_torch import FCD
from rdkit import Chem, DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.rdBase import DisableLog
from rdkit.SimDivFilters import LeaderPicker
from tqdm.auto import tqdm

import wandb
from modules.core.enums import MoleculeType
from modules.core.molecule_filter import MoleculeFilter

DisableLog("rdApp.*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
logger = logging.getLogger(__name__)


def _worker_preprocess_smiles(smiles: str) -> tuple[str | None, Chem.Mol | None]:
    """
    Worker function for multiprocessing pool.
    Converts a single SMILES to a canonical SMILES and RDKit Mol object.

    Args:
        smiles: The SMILES string to process.

    Returns:
        A tuple of (canonical_smiles, mol_object) or (None, None) on failure.
    """
    if not isinstance(smiles, str):
        return None, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Return both the canonical form and the molecule object
            return Chem.MolToSmiles(mol, canonical=True), mol
    except Exception:
        # Catches any errors during parsing
        return None, None
    return None, None


# pylint: disable=too-many-instance-attributes
class MoleculeGenerationEvaluator:
    """
    Evaluates de novo molecule generation tasks using standard cheminformatics metrics.

    See: https://arxiv.org/pdf/1811.12823 for reference on the metrics used.

    This class takes lists of generated, training, and reference SMILES strings,
    processes them (validity check, canonicalization), and calculates metrics
    including Validity, Uniqueness, Novelty, Internal Diversity, and FCD.

    Attributes:
        generated_smiles (list[str]): Original list of generated SMILES.
        training_smiles (list[str]): Original list of training SMILES.
        reference_smiles (list[str]): Original list of reference SMILES.
        n_jobs (int): Number of parallel jobs for FCD calculation.
        device (str): Device ('cpu' or 'cuda:X') for FCD calculation.
        batch_size (int): Batch size for FCD calculation.
    """

    def __init__(
        self, generated_smiles: list[str], training_smiles: list[str] | None = None, reference_smiles: list[str] | None = None, n_jobs: int = 8, device: str = "cpu", batch_size: int = 512, **kwargs
    ):
        self.generated_smiles: list[str] = generated_smiles
        self.training_smiles: list[str] = training_smiles or []
        self.reference_smiles: list[str] = reference_smiles or []

        # FCD Specific Parameters
        self.n_jobs: int = n_jobs
        self.device: str = device  # User provides the specific device string
        self.batch_size: int = batch_size

        self._fcd_calculator: FCD | None = None
        self.molecule_filter = MoleculeFilter(filters=None, molecule_type=kwargs.get("molecule_type", MoleculeType.SUBSTRATE))

        # --- Preprocess SMILES ---
        # This step validates and canonicalizes SMILES, preparing them for metric calculation.
        # It stores the *count* of valid molecules before deduplication (n_generated_valid)
        # and the *list* of unique valid canonical SMILES (valid_generated_smiles_canon).
        self.valid_generated_mols, self.valid_generated_smiles_canon, self.n_generated_valid = self._preprocess_smiles_list(self.generated_smiles, "Generated")
        self.n_generated_total: int = len(self.generated_smiles)

        # If training set is empty, we set valid_training_smiles_canon to an empty list
        if not self.training_smiles:
            self.valid_training_smiles_canon = []
        else:
            _, self.valid_training_smiles_canon, _ = self._preprocess_smiles_list(self.training_smiles, "Training")

        _, self.valid_reference_smiles_canon, _ = self._preprocess_smiles_list(self.reference_smiles, "Reference")

    @staticmethod
    def _preprocess_smiles_list(smiles_list: list[str], list_name: str, n_jobs: int = 10) -> tuple[list[Chem.Mol], list[str], int]:
        """
        Validates and canonicalizes a list of SMILES strings using multiprocessing.

        Args:
            smiles_list: The list of SMILES strings to process.
            list_name: A descriptive name for the list (e.g., "Generated").
            n_jobs: The number of processes to use.

        Returns:
            A tuple containing:
            - list[Mol]: List of unique valid RDKit Mol objects.
            - list[str]: List of unique valid canonical SMILES strings.
            - int:       Count of valid molecules found *before* deduplication.
        """
        if not smiles_list:
            logger.info("%s list is empty, skipping preprocessing.", list_name)
            return [], [], 0

        valid_mols_dict: dict[str, Chem.Mol] = {}  # Tracks unique valid SMILES -> Mol
        valid_mol_count = 0

        # Use multiprocessing Pool to parallelize the SMILES processing
        # The 'with' statement ensures the pool is properly closed.
        with mp.Pool(processes=n_jobs) as pool:
            # The 'desc' provides a nice progress bar with tqdm
            results = list(tqdm(pool.imap(_worker_preprocess_smiles, smiles_list), total=len(smiles_list), desc=f"Processing {list_name} SMILES"))

        # Process the results gathered from the pool
        for canon_smiles, mol in results:
            if canon_smiles and mol:
                valid_mol_count += 1
                # Add to dict to ensure uniqueness based on canonical SMILES
                if canon_smiles not in valid_mols_dict:
                    valid_mols_dict[canon_smiles] = mol

        unique_valid_smiles = list(valid_mols_dict.keys())
        unique_valid_mols = list(valid_mols_dict.values())

        return unique_valid_mols, unique_valid_smiles, valid_mol_count

    def evaluate(
        self, log_wandb: bool = False, log_examples: bool = False, project: str = "molecule-generation", run_name: str | None = None, num_examples: int = 5
    ) -> tuple[dict[str, float], dict[str, dict[str, bool]]]:
        """
        Runs the calculation of defined metrics.

        Args:
            log_wandb (bool): If True, logs the results to Weights & Biases. Defaults to False.
            log_examples (bool): If True, logs examples of valid generated SMILES. Requires wandb to be True. Defaults to False
            project (str): The name of the experiment. Defaults to "molecule-generation".
            run_name (str): The name of the run. Defaults to None.
            num_examples (int): The number of examples to log. Defaults to 5.

        Returns:
            1. A dictionary containing the results. Keys are metric names (str),
            values are the calculated scores (float or np.nan if calculation
            failed or requirements were not met).
            2. dict[str, dict[str, bool]] with detailed filter results for each molecule, key is molecule, value is a dictionary with key as filter name and value as bool indicating pass/fail
        """

        percent_passing_filters, upset_results = self.calculate_percent_passing_filters()

        results = {
            "validity": self.calculate_validity(),
            "uniqueness": self.calculate_uniqueness(),
            "novelty_wrt_training_set": self.calculate_novelty(set(self.valid_training_smiles_canon)),
            "novelty_wrt_reference_set": self.calculate_novelty(set(self.valid_reference_smiles_canon)),
            "internal_diversity": self.calculate_internal_diversity(),
            "fcd": self.calculate_fcd() if self.reference_smiles else np.nan,
            "percent_passing_filters": percent_passing_filters,
            "num_valid_molecules": self.get_num_valid_molecules(),
            "#circles": self.calculate_circles_metric()[0],
        }

        if log_wandb:
            wandb.init(
                project=project,
                name=run_name,
                entity="witold_taisner",
            )

            wandb.log(results, step=0)

            if log_examples:
                # log num_examples random examples from generated molecules that are not in training and reference set
                unique_valid_generated_set = set(self.valid_generated_smiles_canon)
                novel_molecules_set = unique_valid_generated_set - set(self.valid_training_smiles_canon) - set(self.valid_reference_smiles_canon)
                num_novel = len(novel_molecules_set)

                if num_novel == 0:
                    logger.warning("Cannot log examples: No novel molecules found.")
                else:
                    # Randomly select num_examples from the novel molecules
                    selected_examples = np.random.choice(list(novel_molecules_set), size=min(num_examples, num_novel), replace=False)

                    mols = []
                    for smiles in selected_examples:
                        mol = wandb.Molecule.from_smiles(smiles, caption=smiles)
                        mols.append(mol)
                    wandb.log({"unique novel examples": mols})

        return results, upset_results

    # --- Core Metrics ---

    def calculate_validity(self) -> float:
        """
        Calculates the fraction of SMILES strings that yield valid molecules.

        Validity = (Number of Valid Molecules Found) / (Total Input SMILES)

        Returns:
            Validity score [0.0, 1.0]. Returns 0.0 if no SMILES were provided.
        """
        return self.n_generated_valid / self.n_generated_total

    def calculate_uniqueness(self) -> float:
        """
        Calculates the fraction of unique molecules among the *valid* generated molecules.

        Uniqueness = (Number of Unique Valid Molecules) / (Number of Valid Molecules Found)

        Returns:
            Uniqueness score [0.0, 1.0]. Returns 0.0 if no valid molecules generated.
        """
        # The number of unique valid molecules is the length of the deduplicated list.
        num_unique_valid = len(self.valid_generated_smiles_canon)

        if self.n_generated_valid == 0:
            # logger.warning("Cannot calculate uniqueness: No valid generated molecules.")
            return 0.0
        return num_unique_valid / self.n_generated_valid

    def calculate_novelty(self, reference_set: set[str]) -> float:
        """
        Calculates the fraction of *unique valid* generated molecules not present
        in the reference set.

        Novelty = (Number of Unique Valid Generated Molecules not in reference set)
                  / (Number of Unique Valid Generated Molecules)

        Args:
            reference_set: A set of canonical SMILES strings for calculation of novelty.

        Returns:
            Novelty score [0.0, 1.0]. Returns np.nan if training set was not
            provided. Returns 0.0 if no unique valid molecules were generated.
        """
        if reference_set is None or len(reference_set) == 0:
            logger.warning("Cannot calculate novelty: Reference set is None.")
            return np.nan  # Cannot calculate novelty without a reference set

        unique_valid_generated_set = set(self.valid_generated_smiles_canon)
        num_unique_valid = len(unique_valid_generated_set)

        if num_unique_valid == 0:
            logger.warning("Cannot calculate novelty: No unique valid generated molecules.")
            return 0.0

        # Use the precomputed set of canonical training SMILES
        novel_molecules_set = unique_valid_generated_set - reference_set
        num_novel = len(novel_molecules_set)

        # Standard definition: normalize by the number of unique valid generated mols
        return num_novel / num_unique_valid

    def calculate_internal_diversity(self, p: int = 1, fp_radius: int = 2, fp_bits: int = 1024) -> float:
        """
        Calculates internal diversity using Tanimoto similarity of Morgan fingerprints.
        This metric detects a common failure case of generative models—mode collapse. With mode collapse,
        the model produces a limited variety of samples, ignoring some areas of the chemical space. A higher
        value of this metric corresponds to higher diversity in the generated set.

        IntDiv_p = 1 - [ (1/|G|^2) * Sum_{m1,m2 in G}( T(m1,m2)^p ) ]^(1/p)
        Calculated over the set of *unique valid* generated molecules.

        This implementation is optimized to avoid redundant similarity calculations.

        Args:
            p: The power parameter for the internal diversity calculation. Defaults to 1.
            fp_radius: Morgan fingerprint radius. Defaults to 2.
            fp_bits: Morgan fingerprint number of bits. Defaults to 1024.

        Returns:
            Internal diversity score [0.0, 1.0]. Returns 0.0 if fewer than 2
            unique valid molecules exist.
        """
        unique_valid_mols = self.valid_generated_mols
        if len(unique_valid_mols) < 2:
            logger.warning("Cannot calculate internal diversity: requires at least 2 unique valid molecules, but found %d.", len(unique_valid_mols))
            return 0.0

        fingerprints = []
        for mol in unique_valid_mols:
            try:
                fp = GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
                fingerprints.append(fp)
            except (RuntimeError, ValueError) as e:
                logger.error(f"Error calculating fingerprint, skipping molecule: {e}")

        num_fingerprints = len(fingerprints)
        if num_fingerprints < 2:
            logger.warning("Could not generate enough valid fingerprints (%d) for diversity calculation.", num_fingerprints)
            return 0.0

        # --- OPTIMIZED CALCULATION ---
        # Calculate the sum of similarities for the upper triangle of the similarity matrix.
        off_diagonal_sum_sim_p = 0.0

        # Use itertools.combinations to get all unique pairs of indices (i, j) where i < j
        for i, j in tqdm(combinations(range(num_fingerprints), 2), desc="Calculating internal diversity"):
            try:
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                off_diagonal_sum_sim_p += np.power(sim, p)
            except RuntimeError as e:
                logger.error(f"Error calculating Tanimoto similarity: {e}")

        # The full sum includes the symmetric pairs and the diagonal.
        # Total Sum = (2 * Off-Diagonal Sum) + (Diagonal Sum)
        # The sum of diagonal elements (T(m,m)=1) is simply num_fingerprints.
        total_sum_sim_p = (2 * off_diagonal_sum_sim_p) + num_fingerprints

        average_similarity = total_sum_sim_p / (num_fingerprints**2)

        # Calculate the p-th root of the average similarity
        if p == 1:
            root_mean_sim_p = average_similarity
        else:
            root_mean_sim_p = np.power(max(0.0, average_similarity), 1.0 / p)

        int_div = 1.0 - root_mean_sim_p
        return int_div

    def calculate_circles_metric(self, distance_threshold: float = 0.5, fp_radius: int = 2, fp_bits: int = 1024):
        """
        Calculates the #Circles metric (size of the maximally diverse subset)
        for a list of molecules using a sphere exclusion algorithm (LeaderPicker).

        This implementation uses the efficient LazyBitVectorPick method which operates
        directly on fingerprints and uses a distance threshold.

        Args:
            distance_threshold (float): The minimum Tanimoto distance between any two selected molecules.
            fp_radius (int): Morgan fingerprint radius. Defaults to 2.
            fp_bits (int): Morgan fingerprint number of bits. Defaults to 1024.

        Returns:
            tuple[int, list]: A tuple containing:
                - int: The size of the diverse subset (#Circles metric).
                - list: The indices of the selected diverse molecules.
        """
        unique_valid_mols = self.valid_generated_mols
        if not unique_valid_mols or len(unique_valid_mols) < 2:
            logger.warning("Cannot calculate internal diversity: requires at least 2 unique valid molecules, but found %d.", len(unique_valid_mols or []))
            return 0, []

        fingerprints = []
        # Keep track of original indices for valid molecules
        valid_mol_indices = []
        for i, mol in enumerate(unique_valid_mols):
            try:
                if mol is None:
                    continue
                fp = GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
                fingerprints.append(fp)
                valid_mol_indices.append(i)
            except Exception as e:
                logger.error(f"Error calculating fingerprint for molecule at index {i}, skipping: {e}")

        num_fingerprints = len(fingerprints)
        if num_fingerprints < 2:
            logger.warning("Could not generate enough valid fingerprints (%d) for diversity calculation.", num_fingerprints)
            return 0, []

        picker = LeaderPicker()

        diverse_indices_in_fp_list = picker.LazyBitVectorPick(fingerprints, num_fingerprints, distance_threshold)

        original_indices = [valid_mol_indices[i] for i in diverse_indices_in_fp_list]

        return len(original_indices), list(original_indices)

    def calculate_fcd(self) -> float:
        """
        Calculates the Fréchet ChemNet Distance (FCD).

        Compares the distribution of *unique valid* generated molecules to the
        distribution of *unique valid* reference molecules using pre-trained
        ChemNet features. Lower scores indicate higher similarity between datasets.

        Requires `reference_smiles` provided during initialization. Assumes
        `fcd_torch` library is installed and the provided `device` string is valid.

        Uses the callable FCD object API: `fcd_calculator(ref_list, gen_list)`.

        Returns:
            FCD score (float, lower is better). Returns np.nan if requirements
            are not met (missing reference data, no valid molecules in either set,
            or FCD calculation error).
        """
        if not self.reference_smiles:
            raise ValueError("Reference SMILES are required for FCD calculation.")

        # Use the unique valid canonical SMILES lists derived during preprocessing
        gen_smiles_list = self.valid_generated_smiles_canon  # Unique list
        ref_smiles_list = list(self.valid_reference_smiles_canon)  # Unique list

        if not gen_smiles_list:
            return np.nan  # No valid generated molecules to compare

        if not ref_smiles_list:
            return np.nan  # No valid reference molecules to compare

        try:
            # Lazy initialization of FCD calculator
            if self._fcd_calculator is None:
                self._fcd_calculator = FCD(device=self.device, n_jobs=self.n_jobs, batch_size=self.batch_size)

            logger.debug(f"Calculating FCD between {len(gen_smiles_list)} unique generated and {len(ref_smiles_list)} unique reference molecules...")

            fcd_score = self._fcd_calculator(ref_smiles_list, gen_smiles_list)
            return float(fcd_score)  # Ensure result is float

        except RuntimeError as e:
            logger.error(f"Error calculating FCD: {e}")
            return np.nan

    def calculate_percent_passing_filters(self) -> tuple[float, dict]:
        """
        Calculates the percent of generated molecules that pass all filters.

        Returns:
            The percent of generated molecules that pass all filters.
        """
        if self.valid_generated_smiles_canon:
            results = self.molecule_filter.apply_against_all_filters(self.valid_generated_mols)

            percent_passing_filters = len([smiles for smiles, outcomes in results.items() if all(outcomes.values())]) / len(self.valid_generated_mols)
            return percent_passing_filters, results

        return 0.0, {}

    def get_num_valid_molecules(self) -> int:
        """
        Returns the number of valid generated molecules.

        Returns:
            The number of valid generated molecules.
        """
        return self.n_generated_valid


if __name__ == "__main__":
    generated_smiles_example = [
        "CCO",  # duplicate of training / generated
        "CCC",  # novel, valid
        "c1ccccc1",  # duplicate of training
        # "invalid-smiles-string",  # invalid
        "CC(=O)O",  # duplicate of training
        "CCO",  # duplicate of training / generated
        "N#CC1=CC=C(C#N)C=C1",
    ]

    training_smiles_example = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic Acid
        "C",  # Methane
        "CC",  # Ethane
        "c1ccccc1",  # Benzene (make one generated mol non-novel)
        "CC(C)C",  # Isobutane
    ]

    # Reference set (more diverse, drug-like subset)
    reference_smiles_example = [
        "O=C=O",  # Carbon Dioxide
        "c1ccc(C(=O)O)cc1",  # Benzoic Acid (canonical)
        "Nc1ccccc1",  # Aniline
    ]

    evaluator = MoleculeGenerationEvaluator(
        generated_smiles=generated_smiles_example,
        training_smiles=training_smiles_example,
        reference_smiles=reference_smiles_example,
        n_jobs=4,  # Adjust based on CPU cores
        device="cpu",  # Use the determined device
    )

    # --- Evaluate All Metrics ---

    all_results = evaluator.evaluate(log_wandb=False, log_examples=True, run_name="test_run")
    print(all_results)
