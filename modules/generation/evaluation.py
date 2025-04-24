"""
Evaluates generative models for molecule design using standard metrics.

This module provides the MoleculeGenerationEvaluator class to calculate
metrics like Validity, Uniqueness, Novelty, Internal Diversity, and FCD
for sets of generated SMILES strings.
"""

import logging
from itertools import product

import numpy as np
from fcd_torch import FCD
from rdkit import Chem, DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.rdBase import DisableLog
from tqdm.auto import tqdm

import wandb

DisableLog("rdApp.*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
logger = logging.getLogger(__name__)


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
        self,
        generated_smiles: list[str],
        training_smiles: list[str] | None = None,
        reference_smiles: list[str] | None = None,
        n_jobs: int = 1,
        device: str = "cpu",
        batch_size: int = 512,
    ):
        self.generated_smiles: list[str] = generated_smiles
        self.training_smiles: list[str] = training_smiles or []
        self.reference_smiles: list[str] = reference_smiles or []

        # FCD Specific Parameters
        self.n_jobs: int = n_jobs
        self.device: str = device  # User provides the specific device string
        self.batch_size: int = batch_size

        self._fcd_calculator: FCD | None = None

        # --- Preprocess SMILES ---
        # This step validates and canonicalizes SMILES, preparing them for metric calculation.
        # It stores the *count* of valid molecules before deduplication (n_generated_valid)
        # and the *list* of unique valid canonical SMILES (valid_generated_smiles_canon).
        self.valid_generated_mols, self.valid_generated_smiles_canon, self.n_generated_valid = self._preprocess_smiles_list(self.generated_smiles, "Generated")
        self.n_generated_total: int = len(self.generated_smiles)

        _, self.valid_training_smiles_canon, _ = self._preprocess_smiles_list(self.training_smiles, "Training")

        _, self.valid_reference_smiles_canon, _ = self._preprocess_smiles_list(self.reference_smiles, "Reference")

    @staticmethod
    def _preprocess_smiles_list(smiles_list: list[str], list_name: str) -> tuple[list[Chem.Mol], list[str], int]:
        """
        Validates and canonicalizes a list of SMILES strings.

        Filters out invalid SMILES and identifies unique valid representations.

        Args:
            smiles_list: The list of SMILES strings to process.
            list_name: A descriptive name for the list (e.g., "Generated").

        Returns:
            A tuple containing:
            - list[Mol]: List of valid RDKit Mol objects corresponding to unique valid SMILES.
            - list[str]: List of unique valid (and potentially canonicalized) SMILES strings.
            - int:       Count of valid molecules found *before* deduplication.
        """
        valid_mols_dict: dict[str, Chem.Mol] = {}  # Tracks unique valid SMILES -> Mol
        processed_keys_set: set[str] = set()  # Tracks unique keys (canonical or original valid)
        valid_mol_count = 0  # Count valid mols before deduplication

        if not smiles_list:
            logger.info("%s list is empty, skipping preprocessing.", list_name)
            return [], [], 0

        for smiles in tqdm(smiles_list, desc=f"Processing {list_name} SMILES", leave=False):
            try:
                mol = Chem.MolFromSmiles(smiles)
                canon_smiles = Chem.MolToSmiles(mol, canonical=True)
                valid_mol_count += 1
                if canon_smiles not in processed_keys_set:
                    valid_mols_dict[canon_smiles] = mol
                    processed_keys_set.add(canon_smiles)

            except RuntimeError as e:
                logger.error(f"Error processing SMILES '{smiles}': {e}")

        # Extract lists from the dictionary holding unique entries
        unique_valid_smiles = list(valid_mols_dict.keys())
        unique_valid_mols = list(valid_mols_dict.values())

        return unique_valid_mols, unique_valid_smiles, valid_mol_count

    def evaluate(self, log_wandb: bool = False, log_examples: bool = False, project: str = "molecule-generation", run_name: str | None = None, num_examples: int = 5) -> dict[str, float]:
        """
        Runs the calculation of defined metrics.

        Args:
            log_wandb (bool): If True, logs the results to Weights & Biases. Defaults to False.
            log_examples (bool): If True, logs examples of valid generated SMILES. Requires wandb to be True. Defaults to False
            project (str): The name of the experiment. Defaults to "molecule-generation".
            run_name (str): The name of the run. Defaults to None.
            num_examples (int): The number of examples to log. Defaults to 5.

        Returns:
            A dictionary containing the results. Keys are metric names (str),
            values are the calculated scores (float or np.nan if calculation
            failed or requirements were not met).
        """
        results = {
            "validity": self.calculate_validity(),
            "uniqueness": self.calculate_uniqueness(),
            "novelty_wrt_training_set": self.calculate_novelty(set(self.valid_training_smiles_canon)),
            "novelty_wrt_reference_set": self.calculate_novelty(set(self.valid_reference_smiles_canon)),
            "internal_diversity": self.calculate_internal_diversity(),
            "fcd": self.calculate_fcd() if self.reference_smiles else np.nan,
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

        return results

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

    def calculate_internal_diversity(self, p: int = 1, fp_radius: int = 2, fp_bits: int = 2048) -> float:
        """
        Calculates internal diversity using Tanimoto similarity of Morgan fingerprints.
        This metric detects a common failure case of generative models—mode collapse. With mode collapse,
        the model produces a limited variety of samples, ignoring some areas of the chemical space. A higher
        value of this metric corresponds to higher diversity in the generated set.

        IntDiv_p = 1 - [ (1/|G|^2) * Sum_{m1,m2 in G}( T(m1,m2)^p ) ]^(1/p)
        Calculated over the set of *unique valid* generated molecules.

        Args:
            p: The power parameter for the internal diversity calculation. Defaults to 1.
            fp_radius: Morgan fingerprint radius. Defaults to 2.
            fp_bits: Morgan fingerprint number of bits. Defaults to 2048.

        Returns:
            Internal diversity score [0.0, 1.0]. Returns 0.0 if fewer than 2
            unique valid molecules exist.
        """
        # Operate on the unique valid molecules obtained from preprocessing
        unique_valid_mols = self.valid_generated_mols

        fingerprints = []
        for mol in unique_valid_mols:
            try:
                fp = GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
                fingerprints.append(fp)
            except RuntimeError as e:
                logger.error(f"Error calculating fingerprint: {e}")

        num_fingerprints = len(fingerprints)
        if num_fingerprints < 2:
            logger.warning(f"Could not generate enough valid fingerprints {num_fingerprints} for diversity calculation.")
            return 0.0

        sum_sim_p = 0.0

        # Iterate over all n_fps * n_fps pairs (m1, m2), including (m, m)
        for i, j in product(range(num_fingerprints), repeat=2):
            try:
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                # Handle potential 0^p case carefully if p isn't integer, although p is int here
                sum_sim_p += np.power(sim, p)
            except RuntimeError as e:
                logger.error(f"Error calculating Tanimoto similarity: {e}")

        average_similarity = sum_sim_p / (num_fingerprints**2)
        if p == 1:
            root_mean_sim_p = average_similarity
        else:
            # Ensure argument for root is non-negative
            root_mean_sim_p = np.power(max(0.0, average_similarity), 1.0 / p)

        int_div = 1.0 - root_mean_sim_p
        return int_div

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
            raise ValueError("Cannot calculate FCD: No unique valid generated molecules found.")

        if not ref_smiles_list:
            raise ValueError("Cannot calculate FCD: No unique valid reference molecules found.")

        try:
            # Lazy initialization of FCD calculator
            if self._fcd_calculator is None:
                # Initialize FCD with user-provided device string
                self._fcd_calculator = FCD(device=self.device, n_jobs=self.n_jobs, batch_size=self.batch_size)

            logger.debug(f"Calculating FCD between {len(gen_smiles_list)} unique generated and {len(ref_smiles_list)} unique reference molecules...")

            # --- Use the callable FCD object API as requested ---
            fcd_score = self._fcd_calculator(ref_smiles_list, gen_smiles_list)

            return float(fcd_score)  # Ensure result is float

        except RuntimeError as e:
            logger.error(f"Error calculating FCD: {e}")
            return 0.0


if __name__ == "__main__":
    generated_smiles_example = [
        "CCO",  # duplicate of training / generated
        "CCC",  # novel, valid
        "c1ccccc1",  # duplicate of training
        "invalid-smiles-string",  # invalid
        "CC(=O)O",  # duplicate of training
        "CCO",  # duplicate of training / generated
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

    all_results = evaluator.evaluate(log_wandb=True, log_examples=True)
