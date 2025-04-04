"""Property evaluator class, used to evaluate a set of defined properties for a given molecule(s)."""
import logging
import math

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from modules.core.features.filters.conjugation_filter import ConjugationFilter
from modules.core.features.filters.point_group_symmetry_filter import PointGroupSymmetryFilter
from modules.core.features.filters.smarts_filter import SMARTSFilter
from modules.core.features.filters.steric_hindrance_filter import StericHindranceFilter
from modules.core.features.flatness import get_flatness_mol
from modules.core.features.pore_size import estimate_pore_size

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Define the log message format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)


def score_value_exponential(value: float, min_val: float = 2.0, max_val: float = 50.0, decay_rate: float = 0.1) -> float:
    """
    Scores a value based on its proximity to the range [min_val, max_val].

    - Score is 1.0 if value is within the range [min_val, max_val].
    - Score decreases exponentially based on distance outside the range.

    Args:
      value: The numerical value to score.
      min_val: The lower bound of the optimal range.
      max_val: The upper bound of the optimal range.
      decay_rate: Controls how quickly the score drops off with distance.
                  A higher value means a faster drop.

    Returns:
      The calculated score (between 0 and 1.0).
    """
    if min_val <= value <= max_val:
        return 1.0
    if value < min_val:
        distance = min_val - value
        # Exponential decay: score = exp(-k * distance)
        return math.exp(-decay_rate * distance)
    # value > max_val
    distance = value - max_val
    # Exponential decay: score = exp(-k * distance)
    return math.exp(-decay_rate * distance)


class PropertyEvaluator:
    """Property evaluator class, used to evaluate a set of defined properties for a given molecule(s).

    Args:
        known_smiles_path (str, optional): Path to a .smi file containing known SMILES strings. Defaults to None.
    """

    def __init__(self, known_smiles_path: str | None = None, **kwargs):
        self.conjugation_filter = ConjugationFilter()
        self.smarts_filter = SMARTSFilter()
        self.point_group_symmetry_filter = PointGroupSymmetryFilter(kwargs.get("translation_table_path", "data/symmetries/symmetry_translation.csv"))
        self.steric_hindrance_filter = StericHindranceFilter()

        self.fingerprint_generator = GetMorganGenerator(radius=2)

        self.known_smiles = None

        if known_smiles_path:
            self.known_smiles = pd.read_csv(known_smiles_path, sep=" ", header=None)
            self.known_smiles.columns = ["smiles"]
            self.known_smiles["fingerprint"] = self.known_smiles["smiles"].apply(lambda x: self.fingerprint_generator.GetFingerprint(Chem.MolFromSmiles(x)))

    def evaluate(self, smiles: str) -> float:
        """Evaluate the properties for a given molecule.

        Args:
            smiles (str): The SMILES representation of molecule to evaluate.

        Returns:
            float: The evaluated score for the molecule.
        """
        if len(smiles) == 0:
            return 1e-10  # Set to a small positive value to avoid negative scores
        molecule = Chem.MolFromSmiles(smiles)

        # Evaluate the properties
        pore_size_score = self._calculate_pore_size(molecule)
        smarts_score = self._check_smarts(molecule)
        conjugation_score = self._check_conjugation(molecule)
        symmetry_score = self._check_symmetry(molecule)
        flatness_score = self._calculate_flatness(molecule)
        steric_hindrance_score = self._calculate_steric_hindrance(molecule)

        if self.known_smiles is not None:
            similarity_score = self._calculate_mean_similarity(molecule)
        else:
            similarity_score = 1e-10

        # Combine the scores
        total_score = pore_size_score + smarts_score + conjugation_score + symmetry_score + flatness_score + similarity_score + steric_hindrance_score

        logger.debug(
            f" {smiles} \n"
            f"Properties: Total score: {total_score}, Pore size: {pore_size_score}, SMARTS: {smarts_score}, Conjugation: {conjugation_score}, Symmetry: {symmetry_score}, Flatness: {flatness_score}, Mean Similarity: {similarity_score}, Steric Hindrance: {steric_hindrance_score}"
        )

        if total_score < 0 or math.isnan(total_score):
            logger.debug("Total score set to 1e-10.")
            total_score = 1e-10  # Set to a small positive value to avoid negative scores
        return total_score

    @staticmethod
    def _calculate_pore_size(molecule: Mol) -> float:
        try:
            pore_size = estimate_pore_size(molecule)
            return score_value_exponential(pore_size, min_val=2, max_val=50, decay_rate=0.1)
        except ValueError as e:
            logger.error(f"Error calculating pore size: {e}")
            return 1e-10

    @staticmethod
    def _calculate_flatness(molecule: Mol) -> float:
        try:
            flatness_error = get_flatness_mol(molecule)
            if not math.isnan(flatness_error):
                return score_value_exponential(flatness_error, min_val=1e-10, max_val=1.60, decay_rate=0.2)
            return 1e-10
        except ValueError as e:
            logger.error(f"Error calculating flatness: {e}")
            return 1e-10

    def _check_smarts(self, molecule: Mol) -> float:
        """
        Evaluate the SMARTS patterns of the molecule.
        """
        smarts = self.smarts_filter.apply([molecule])
        return float(len(smarts))

    def _check_conjugation(self, molecule: Mol) -> float:
        """
        Evaluate the conjugation of the molecule.
        """
        conjugated = self.conjugation_filter.apply([molecule])
        return float(len(conjugated))

    def _check_symmetry(self, molecule: Mol) -> float:
        """
        Evaluate the symmetry of the molecule.
        """
        symmetrical = self.point_group_symmetry_filter.apply([molecule])
        return float(len(symmetrical))

    def _calculate_mean_similarity(self, molecule: Mol) -> float:
        """
        Calculate the mean similarity of the molecule to a set of known molecules.
        """
        fingerprint = self.fingerprint_generator.GetFingerprint(molecule)
        similarities = self.known_smiles["fingerprint"].apply(lambda x: Chem.DataStructs.TanimotoSimilarity(fingerprint, x))
        return similarities.mean()

    def _calculate_steric_hindrance(self, molecule: Mol) -> float:
        """
        Calculate the steric hindrance of the molecule.
        """
        steric_hindrance = self.steric_hindrance_filter.apply([molecule])
        return float(len(steric_hindrance))


if __name__ == "__main__":
    evaluator = PropertyEvaluator(known_smiles_path="data/raw/experts_merged.smi")
    evaluator.evaluate(
        "N#Cc%19ccc(c%17cc%15c(cc(c%14ccc(c%13nc(c6ccc(c4cc2c(cc(c1ccc(C#N)cc1)n2c3ccc(C#N)cc3)n4c5ccc(C#N)cc5)cc6)nc(c%12ccc(c%10cc8c(cc(c7ccc(C#N)cc7)n8c9ccc(C#N)cc9)n%10c%11ccc(C#N)cc%11)cc%12)n%13)cc%14)n%15c%16ccc(C#N)cc%16)n%17c%18ccc(C#N)cc%18)cc%19"
    )
    evaluator.evaluate("C12=CC=C(C=C1)CCC2")
    # flatness -> nan for
    # C12=CC=C(C=C1)CCC2
    # C12=CC=C(C=C1)COC=C2
    # C1=C(F)C(Cl)=CC=C1C(=O)C
    # C12=CC=C(C=C1)C(CC2)NC=CC
