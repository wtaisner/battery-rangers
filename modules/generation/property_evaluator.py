"""Property evaluator class, used to evaluate a set of defined properties for a given molecule(s)."""
import logging
import math

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from modules.core.features.filters.conjugation_filter import ConjugationFilter
from modules.core.features.filters.point_group_symmetry_filter import PointGroupSymmetryFilter
from modules.core.features.filters.smarts_filter import SMARTSFilter
from modules.core.features.flatness import get_flatness_mol
from modules.core.features.pore_size import estimate_pore_size

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        self.molecule = None

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
        self.molecule = Chem.MolFromSmiles(smiles)

        # Evaluate the properties
        pore_size_score = self._calculate_pore_size()
        smarts_score = self._check_smarts()
        conjugation_score = self._check_conjugation()
        symmetry_score = self._check_symmetry()
        flatness_score = self._calculate_flatness()

        if self.known_smiles is not None:
            similarity_score = self._calculate_mean_similarity()
        else:
            similarity_score = 0.0

        logger.info(
            f" {smiles} | Properties: Pore size: {pore_size_score}, SMARTS: {smarts_score}, Conjugation: {conjugation_score}, Symmetry: {symmetry_score}, Flatness: {flatness_score}, Mean Similarity: {similarity_score}"
        )

        # Combine the scores
        total_score = pore_size_score + smarts_score + conjugation_score + symmetry_score + flatness_score + similarity_score
        if total_score < 0:
            # raise ValueError("Total score cannot be negative.")
            print("Total score set to 0.")
            total_score = 0
        return total_score

    def _calculate_pore_size(self) -> float:
        pore_size = estimate_pore_size(self.molecule)
        return score_value_exponential(pore_size, min_val=2, max_val=50, decay_rate=0.1)

    def _calculate_flatness(self) -> float:
        flatness_error = get_flatness_mol(self.molecule)
        return -flatness_error + 5  # since lower error is better, 1 is added to make it positive / only slightly negative

    def _check_smarts(self) -> int:
        """
        Evaluate the SMARTS patterns of the molecule.
        """
        smarts = self.smarts_filter.apply([self.molecule])
        return len(smarts)

    def _check_conjugation(self) -> int:
        """
        Evaluate the conjugation of the molecule.
        """
        conjugated = self.conjugation_filter.apply([self.molecule])
        return len(conjugated)

    def _check_symmetry(self) -> int:
        """
        Evaluate the symmetry of the molecule.
        """
        symmetrical = self.point_group_symmetry_filter.apply([self.molecule])
        return len(symmetrical)

    def _calculate_mean_similarity(self) -> float:
        """
        Calculate the mean similarity of the molecule to a set of known molecules.
        """
        fingerprint = self.fingerprint_generator.GetFingerprint(self.molecule)
        similarities = self.known_smiles["fingerprint"].apply(lambda x: Chem.DataStructs.TanimotoSimilarity(fingerprint, x))
        return similarities.mean()


if __name__ == "__main__":
    evaluator = PropertyEvaluator(known_smiles_path="data/raw/experts_merged.smi")
    score = evaluator.evaluate(
        "N#Cc%19ccc(c%17cc%15c(cc(c%14ccc(c%13nc(c6ccc(c4cc2c(cc(c1ccc(C#N)cc1)n2c3ccc(C#N)cc3)n4c5ccc(C#N)cc5)cc6)nc(c%12ccc(c%10cc8c(cc(c7ccc(C#N)cc7)n8c9ccc(C#N)cc9)n%10c%11ccc(C#N)cc%11)cc%12)n%13)cc%14)n%15c%16ccc(C#N)cc%16)n%17c%18ccc(C#N)cc%18)cc%19"
    )
