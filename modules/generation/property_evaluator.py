"""Property evaluator class, used to evaluate a set of defined properties for a given molecule(s)."""
import logging
import math

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from modules.core.enums import MoleculeType
from modules.core.features.csm_runner import CSMRunner
from modules.core.features.flatness import get_flatness_mol
from modules.core.features.pore_size import estimate_pore_size
from modules.core.filters.conjugation_filter import ConjugationFilter
from modules.core.filters.smarts_filter import SMARTSFilter
from modules.core.filters.steric_hindrance_filter import StericHindranceFilter
from modules.generation.utils import score_value_exponential

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Define the log message format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)


class PropertyEvaluator:
    """Property evaluator class, used to evaluate a set of defined properties for a given molecule(s).

    Args:
        molecule_type (MoleculeType): Type of the molecule to evaluate, e.g., MoleculeType.SUBSTRATE or MoleculeType.NODE. Defaults to MoleculeType.SUBSTRATE for compatibility with previous versions.
        num_criteria (int, optional): Number of criteria to evaluate (e.g. computed scores will be divided by num_criteria so that the final range is 0-1). Defaults to 5.
        reference_smiles (str, optional): Path to a .smi file containing known SMILES strings that will be used to evaluate similarity between the generated molecules and reference ones. Defaults to None.
        **kwargs: Additional keyword arguments for filters.
    """

    def __init__(self, molecule_type: MoleculeType = MoleculeType.SUBSTRATE, num_criteria: int = 5, reference_smiles: str | None = None, **kwargs):
        if molecule_type not in [MoleculeType.SUBSTRATE, MoleculeType.NODE]:
            raise ValueError(f"Invalid molecule type: {molecule_type}. Must be either MoleculeType.SUBSTRATE or MoleculeType.NODE.")
        self.molecule_type = molecule_type
        self.num_criteria = num_criteria

        self.conjugation_filter = ConjugationFilter()
        self.smarts_filter = SMARTSFilter(kwargs.get("smarts", None))
        self.steric_hindrance_filter = StericHindranceFilter()

        # self.point_group_symmetry_filter = PointGroupSymmetryFilter(kwargs.get("translation_table_path", "data/symmetries/symmetry_translation.csv"))
        # self.symmetry_filter = SymmetryFilter(kwargs.get("min_isomorphic_nodes", 8), kwargs.get("types_to_check", None))

        self.csm_runner = CSMRunner()

        self.fingerprint_generator = GetMorganGenerator(radius=2)

        self.reference_smiles = None

        if reference_smiles:
            self.reference_smiles = pd.read_csv(reference_smiles, sep=" ", header=None)
            self.reference_smiles.columns = ["smiles"]
            self.reference_smiles["fingerprint"] = self.reference_smiles["smiles"].apply(lambda x: self.fingerprint_generator.GetFingerprint(Chem.MolFromSmiles(x)))

    def evaluate(self, smiles: str) -> float:
        """Evaluate the properties for a given molecule.

        Args:
            smiles (str): The SMILES representation of molecule to evaluate.

        Returns:
            float: The evaluated score for the molecule.
        """
        if len(smiles) == 0 or smiles is None:
            return 1e-10  # Set to a small positive value to avoid negative scores
        molecule = Chem.MolFromSmiles(smiles)

        if molecule is None:
            return 1e-10

        conjugation_score = self._check_conjugation(molecule)
        flatness_score = self._calculate_flatness(molecule)
        steric_hindrance_score = self._calculate_steric_hindrance(molecule)

        if self.reference_smiles is not None:
            similarity_score = self._calculate_mean_similarity(molecule)
        else:
            similarity_score = 1e-10

        if self.molecule_type == MoleculeType.SUBSTRATE:
            smarts_score = self._check_smarts(molecule)

            total_score = smarts_score + conjugation_score + flatness_score + similarity_score + steric_hindrance_score

            logger.debug(
                f" {smiles} | {self.molecule_type} \n"
                f"Properties: Total score: {total_score}, SMARTS: {smarts_score}, Conjugation: {conjugation_score}, Flatness: {flatness_score}, Mean Similarity: {similarity_score}, Steric Hindrance: {steric_hindrance_score}"
            )

        else:
            symmetry_score = self._check_symmetry(molecule)

            total_score = conjugation_score + symmetry_score + flatness_score + similarity_score + steric_hindrance_score

            logger.debug(
                f" {smiles} | {self.molecule_type} \n"
                f"Properties: Total score: {total_score}, Conjugation: {conjugation_score}, Symmetry: {symmetry_score}, Flatness: {flatness_score}, Mean Similarity: {similarity_score}, Steric Hindrance: {steric_hindrance_score}"
            )

        if total_score < 0 or math.isnan(total_score):
            logger.debug("Total score set to 1e-10.")
            return 1e-10  # Set to a small positive value to avoid negative scores
        return total_score / self.num_criteria

    @staticmethod
    def _calculate_pore_size(molecule: Mol) -> float:
        try:
            pore_size = estimate_pore_size(molecule)
            return score_value_exponential(pore_size, min_val=2, max_val=50)
        except AttributeError as e:
            logger.error(f"Error calculating pore size: {e}")
            return 1e-10

    @staticmethod
    def _calculate_flatness(molecule: Mol) -> float:
        try:
            flatness_error = get_flatness_mol(molecule)
            if flatness_error and not math.isnan(flatness_error):
                return score_value_exponential(flatness_error, min_val=1e-10, max_val=6)
            return 1e-10
        except AttributeError as e:
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
        result = self.csm_runner.analyze_molecule(molecule, point_groups=["c2", "c3", "c4"], exact=False)
        if result is None or result.lowest_csm is None:
            # logger.warning("No symmetry/conformation found for the molecule.")
            return 1e-10  # Return a small positive value to avoid negative scores
        return score_value_exponential(result.lowest_csm[1], min_val=1e-10, max_val=5.0, decay_rate=0.1)

    # def _check_symmetry(self, molecule: Mol) -> float:
    #     """
    #     Evaluate the symmetry of the molecule.
    #     """
    #     symmetry_group, symmetrical_point = self.point_group_symmetry_filter.apply([molecule], return_point_group_symmetry=True)
    #
    #     # Check if the molecule is symmetrical according to pymatgen
    #     if len(symmetry_group) == 0:
    #         symmetry_group = ["C1"]
    #
    #     symmetrical_graph = self.symmetry_filter.apply([molecule])
    #
    #     if len(symmetrical_graph) == 1:
    #         if symmetry_group[0] == "Cs":
    #             return 0.75  # Cs symmetry with graph
    #         elif symmetry_group[0] == "C1":
    #             return 0.25  # no-symmetry with graph
    #         else:
    #             return 1  # desired allowed symmetries with graph
    #     else:
    #         if symmetry_group[0] == "Cs":
    #             return 0.5  # Cs symmetry with graph
    #         elif symmetry_group[0] == "C1":
    #             return 0.0  # C1 no-symmetry without graph
    #         else:
    #             return 1  # desired allowed symmetries without graph

    def _calculate_mean_similarity(self, molecule: Mol) -> float:
        """
        Calculate the mean similarity of the molecule to a set of known molecules.
        """
        fingerprint = self.fingerprint_generator.GetFingerprint(molecule)
        similarities = self.reference_smiles["fingerprint"].apply(lambda x: Chem.DataStructs.TanimotoSimilarity(fingerprint, x))
        return similarities.mean()

    def _calculate_steric_hindrance(self, molecule: Mol) -> float:
        """
        Calculate the steric hindrance of the molecule.
        """
        steric_hindrance = self.steric_hindrance_filter.apply([molecule])
        return float(len(steric_hindrance))


if __name__ == "__main__":
    evaluator = PropertyEvaluator(reference_smiles="data/raw/experts_merged.smi")
    evaluator.evaluate(
        "N#Cc%19ccc(c%17cc%15c(cc(c%14ccc(c%13nc(c6ccc(c4cc2c(cc(c1ccc(C#N)cc1)n2c3ccc(C#N)cc3)n4c5ccc(C#N)cc5)cc6)nc(c%12ccc(c%10cc8c(cc(c7ccc(C#N)cc7)n8c9ccc(C#N)cc9)n%10c%11ccc(C#N)cc%11)cc%12)n%13)cc%14)n%15c%16ccc(C#N)cc%16)n%17c%18ccc(C#N)cc%18)cc%19"
    )
    # evaluator.evaluate("N#Cc1cc(C#N)c(F)c(C#N)c1F")
    # flatness -> nan for
    # C12=CC=C(C=C1)CCC2
    # C12=CC=C(C=C1)COC=C2
    # C1=C(F)C(Cl)=CC=C1C(=O)C
    # C12=CC=C(C=C1)C(CC2)NC=CC
    # score = evaluator.evaluate("XYZ")
    # print(score)
