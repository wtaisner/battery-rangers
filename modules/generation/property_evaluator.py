"""Property evaluator class, used to evaluate a set of defined properties for a given molecule(s)."""
import logging
import math

import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from modules.core.database import MoleculeDB
from modules.core.enums import MoleculeType
from modules.core.features.csm_runner import CSMRunner
from modules.core.features.flatness import get_flatness_mol
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
    """
    Property evaluator class, correctly using a database as a write-through cache.
    """

    def __init__(
        self,
        molecule_type: MoleculeType = MoleculeType.SUBSTRATE,
        # reference_smiles: str | None = None,
        csm_threshold: float = 0.2,
        flatness_threshold: float = 4.0,
        db_file: str = "modules/bionemo/data/mol_db/substrate_properties.db",
        **kwargs,
    ):
        if molecule_type not in [MoleculeType.SUBSTRATE, MoleculeType.NODE]:
            raise ValueError(f"Invalid molecule type: {molecule_type}. Must be either MoleculeType.SUBSTRATE or MoleculeType.NODE.")

        # generic
        self.molecule_type = molecule_type
        self.num_criteria = 5 if molecule_type == MoleculeType.NODE else 6
        self.db_file = db_file if molecule_type == MoleculeType.SUBSTRATE else "modules/bionemo/data/mol_db/node_properties.db"
        self.database = MoleculeDB(self.db_file)

        # filters/estimators
        self.conjugation_filter = ConjugationFilter()
        self.smarts_filter = SMARTSFilter(kwargs.get("smarts", None))
        self.steric_hindrance_filter = StericHindranceFilter()
        self.csm_runner = CSMRunner()
        self.fingerprint_generator = GetMorganGenerator(radius=2)

        if self.molecule_type == MoleculeType.SUBSTRATE:
            reference_smiles = "data/raw/substrate/ctf_train.smi"
        else:
            reference_smiles = "data/raw/node/ctf_train.smi"

        # if file ends with .smi assume it is a space separated file with smiles only (for REINVENT)
        if reference_smiles.endswith(".smi"):
            self.reference_smiles = pd.read_csv(reference_smiles, sep=" ", header=None)
        elif reference_smiles.endswith(".csv"):
            self.reference_smiles = pd.read_csv(reference_smiles)

        self.reference_smiles.columns = ["canon_smiles"]
        try:
            self.reference_smiles["fingerprint"] = self.reference_smiles["canon_smiles"].apply(lambda x: self.fingerprint_generator.GetFingerprint(Chem.MolFromSmiles(x)))
        except IndexError as e:
            logger.error(f"Error processing reference SMILES file: {e}")
            self.reference_smiles = None

        logger.info(f"Reference SMILES loaded with {len(self.reference_smiles) if self.reference_smiles is not None else 'EMPTY'} entries.")

        # thresholds
        self.csm_threshold = csm_threshold
        self.flatness_threshold = flatness_threshold
        if self.molecule_type == MoleculeType.NODE:
            self.flatness_threshold = 5.0
            logger.info(f"Molecule type set to NODE. Adjusted csm_threshold to {self.csm_threshold} and flatness_threshold to {self.flatness_threshold}.")

    def evaluate(self, smiles: str) -> float:
        """
        Evaluate properties for a molecule, using the database as a write-through cache.
        """
        if not smiles:
            return 1e-10

        # canonicalize SMILES before using it as a key
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 1e-10
        canon_smiles = Chem.MolToSmiles(mol)

        # 1. Check database for the CANONICAL smiles (Cache Hit)
        db_properties = self.database.get_molecule_properties(canon_smiles)
        if db_properties:
            logger.debug(f"'{canon_smiles}' found in database. Using cached properties.")
            return self._calculate_score_from_properties(db_properties)

        # 2. If not in DB, calculate RAW properties (Cache Miss)
        logger.debug(f"'{canon_smiles}' not in database. Calculating properties.")
        try:
            # This dictionary will hold the RAW values to be stored in the database
            raw_properties = {
                "canon_smiles": canon_smiles,
                "flatness": get_flatness_mol(mol) or float("inf"),  # Store raw flatness error
                "normalized_csm": self._get_raw_csm(mol),  # Store raw CSM value
                "similarity": self._calculate_mean_similarity(mol) if self.reference_smiles is not None else 0.0,
                "selfies": sf.encoder(canon_smiles),  # Convert to SELFIES for DB
                "smarts_filter": len(self.smarts_filter.apply([mol])) > 0,
                "conjugation_filter": len(self.conjugation_filter.apply([mol])) > 0,
                "steric_hindrance": len(self.steric_hindrance_filter.apply([mol])) > 0,
            }

            # 3. Write the newly calculated properties to the database
            self.database.add_molecule(**raw_properties)
            logger.debug(f"Added '{canon_smiles}' properties to the database.")

            # 4. Calculate the final score from the new RAW properties
            return self._calculate_score_from_properties(raw_properties)

        except Exception as e:
            logger.error(f"Failed to evaluate or add '{canon_smiles}' to database: {e}", exc_info=True)
            return 1e-10

    def _calculate_score_from_properties(self, properties: dict) -> float:
        """
        Calculates the final score from a dictionary of RAW properties.
        This function is now the single source of truth for scoring.
        """
        # Apply scoring functions to RAW values from the properties dict
        flatness_score = score_value_exponential(properties["flatness"], min_val=1e-10, max_val=self.flatness_threshold)
        symmetry_score = score_value_exponential(properties["normalized_csm"], min_val=1e-10, max_val=self.csm_threshold)

        # Convert boolean filter results to scores
        conjugation_score = 1.0 if properties["conjugation_filter"] else 1e-10
        steric_hindrance_score = 1.0 if properties["steric_hindrance"] else 1e-10

        similarity_score = properties["similarity"] if properties["similarity"] > 0 else 1e-10

        if self.molecule_type == MoleculeType.SUBSTRATE:
            smarts_score = 1.0 if properties["smarts_filter"] else 1e-10
            total_score = smarts_score + conjugation_score + flatness_score + similarity_score + steric_hindrance_score + symmetry_score
        else:  # MoleculeType.NODE
            total_score = conjugation_score + symmetry_score + flatness_score + similarity_score + steric_hindrance_score

        if total_score < 0 or math.isnan(total_score):
            logger.debug(f"Total score for '{properties['canon_smiles']}' is invalid, returning 1e-10.")
            return 1e-10

        return total_score / self.num_criteria

    def _get_raw_csm(self, molecule: Mol) -> float:
        """Helper function to get the raw CSM value for database storage."""
        result = self.csm_runner.analyze_molecule(molecule, point_groups=["c2", "c3", "c4"], exact=False)
        if result and result.lowest_csm_normalized:
            return result.lowest_csm_normalized[1]
        return float("inf")  # Return a large value if not found, which will result in a low score

    def _calculate_mean_similarity(self, molecule: Mol) -> float:
        """Calculates the raw mean similarity of the molecule."""
        if self.reference_smiles is None:
            return 0.0
        try:
            fingerprint = self.fingerprint_generator.GetFingerprint(molecule)
            similarities = self.reference_smiles["fingerprint"].apply(lambda x: Chem.DataStructs.TanimotoSimilarity(fingerprint, x))
            return similarities.mean()
        except Exception as e:
            print(f"Error calculating similarity for {Chem.MolToSmiles(molecule)}: {e}")
            return 0.0


if __name__ == "__main__":
    evaluator = PropertyEvaluator(reference_smiles="data/raw/node/ctf_train.smi", molecule_type=MoleculeType.NODE)
    # evaluator.evaluate(
    #     "Cc1ccc(C=Cc2c(O)n(-c3ccccc3)c(=Nc3ccc(S(N)(=O)=O)cc3)n2-c2ccccc2)cc1"
    # )
    print(evaluator._calculate_mean_similarity(Chem.MolFromSmiles("Cc1ccc(C=Cc2c(O)n(-c3ccccc3)c(=Nc3ccc(S(N)(=O)=O)cc3)n2-c2ccccc2)cc1")))
