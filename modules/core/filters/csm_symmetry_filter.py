"""Filter that leaves molecules with CSM symmetry below threshold."""

import logging

from rdkit.Chem import Mol

from modules.core.features.csm_runner import CSMRunner
from modules.core.filters.generic_filter import GenericMoleculeFilter

logger = logging.getLogger(__name__)  # __name__ ensures the logger is specific to this module


class CSMSymmetryFilter(GenericMoleculeFilter):
    """
    A filter that retains molecules with a CSM symmetry below a specified threshold.

    Args:
        symmetry_measure_threshold (float): The threshold for the CSM symmetry measure. Molecules with
            a symmetry measure below this threshold will be retained.
        normalize_score (bool): Whether to normalize the symmetry score by number of atoms. Default is True.
        evaluated_symmetry_groups (list[str] | None): A list of symmetry groups to evaluate.
            If None, defaults to ["c2", "c3", "c4"].
    """

    def __init__(
        self,
        symmetry_measure_threshold: float = 0.2,
        normalize_score: bool = True,
        evaluated_symmetry_groups: list[str] | None = None,
    ):
        super().__init__()
        self.symmetry_measure_threshold = symmetry_measure_threshold
        self.csm_runner = CSMRunner()
        self.normalize_score = normalize_score

        if evaluated_symmetry_groups is None:
            self.evaluated_symmetry_groups = ["c2", "c3", "c4"]
        else:
            self.evaluated_symmetry_groups = evaluated_symmetry_groups

    # pylint: disable=arguments-differ
    def apply(self, molecules: list[Mol], **kwargs) -> list[Mol]:
        """
        Apply the filter to a list of RDKit Mol objects.

        Args:
            molecules (list[Mol]): The list of RDKit Mol objects to filter.
        Returns:
            list[Mol]: The list of RDKit Mol objects that passed the filter.

        """
        filtered_molecules = []
        for mol in molecules:
            if not isinstance(mol, Mol):
                logger.warning("Skipping non-Mol object: %s", mol)
                continue

            csm_result = self.csm_runner.analyze_molecule(mol, point_groups=self.evaluated_symmetry_groups, exact=False)

            if self.normalize_score:
                if csm_result and csm_result.lowest_csm_normalized and csm_result.lowest_csm_normalized[1] < self.symmetry_measure_threshold:
                    filtered_molecules.append(mol)
            else:
                if csm_result and csm_result.lowest_csm and csm_result.lowest_csm[1] < self.symmetry_measure_threshold:
                    filtered_molecules.append(mol)

        return filtered_molecules

    def filter_from_property(self, properties: dict) -> bool:
        """
        Reads properties from a dictionary (database) and decides whether to filter the molecule.
        """
        csm_data = properties.get("normalized_csm", {})
        if not csm_data:
            return False  # Filter out if no CSM data is available

        if self.normalize_score:
            lowest_csm_normalized = csm_data.get("normalized_csm", None)
            if lowest_csm_normalized and lowest_csm_normalized < self.symmetry_measure_threshold:
                return True  # Do not filter out
        else:
            lowest_csm = csm_data.get("lowest_csm", None)
            if lowest_csm and lowest_csm < self.symmetry_measure_threshold:
                return True  # Do not filter out

        return False  # Filter out
