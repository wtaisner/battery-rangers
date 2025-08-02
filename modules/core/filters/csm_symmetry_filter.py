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
        evaluated_symmetry_groups (list[str] | None): A list of symmetry groups to evaluate.
            If None, defaults to ["c2", "c3", "c4"].
    """

    def __init__(
        self,
        symmetry_measure_threshold: float = 5.0,
        evaluated_symmetry_groups: list[str] | None = None,
    ):
        super().__init__()
        self.symmetry_measure_threshold = symmetry_measure_threshold
        self.csm_runner = CSMRunner()

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

            if csm_result.lowest_csm and csm_result.lowest_csm[1] < self.symmetry_measure_threshold:
                filtered_molecules.append(mol)

        return filtered_molecules
