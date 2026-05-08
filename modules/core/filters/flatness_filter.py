"""Filter that leaves molecules with flatness."""

from rdkit.Chem import Mol

from modules.core.features.flatness import get_flatness_mol
from modules.core.filters.generic_filter import GenericMoleculeFilter


class FlatnessFilter(GenericMoleculeFilter):
    """Filter that leaves molecules with flatness score below specified threshold.

    Args:
        max_attempts (int): Maximum number of attempts to generate conformers.
        num_conformers (int): Number of conformers to generate for flatness calculation.
        max_flatness (float): Maximum allowed flatness value for the molecules to be kept.

    """

    def __init__(
        self,
        max_attempts: int = 100,
        num_conformers: int = 10,
        max_flatness: float = 4.0,
    ):
        super().__init__()
        self.max_attempts = max_attempts
        self.num_conformers = num_conformers
        self.max_flatness = max_flatness

    # pylint: disable=arguments-differ
    def apply(self, molecules: list[Mol], return_flatness: bool = False, **kwargs) -> list[Mol] | tuple[list[Mol], list[tuple[float, Mol]]]:  # type: ignore[override]
        """Apply the filter to a list of RDKit molecules.

        Args:
            molecules (list[Mol]): The list of RDKit molecules to filter.
            return_flatness (bool): If True, return the flatness values of the molecules.
            **kwargs: Additional keyword arguments (for API compatibility).

        Returns:
            if return_flatness is True:
                a tuple of two lists: filtered_molecules and list of tuples with flatness scores and molecules.
            else:
                list[Mol]: The list of RDKit molecules that passed the filter.

        """
        filtered_molecules = []
        flatness_scores: list[tuple[float, Mol]] = []
        for mol in molecules:
            f = get_flatness_mol(
                molecule=mol,
                plot_visualization=False,
                num_conformers=self.num_conformers,
                max_attempts=self.max_attempts,
            )

            flatness_scores.append((f, mol))
            if f and f <= self.max_flatness:
                filtered_molecules.append(mol)
        if return_flatness:
            return filtered_molecules, flatness_scores
        return filtered_molecules

    def filter_from_property(self, properties: dict) -> bool:
        """Reads properties from a dictionary (database) and decides whether to filter the molecule.

        Args:
            properties (dict): A dictionary containing molecule properties, including 'flatness'.

        Returns:
            bool: True if the molecule passes the flatness filter, False otherwise.

        """
        flatness = properties.get("flatness", None)
        if flatness is None:
            return False  # Filter out if no flatness data is available
        return flatness <= self.max_flatness
