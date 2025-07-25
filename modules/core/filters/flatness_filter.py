"""Filter that leaves molecules with flatness."""
from rdkit.Chem import Mol

from modules.core.features.flatness import get_flatness_mol
from modules.core.filters.generic_filter import GenericMoleculeFilter


class FlatnessFilter(GenericMoleculeFilter):
    """Filter that leaves molecules with flatness.

    Args:
        max_attempts (int): Maximum number of attempts to generate conformers.
        num_conformers (int): Number of conformers to generate for flatness calculation.
        max_flatness (float): Maximum allowed flatness value for the molecules to be kept.
    """

    def __init__(self, max_attempts: int = 1, num_conformers: int = 20, max_flatness: float = 0.5):
        """
        Initialize the filter.
        """
        super().__init__()
        self.max_attempts = max_attempts
        self.num_conformers = num_conformers
        self.max_flatness = max_flatness

    # pylint: disable=arguments-differ
    def apply(self, molecules: list[Mol], return_flatness: bool = False) -> list[Mol] | list[tuple[float, Mol]]:
        """
        Apply the filter to a list of RDKit molecules.

        Args:
            molecules (list[Mol]): The list of RDKit molecules to filter.
            return_flatness (bool): If True, return the flatness values of the molecules.
        Returns:
            if return_flatness is True:
                list[tuple[float, Mol]]: A list of tuples containing the flatness value and the corresponding RDKit molecule.
            else:
                list[Mol]: The list of RDKit molecules that passed the filter.
        """
        flatness = []
        for mol in molecules:
            f = get_flatness_mol(mol, False, num_conformers=self.num_conformers, max_attempts=self.max_attempts)
            # TODO: consider filtering flatness / flatness being None
            if f:
                flatness.append((f, mol))
        flatness = sorted(flatness, key=lambda x: x[0])
        if return_flatness:
            return flatness
        return [x[1] for x in flatness]
