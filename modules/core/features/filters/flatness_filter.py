"""Filter that leaves molecules with flatness."""
from rdkit.Chem import Mol

from modules.core.features.filters.generic_filter import GenericMoleculeFilter
from modules.core.features.flatness import get_flatness_mol


class FlatnessFilter(GenericMoleculeFilter):
    """Filter that leaves molecules with flatness."""

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
            f = get_flatness_mol(mol, plot_visualization=False, sample_size=20)
            flatness.append((f, mol))
        flatness = sorted(flatness, key=lambda x: x[0])
        if return_flatness:
            return flatness
        return [x[1] for x in flatness]
