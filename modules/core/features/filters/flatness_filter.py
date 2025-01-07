"""Filter that leaves molecules with flatness."""
from rdkit.Chem import Mol
from tqdm import tqdm

from modules.core.features.filters.generic_filter import GenericMoleculeFilter
from modules.core.features.flatness import get_flatness_mol


class FlatnessFilter(GenericMoleculeFilter):
    """Filter that leaves molecules with flatness."""

    def apply(self, molecules: list[Mol], **kwargs) -> list[Mol]:
        """
        Apply the filter to a list of RDKit molecules.

        Args:
            molecules (list[Mol]): The list of RDKit molecules to filter.
        Returns:
            list[Mol]: The list of RDKit molecules that passed the filter.
        """
        flatness = []
        for mol in tqdm(molecules, total=len(molecules), desc="Calculating flatness"):
            f = get_flatness_mol(mol, plot_visualization=False, sample_size=20)
            flatness.append((f, mol))
        flatness = sorted(flatness, key=lambda x: x[0])
        return [x[1] for x in flatness]
