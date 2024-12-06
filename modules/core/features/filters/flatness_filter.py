"""Filter that leaves molecules with flatness."""
from tqdm import tqdm

from modules.core.features.filters.generic_filter import GenericMoleculeFilter
from modules.core.features.flatness import get_flatness_smiles


class FlatnessFilter(GenericMoleculeFilter):
    """Filter that leaves molecules with flatness."""

    def apply(self, smiles: list[str], **kwargs) -> list[str]:
        """
        Apply the filter to a list of SMILES strings.

        Args:
            smiles (list[str]): The list of SMILES strings to filter.
        Returns:
            list[str]: The list of SMILES strings that passed the filter.
        """
        flatness = []
        for sml in tqdm(smiles, total=len(smiles), desc="Calculating flatness"):
            f = get_flatness_smiles(sml, sample_size=20)
            flatness.append((f, sml))
        flatness = sorted(flatness, key=lambda x: x[0])
        return [x[1] for x in flatness]
