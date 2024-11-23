"""Abstract class for a generic molecule filter."""
from abc import ABC, abstractmethod


class GenericMoleculeFilter(ABC):
    """
    Abstract base class for molecule filters.

    Subclasses must implement the `apply` method to filter molecules
    represented as SMILES strings.
    """

    @abstractmethod
    def apply(self, smiles: list[str], **kwargs) -> list[str]:
        """
        Filters the given list of SMILES strings.

        Args:
            smiles (List[str]): A list of SMILES strings representing molecules.
            **kwargs: Additional keyword arguments.

        Returns:
            List[str]: A list of SMILES strings that pass the filter.
        """
        raise NotImplementedError
