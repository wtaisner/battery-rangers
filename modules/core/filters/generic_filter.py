"""Abstract class for a generic molecule filter."""

from abc import ABC, abstractmethod
from typing import Any

from rdkit.Chem import Mol


class GenericMoleculeFilter(ABC):
    """Abstract base class for molecule filters.

    Subclasses must implement the `apply` method to filter molecules
    represented as RDKit Mol objects.
    """

    @abstractmethod
    def apply(self, molecules: list[Mol], **kwargs) -> Any:
        """Filters the given list of RDKit Mol objects.

        Args:
            molecules (List[mMol]): A list of RDKit Mol objects representing molecules.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Mol]: A list of RDKit Mol objects that pass the filter.

        """
        raise NotImplementedError

    @abstractmethod
    def filter_from_property(self, properties: dict) -> bool:
        """Reads properties from a dictionary (database) and decides whether to filter the molecule."""
        raise NotImplementedError
