"""Enums for the project."""
from enum import Enum


class MoleculeType(Enum):
    """
    Enum for different types of molecules.
    """

    SUBSTRATE = "substrate"
    NODE = "node"

    def __str__(self):
        return self.value
