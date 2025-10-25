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


class PriorType(Enum):
    VANILLA = "vanilla"
    CHEMBL35 = "chembl35"


class Recipe(Enum):
    SAMPLING = "sampling"
    FT = "ft"
    RL = "rl"
    RL_INCEPTION = "rl_inception"
    FT_RL = "ft_rl"
    FT_RL_INCEPTION = "ft_rl_inception"
