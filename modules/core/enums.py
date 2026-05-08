"""Enums for the project."""

from enum import Enum


class MoleculeType(Enum):
    """Enum for different types of molecules."""

    SUBSTRATE = "substrate"
    NODE = "node"

    def __str__(self):
        return self.value


class PriorType(Enum):
    """Enum for different REINVENT's prior."""

    VANILLA = "vanilla"
    CHEMBL35 = "chembl35"
    DISCOVERED = "discovered"


class Recipe(Enum):
    """Enum for different training/sampling recipes."""

    SAMPLING = "sampling"
    FT = "ft"
    RL = "rl"
    RL_INCEPTION = "rl_inception"
    FT_RL = "ft_rl"
    FT_RL_INCEPTION = "ft_rl_inception"
