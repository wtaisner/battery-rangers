"""Scoring module for REINVENT4 RL"""
__all__ = ["BatteryPropertiesSubstrate", "BatteryPropertiesNode"]

from dataclasses import dataclass

import numpy as np

# pylint: disable=import-error
from modules.core.enums import MoleculeType
from modules.generation.property_evaluator import PropertyEvaluator
from modules.REINVENT4.reinvent_plugins.components.add_tag import add_tag
from modules.REINVENT4.reinvent_plugins.components.component_results import ComponentResults


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    known_smiles_path: str | None = None


@add_tag("__component")
class BatteryPropertiesSubstrate:
    """Scoring component"""

    def __init__(self, params: Parameters):
        self.known_smiles_path = params.known_smiles_path
        self.property_evaluator = PropertyEvaluator(molecule_type=MoleculeType.SUBSTRATE)

    def __call__(self, smiles: list[str]) -> ComponentResults:
        """Evaluate the properties for a given molecule.

        Args:
            smiles (list[str]): The SMILES representation of molecule to evaluate.

        Returns:
            ComponentResults: The evaluated score for the molecule.
        """
        scores = []
        for smile in smiles:
            score = self.property_evaluator.evaluate(smile)
            scores.append(score)
        return ComponentResults(np.array([scores]))


@add_tag("__component")
class BatteryPropertiesNode:
    """Scoring component"""

    def __init__(self, params: Parameters):
        self.known_smiles_path = params.known_smiles_path
        self.property_evaluator = PropertyEvaluator(molecule_type=MoleculeType.NODE)

    def __call__(self, smiles: list[str]) -> ComponentResults:
        """Evaluate the properties for a given molecule.

        Args:
            smiles (list[str]): The SMILES representation of molecule to evaluate.

        Returns:
            ComponentResults: The evaluated score for the molecule.
        """
        scores = []
        for smile in smiles:
            score = self.property_evaluator.evaluate(smile)
            scores.append(score)
        return ComponentResults(np.array([scores]))
