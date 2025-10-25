"""Test PropertyEvaluator"""
import math

import pytest

# pylint: disable=import-error
from modules.generation.property_evaluator import PropertyEvaluator


@pytest.mark.parametrize(
    "smiles",
    [
        "C12=CC=C(C=C1)CCC2",
        "N#Cc%19ccc(c%17cc%15c(cc(c%14ccc(c%13nc(c6ccc(c4cc2c(cc(c1ccc(C#N)cc1)n2c3ccc(C#N)cc3)n4c5ccc(C#N)cc5)cc6)nc(c%12ccc(c%10cc8c(cc(c7ccc(C#N)cc7)n8c9ccc(C#N)cc9)n%10c%11ccc(C#N)cc%11)cc%12)n%13)cc%14)n%15c%16ccc(C#N)cc%16)n%17c%18ccc(C#N)cc%18)cc%19",
        "Xyz",
        "",
    ],
)
def test_property_evaluator(smiles):
    """Test whether returned score is positive and a number, not nan."""
    evaluator = PropertyEvaluator()

    score = evaluator.evaluate(smiles)

    assert math.isnan(score) is False, "score cannot be nan"
    assert score > 0, "score must be greater than 0"
