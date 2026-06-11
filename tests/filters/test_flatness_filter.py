"""Test for the FlatnessFilter class"""

import pytest
from rdkit import Chem

from modules.core.filters.flatness_filter import FlatnessFilter


@pytest.mark.parametrize(
    "smiles, expected",
    [
        # Empty input
        ([], []),
        # TODO: more elaborate test cases?
        (
            [
                "N#Cc1ccnc(C#N)n1",
                "N#Cc1cc(C#N)c(F)c(C#N)c1F",
                "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1",
            ],
            [
                "N#Cc1ccnc(C#N)n1",
                "N#Cc1cc(C#N)c(F)c(C#N)c1F",
                "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1",
            ],
        ),
    ],
)
def test_flatness_filter(smiles, expected):
    """Test the FlatnessFilter"""
    smiles = [Chem.MolFromSmiles(smile) for smile in smiles]
    molecule_filter = FlatnessFilter()
    filtered = molecule_filter.apply(smiles)
    assert [Chem.MolToSmiles(mol) for mol in filtered] == expected
