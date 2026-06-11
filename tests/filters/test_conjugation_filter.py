"""Test for the ConjugationFilter class"""

import pytest
from rdkit import Chem

from modules.core.filters.conjugation_filter import ConjugationFilter


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ([], []),
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
        (["CCC", "CCCCC"], []),
        (
            [
                "Cc1ccc(C#N)c(SCC#N)n1",
                "N#CCC#N",
                "N#Cc1c2c(c(C#N)c3ccccc13)CCCC2",
                "N#Cc1ccc2oc(-c3cc4ccc(C#N)cc4[nH]3)cc2c1",
            ],
            [
                "N#Cc1c2c(c(C#N)c3ccccc13)CCCC2",
                "N#Cc1ccc2oc(-c3cc4ccc(C#N)cc4[nH]3)cc2c1",
            ],
        ),
        (
            ["Nc1ccc(Oc2ccc(/N=C/c3ccc(C=O)cc3)cc2)cc1", "COCCl"],
            ["Nc1ccc(Oc2ccc(/N=C/c3ccc(C=O)cc3)cc2)cc1"],
        ),
    ],
)
def test_conjugation_filter(smiles, expected):
    """Test the XYZPatternFilter"""
    smiles = [Chem.MolFromSmiles(smile) for smile in smiles]
    molecule_filter = ConjugationFilter()
    filtered = molecule_filter.apply(smiles)
    assert [Chem.MolToSmiles(mol) for mol in filtered] == expected
