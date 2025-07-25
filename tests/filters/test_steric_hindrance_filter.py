"""Test for StericHindranceFilter class"""
import pytest
from rdkit import Chem

from modules.core.features.preprocessing import canon_smiles
from modules.core.filters.steric_hindrance_filter import StericHindranceFilter


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ([], []),
        (
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
        ),
    ],
)
def test_steric_hindrance_filter(smiles, expected):
    """Test the StericHindranceFilter"""
    smiles = [Chem.MolFromSmiles(smile) for smile in smiles]
    molecule_filter = StericHindranceFilter()
    filtered = molecule_filter.apply(smiles)
    filtered = [Chem.RemoveAllHs(mol) for mol in filtered]
    assert sorted([Chem.MolToSmiles(mol) for mol in filtered]) == sorted([canon_smiles(e) for e in expected])
