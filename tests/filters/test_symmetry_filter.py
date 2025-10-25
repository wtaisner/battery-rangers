"""Test for the SymmetryFilter class"""
import pytest
from rdkit import Chem

from modules.core.filters.symmetry_filter import SymmetryFilter


@pytest.mark.parametrize(
    "smiles, expected",
    [
        (
            [
                "Nc1ccc2c(c1)C(=O)c1ccc(Nc3nc(Nc4ccc5c(c4)C(=O)c4ccc(N)cc4C5=O)nc(Nc4ccc5c(c4)C(=O)c4ccc(N)cc4C5=O)n3)cc1C2=O",  # rings are connected by a single bond
                "Clc1nc(Cl)nc(-c2ccc(-c3cc(-c4ccc(-c5nc(Cl)nc(Cl)n5)cc4)cc(-c4ccc(-c5nc(Cl)nc(Cl)n5)cc4)c3)cc2)n1",
                "Nc1cc2nc3cc4nc5c6nc7cc8nc9cc(N)c(N)cc9nc8cc7nc6c6nc7cc8nc9cc(N)c(N)cc9nc8cc7nc6c5nc4cc3nc2cc1N",  # rings are connected by having a common edge
                "Nc1cc2nc3c4nc5cc(N)c(N)cc5nc4c4nc5cc(N)c(N)cc5nc4c3nc2cc1N",
                "NC1=C(N)C(=O)C2N=c3c(c4c(c5c3=NC3C(=O)C(N)=C(N)C(=O)C3N=5)=NC3C(=O)C(N)=C(N)C(=O)C3N=4)=NC2C1=O",
                # "O=Cc1ccc(N(c2ccc(C=O)cc2)c2ccc(/C=N/c3ccc(-n4c5ccc(/N=C/c6ccc(N(c7ccc(C=O)cc7)c7ccc(C=O)cc7)cc6)cc5c5cc(/N=C/c6ccc(N(c7ccc(C=O)cc7)c7ccc(C=O)cc7)cc6)ccc54)cc3)cc2)cc1",  # not symmetrical
            ],
            [
                "Nc1ccc2c(c1)C(=O)c1ccc(Nc3nc(Nc4ccc5c(c4)C(=O)c4ccc(N)cc4C5=O)nc(Nc4ccc5c(c4)C(=O)c4ccc(N)cc4C5=O)n3)cc1C2=O",
                "Clc1nc(Cl)nc(-c2ccc(-c3cc(-c4ccc(-c5nc(Cl)nc(Cl)n5)cc4)cc(-c4ccc(-c5nc(Cl)nc(Cl)n5)cc4)c3)cc2)n1",
                "Nc1cc2nc3cc4nc5c6nc7cc8nc9cc(N)c(N)cc9nc8cc7nc6c6nc7cc8nc9cc(N)c(N)cc9nc8cc7nc6c5nc4cc3nc2cc1N",
                "Nc1cc2nc3c4nc5cc(N)c(N)cc5nc4c4nc5cc(N)c(N)cc5nc4c3nc2cc1N",
                "NC1=C(N)C(=O)C2N=c3c(c4c(c5c3=NC3C(=O)C(N)=C(N)C(=O)C3N=5)=NC3C(=O)C(N)=C(N)C(=O)C3N=4)=NC2C1=O",
            ],
        ),
        ([], []),
    ],
)
def test_symmetry_filter(smiles, expected):
    """Test the SymmetryFilter"""
    smiles = [Chem.MolFromSmiles(smile) for smile in smiles]
    molecule_filter = SymmetryFilter()
    filtered = molecule_filter.apply(smiles)
    assert [Chem.MolToSmiles(mol) for mol in filtered] == expected


@pytest.mark.parametrize(
    "smiles, expected",
    [
        (
            [
                "Nc1ccc2c(c1)C(=O)c1ccc(Nc3nc(Nc4ccc5c(c4)C(=O)c4ccc(N)cc4C5=O)nc(Nc4ccc5c(c4)C(=O)c4ccc(N)cc4C5=O)n3)cc1C2=O",  # rings are connected by a single bond
                "Nc1cc2nc3cc4nc5c6nc7cc8nc9cc(N)c(N)cc9nc8cc7nc6c6nc7cc8nc9cc(N)c(N)cc9nc8cc7nc6c5nc4cc3nc2cc1N",  # rings are connected by having a common edge
            ],
            [3, 3],
        ),
        ([], []),
    ],
)
def test_symmetry_filter_2(smiles, expected):
    """Test the SymmetryFilter with returning counts"""
    smiles = [Chem.MolFromSmiles(smile) for smile in smiles]
    molecule_filter = SymmetryFilter()
    _, summaries = molecule_filter.apply(smiles, return_branch_counts=True)
    summaries = [s["num_in_group"] for s in summaries]
    assert summaries == expected
