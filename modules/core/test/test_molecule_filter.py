"""Test molecule filters"""
import pytest
from rdkit import Chem

from modules.core.features.filters.conjugation_filter import ConjugationFilter
from modules.core.features.filters.flatness_filter import FlatnessFilter
from modules.core.features.filters.point_group_symmetry_filter import PointGroupSymmetryFilter

# pylint: disable=import-error
from modules.core.features.filters.smarts_filter import SMARTSFilter
from modules.core.features.filters.steric_hindrance_filter import StericHindranceFilter
from modules.core.features.filters.symmetry_filter import SymmetryFilter
from modules.core.features.molecule_filter import MoleculeFilter


@pytest.mark.parametrize(
    "smiles, expected",
    [
        (
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
        ),
        ([], []),
        (["N#Cc1ccnc(C#N)n1", "CCC", "CCCCC"], ["N#Cc1ccnc(C#N)n1"]),
    ],
)
def test_entire_pipeline(smiles, expected):
    """Test the entire pipeline"""
    molecule_filter = MoleculeFilter()
    filtered, _ = molecule_filter.apply(smiles)
    assert filtered == expected


@pytest.mark.parametrize(
    "smiles, expected",
    [
        (
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
        ),
        ([], []),
        (["N#Cc1ccnc(C#N)n1", "CCC", "CCCCC"], ["N#Cc1ccnc(C#N)n1"]),
        (
            [
                "Nc1nc(N)nc(/N=C/N2CCN(/C=N/c3nc(N)nc(N)n3)CC2)n1",
                "Nc1cc2nc3cc4nc5c6nc7cc8nc9cc(N)c(N)cc9nc8cc7nc6c6nc7cc8nc9cc(N)c(N)cc9nc8cc7nc6c5nc4cc3nc2cc1N",
                "Brc1ccc(C(c2ccc(Br)cc2)C(c2ccc(Br)cc2)c2ccc(-c3ccc(-c4ccc5c(c4)Sc4cc(Br)ccc4S5)cc3)cc2)cc1",
            ],  # NH2 + one random without any substructure
            ["Nc1nc(N)nc(/N=C/N2CCN(/C=N/c3nc(N)nc(N)n3)CC2)n1", "Nc1cc2nc3cc4nc5c6nc7cc8nc9cc(N)c(N)cc9nc8cc7nc6c6nc7cc8nc9cc(N)c(N)cc9nc8cc7nc6c5nc4cc3nc2cc1N"],
        ),
    ],
)
def test_smarts_filter(smiles, expected):
    """Test the SMARTSFilter"""
    smiles = [Chem.MolFromSmiles(smile) for smile in smiles]
    molecule_filter = SMARTSFilter()
    filtered = molecule_filter.apply(smiles)
    assert [Chem.MolToSmiles(mol) for mol in filtered] == expected


@pytest.mark.parametrize(
    "smiles, expected",
    [
        (
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
        ),
        ([], []),
        (["N#Cc1ccnc(C#N)n1"], ["N#Cc1ccnc(C#N)n1"]),
    ],
)
def test_flatness_filter(smiles, expected):
    """Test the FlatnessFilter"""
    smiles = [Chem.MolFromSmiles(smile) for smile in smiles]
    molecule_filter = FlatnessFilter()
    filtered = molecule_filter.apply(smiles)
    assert [Chem.MolToSmiles(mol) for mol in filtered] == expected


@pytest.mark.parametrize(
    "smiles, expected",
    [
        (
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
        ),
        ([], []),
        (["N#Cc1ccnc(C#N)n1", "CCCCC"], ["N#Cc1ccnc(C#N)n1"]),
    ],
)
def test_point_group_symmetry_filter(smiles, expected):
    """Test the PointGroupSymmetryFilter"""
    smiles = [Chem.MolFromSmiles(smile) for smile in smiles]
    molecule_filter = PointGroupSymmetryFilter()
    filtered = molecule_filter.apply(smiles)
    assert sorted([Chem.MolToSmiles(mol) for mol in filtered]) == sorted(expected)


@pytest.mark.parametrize(
    "smiles, expected",
    [
        (
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
        ),
        ([], []),
        (["CCC", "CCCCC"], []),
        (
            ["Cc1ccc(C#N)c(SCC#N)n1", "N#CCC#N", "N#Cc1c2c(c(C#N)c3ccccc13)CCCC2", "N#Cc1ccc2oc(-c3cc4ccc(C#N)cc4[nH]3)cc2c1"],
            ["N#Cc1c2c(c(C#N)c3ccccc13)CCCC2", "N#Cc1ccc2oc(-c3cc4ccc(C#N)cc4[nH]3)cc2c1"],
        ),
        (["Nc1ccc(Oc2ccc(/N=C/c3ccc(C=O)cc3)cc2)cc1", "COCCl"], ["Nc1ccc(Oc2ccc(/N=C/c3ccc(C=O)cc3)cc2)cc1"]),
    ],
)
def test_conjugation_filter(smiles, expected):
    """Test the XYZPatternFilter"""
    smiles = [Chem.MolFromSmiles(smile) for smile in smiles]
    molecule_filter = ConjugationFilter()
    filtered = molecule_filter.apply(smiles)
    assert [Chem.MolToSmiles(mol) for mol in filtered] == expected


@pytest.mark.parametrize(
    "smiles, expected",
    [
        (
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
        ),
        ([], []),
        (["N#Cc1ccnc(C#N)n1"], ["N#Cc1ccnc(C#N)n1"]),
    ],
)
def test_steric_hindrance_filter(smiles, expected):
    """Test the StericHindranceFilter"""
    smiles = [Chem.MolFromSmiles(smile) for smile in smiles]
    molecule_filter = StericHindranceFilter()
    filtered = molecule_filter.apply(smiles)
    assert [Chem.MolToSmiles(mol) for mol in filtered] == expected


@pytest.mark.parametrize(
    "smiles, expected",
    [
        (
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
        ),
        ([], []),
        (["N#Cc1ccnc(C#N)n1"], ["N#Cc1ccnc(C#N)n1"]),
    ],
)
def test_symmetry_filter(smiles, expected):
    """Test the SymmetryFilter"""
    smiles = [Chem.MolFromSmiles(smile) for smile in smiles]
    molecule_filter = SymmetryFilter()
    filtered = molecule_filter.apply(smiles)
    assert [Chem.MolToSmiles(mol) for mol in filtered] == expected
