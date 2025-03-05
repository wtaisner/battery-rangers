"""Test molecule filters"""
import pytest
from rdkit import Chem

# pylint: disable=import-error
from modules.core.features.filters.c_n_triple_bonds_filter import CNTripleBondsFilter
from modules.core.features.filters.conjugation_filter import ConjugationFilter
from modules.core.features.filters.flatness_filter import FlatnessFilter
from modules.core.features.filters.point_group_symmetry_filter import PointGroupSymmetryFilter
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
    ],
)
def test_c_n_triple_bonds_filter(smiles, expected):
    """Test the CNTripleBondsFilter"""
    smiles = [Chem.MolFromSmiles(smile) for smile in smiles]
    molecule_filter = CNTripleBondsFilter()
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
