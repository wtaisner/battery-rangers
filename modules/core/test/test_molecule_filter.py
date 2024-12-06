"""Test molecule filters"""
import pytest

# pylint: disable=import-error
from modules.core.features.filters.c_n_triple_bonds_filter import CNTripleBondsFilter
from modules.core.features.filters.flatness_filter import FlatnessFilter
from modules.core.features.filters.point_group_symmetry_filter import PointGroupSymmetryFilter
from modules.core.features.filters.single_c_c_bonds_outside_rings_filter import SingleCCBondsOutsideRingsFilter
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
    assert molecule_filter.apply(smiles) == expected


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
    molecule_filter = CNTripleBondsFilter()
    assert molecule_filter.apply(smiles) == expected


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
    molecule_filter = FlatnessFilter()
    assert molecule_filter.apply(smiles) == expected


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
    molecule_filter = PointGroupSymmetryFilter()
    assert sorted(molecule_filter.apply(smiles)) == sorted(expected)


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
def test_single_c_c_bonds_outside_rings_filter(smiles, expected):
    """Test the SingleCCBondsOutsideRingsFilter"""
    molecule_filter = SingleCCBondsOutsideRingsFilter()
    assert molecule_filter.apply(smiles) == expected


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
    molecule_filter = StericHindranceFilter()
    assert molecule_filter.apply(smiles) == expected


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
    molecule_filter = SymmetryFilter()
    assert molecule_filter.apply(smiles) == expected
