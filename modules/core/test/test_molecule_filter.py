"""Test molecule filters"""
import unittest

import pytest

from modules.core.features.molecule_filter import MoleculeFilter


class MyTestCase(unittest.TestCase):
    """
    t
    """

    def __init__(self):
        super().__init__()
        self.molecule_filter = MoleculeFilter()

    # TODO: fill in the smiles and expected values
    @pytest.mark.parametrize(
        "smiles, expected",
        [
            ([], []),
            ([], []),
        ],
    )
    def test_entire_pipeline(self, smiles: list[str], expected: list[str]):
        """

        :param smiles:
        :param expected:
        :return:
        """
        self.assertEqual(self.molecule_filter.apply(smiles), expected)

    def test_c_n_triple_bonds_filter(self):
        """

        :return:
        """

    def test_flatness_filter(self):
        """

        :return:
        """

    def test_point_group_symmetry_filter(self):
        """

        :return:
        """

    def test_single_c_c_bonds_outside_rings_filter(self):
        """

        :return:
        """

    def test_steric_hindrance_filter(self):
        """

        :return:
        """

    def test_symmetry_filter(self):
        """

        :return:
        """


if __name__ == "__main__":
    unittest.main()
