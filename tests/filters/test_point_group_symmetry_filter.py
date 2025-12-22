# """Tests for the PointGroupSymmetryFilter class."""
# import pytest
# from rdkit import Chem
#
# from modules.core.filters.point_group_symmetry_filter import PointGroupSymmetryFilter
#
#
# @pytest.mark.parametrize(
#     "smiles, expected",
#     [
#         ([], []),
#         (
#             [
#                 # "N#Cc1ccnc(C#N)n1",
#                 "N#Cc1cc(C#N)c(F)c(C#N)c1F",
#                 # "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1",
#                 "N#Cc1ccc(/C=C/C(/C=C/c2ccc(C#N)cc2)/C=C/c2ccc(-c3nc(-c4ccc(/C=C/C(/C=C/c5ccc(C#N)cc5)/C=C/c5ccc(C#N)cc5)cc4)nc(-c4ccc(/C=C/C(/C=C/c5ccc(C#N)cc5)/C=C/c5ccc(C#N)cc5)cc4)n3)cc2)cc1",
#                 "N#Cc1ccc(-n2c(=O)c3cc4c(=O)n(-c5ccc(-c6nc(-c7ccc(-n8c(=O)c9cc%10c(=O)n(-c%11ccc(C#N)cc%11)c(=O)c%10cc9c8=O)cc7)nc(-c7ccc(-n8c(=O)c9cc%10c(=O)n(-c%11ccc(C#N)cc%11)c(=O)c%10cc9c8=O)cc7)n6)cc5)c(=O)c4cc3c2=O)cc1",
#             ],
#             [
#                 "N#Cc1ccc(/C=C/C(/C=C/c2ccc(C#N)cc2)/C=C/c2ccc(-c3nc(-c4ccc(/C=C/C(/C=C/c5ccc(C#N)cc5)/C=C/c5ccc(C#N)cc5)cc4)nc(-c4ccc(/C=C/C(/C=C/c5ccc(C#N)cc5)/C=C/c5ccc(C#N)cc5)cc4)n3)cc2)cc1",
#                 "N#Cc1ccc(-n2c(=O)c3cc4c(=O)n(-c5ccc(-c6nc(-c7ccc(-n8c(=O)c9cc%10c(=O)n(-c%11ccc(C#N)cc%11)c(=O)c%10cc9c8=O)cc7)nc(-c7ccc(-n8c(=O)c9cc%10c(=O)n(-c%11ccc(C#N)cc%11)c(=O)c%10cc9c8=O)cc7)n6)cc5)c(=O)c4cc3c2=O)cc1",
#             ],
#         ),
#     ],
# )
# def test_point_group_symmetry_filter(smiles, expected):
#     """Test the PointGroupSymmetryFilter"""
#     smiles = [Chem.MolFromSmiles(smile) for smile in smiles]
#     molecule_filter = PointGroupSymmetryFilter()
#     filtered = molecule_filter.apply(smiles)
#     filtered = [Chem.RemoveAllHs(mol) for mol in filtered]
#     assert sorted([Chem.MolToSmiles(mol) for mol in filtered]) == sorted(expected)
