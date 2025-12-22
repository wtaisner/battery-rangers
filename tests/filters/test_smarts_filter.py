"""Test for SMARTSFilter class"""
import pytest
from rdkit import Chem

from modules.core.filters.smarts_filter import SMARTSFilter


@pytest.mark.parametrize(
    "smiles, expected",
    [
        # Empty input
        ([], []),
        # C#N
        # 2, 3 ,4 ,8-17, 25, 44, 67-72,
        (
            [
                "N#Cc1ccnc(C#N)n1",
                "N#Cc1cc(C#N)c(F)c(C#N)c1F",
                "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1",
            ],
            ["N#Cc1ccnc(C#N)n1", "N#Cc1cc(C#N)c(F)c(C#N)c1F", "N#Cc1ccc(-c2cc(=O)nc(-c3ccc(C#N)cc3)[nH]2)cc1"],
        ),
        # Cl/Br
        # 5, 6, 7, 39, 40, 43, 60
        (
            [
                "ClC1=NC(C2=CC=C(C3=CC(C4=CC=C(C5=NC(Cl)=NC(Cl)=N5)C=C4)=CC(C6=CC=C(C7=NC(Cl)=NC(Cl)=N7)C=C6)=C3)C=C2)=NC(Cl)=N1",  # 5
                "O=S8(=O)c1ccc(Br)cc1S(=O)(=O)c7cc(c6ccc(c5ccc(C(c2ccc(Br)cc2)C(c3ccc(Br)cc3)c4ccc(Br)cc4)cc5)cc6)ccc78",  # 60
                "BrC1=CC=C(C(C(C2=CC=C(C=C2)C3=CC=C(C=C3)C4=CC5=C(C=C4Br)SC6=C(C=C(C(Br)=C6)C7=CC=C(C8=CC=C(C(C(C9=CC=C(Br)C=C9)C%10=CC=C(Br)C=C%10)C%11=CC=C(Br)C=C%11)C=C8)C=C7)S5)C%12=CC=C(Br)C=C%12)C%13=CC=C(Br)C=C%13)C=C1",  # 39
            ],
            [
                # "ClC1=NC(C2=CC=C(C3=CC(C4=CC=C(C5=NC(Cl)=NC(Cl)=N5)C=C4)=CC(C6=CC=C(C7=NC(Cl)=NC(Cl)=N7)C=C6)=C3)C=C2)=NC(Cl)=N1",
                # "O=S8(=O)c1ccc(Br)cc1S(=O)(=O)c7cc(c6ccc(c5ccc(C(c2ccc(Br)cc2)C(c3ccc(Br)cc3)c4ccc(Br)cc4)cc5)cc6)ccc78",
                # "BrC1=CC=C(C(C(C2=CC=C(C=C2)C3=CC=C(C=C3)C4=CC5=C(C=C4Br)SC6=C(C=C(C(Br)=C6)C7=CC=C(C8=CC=C(C(C(C9=CC=C(Br)C=C9)C%10=CC=C(Br)C=C%10)C%11=CC=C(Br)C=C%11)C=C8)C=C7)S5)C%12=CC=C(Br)C=C%12)C%13=CC=C(Br)C=C%13)C=C1",
            ],
        ),
        # NH2 + Cl/Br
        # 6, 27, 91, 92, 93, 94, 95
        (
            [
                "ClC1=NC(Cl)=NC(NC2=CC3=C(C=C2)C(C4=C(C3=O)C=CC(NC5=NC(Cl)=NC(Cl)=N5)=C4)=O)=N1",  # 6
                "Nc1nc(N)nc(Nc2sc(Br)cc2)n1",  # 27
                "NC1=CC2=C(C=C1)C(C3=CC(NC4=CC=C(N(C5=CC=C(Br)C=C5)C6=CC=C(Br)C=C6)C=C4)=CC=C3C2=O)=O",  # 91
                "NC1=CC2=C(C=C1)C(C3=CC(NC4=CC=C(C(C5=C6C=C(Br)C=C5)(C7=CC=C(Br)C=C76)C8=CC=C(Br)C=C89)C9=C4)=CC=C3C2=O)=O",  # 95
            ],
            [
                # "ClC1=NC(Cl)=NC(NC2=CC3=C(C=C2)C(C4=C(C3=O)C=CC(NC5=NC(Cl)=NC(Cl)=N5)=C4)=O)=N1",
                # "Nc1nc(N)nc(Nc2sc(Br)cc2)n1",
                # "NC1=CC2=C(C=C1)C(C3=CC(NC4=CC=C(N(C5=CC=C(Br)C=C5)C6=CC=C(Br)C=C6)C=C4)=CC=C3C2=O)=O",
                # "NC1=CC2=C(C=C1)C(C3=CC(NC4=CC=C(C(C5=C6C=C(Br)C=C5)(C7=CC=C(Br)C=C76)C8=CC=C(Br)C=C89)C9=C4)=CC=C3C2=O)=O",
            ],
        ),
        # NH2 + CHO
        # 20-24, 29, 33, 34, 45-52, 54-59,
        # 75, 78, 79, 81, 82, 85-90, 98,
        # 101-105
        (
            [
                "NC(N=C1)=CC=C1C2=NC(C3=CC=C(N=CC4=C(O)C(C=O)=C(O)C(C=O)=C4)N=C3)=NC(C5=CN=C(N)C=C5)=N2",  # 20
                "NC(N=C1)=CC=C1C2=NC(C3=CC=C(N=CC4=C(O)C(C=O)=C(O)C(C=O)=C4O)N=C3)=NC(C5=CN=C(N)C=C5)=N2",  # 21
                "Nc1nc(/N=C/c2ccc(C=O)cc2)nc(N)n1",  # 45
                "O=C1C(C2=C(C3=CC(/C=N/C4=CC=C(C5=C(C=CC6=C(C7=CC=C(N)C=C7)C=C(C8=CC=C(N)C=C8)C(C=C9)=C6%10)C%10=C9C(C%11=CC=C(N)C=C%11)=C5)C=C4)=CS3)N1CCCCCCCC)=C(C%12=CC(C=O)=CS%12)N(CCCCCCCC)C2=O",  # 101
            ],
            [
                # "NC(N=C1)=CC=C1C2=NC(C3=CC=C(N=CC4=C(O)C(C=O)=C(O)C(C=O)=C4)N=C3)=NC(C5=CN=C(N)C=C5)=N2",
                # "NC(N=C1)=CC=C1C2=NC(C3=CC=C(N=CC4=C(O)C(C=O)=C(O)C(C=O)=C4O)N=C3)=NC(C5=CN=C(N)C=C5)=N2",
                # TODO: following 2 will work with cardinality = 1 for rule #5
                # "Nc1nc(/N=C/c2ccc(C=O)cc2)nc(N)n1",
                # "O=C1C(C2=C(C3=CC(/C=N/C4=CC=C(C5=C(C=CC6=C(C7=CC=C(N)C=C7)C=C(C8=CC=C(N)C=C8)C(C=C9)=C6%10)C%10=C9C(C%11=CC=C(N)C=C%11)=C5)C=C4)=CS3)N1CCCCCCCC)=C(C%12=CC(C=O)=CS%12)N(CCCCCCCC)C2=O",
            ],
        ),
        # NH2 + rings with one N atom
        # 18, 19, 26, 77
        (
            [
                "NC1=CC=C(C2=NC(C3=CC=C(C=C3)N(C(C4=CC=C5C6=C4C7=CC=C6C(N(C8=CC=C(C9=NC(C%10=CC=C(N)C=C%10)=NC(C%11=CC=C(N)C=C%11)=N9)C=C8)C5=O)=O)=O)C7=O)=NC(C%12=CC=C(C=C%12)N)=N2)C=C1",  # 19
                "NC1=CC2=C(C=C1)N(CCCCCC)C3=C2C(N(CCCCCC)C4=C5C=C(N)C=C4)=C5C6=C3C7=C(C=CC(N8C(C(C=C(C(N(C(C=C9)=CC%10=C9C%11=C(N%10CCCCCC)C%12=C(N(CCCCCC)C%13=C%12C=C(N)C=C%13)C%14=C%11N(CCCCCC)C%15=C%14C=C(N)C=C%15)C%16=O)=O)C%16=C%17)=C%17C8=O)=O)=C7)N6CCCCCC",
                # 77
            ],
            [
                # "NC1=CC=C(C2=NC(C3=CC=C(C=C3)N(C(C4=CC=C5C6=C4C7=CC=C6C(N(C8=CC=C(C9=NC(C%10=CC=C(N)C=C%10)=NC(C%11=CC=C(N)C=C%11)=N9)C=C8)C5=O)=O)=O)C7=O)=NC(C%12=CC=C(C=C%12)N)=N2)C=C1",
                # "NC1=CC2=C(C=C1)N(CCCCCC)C3=C2C(N(CCCCCC)C4=C5C=C(N)C=C4)=C5C6=C3C7=C(C=CC(N8C(C(C=C(C(N(C(C=C9)=CC%10=C9C%11=C(N%10CCCCCC)C%12=C(N(CCCCCC)C%13=C%12C=C(N)C=C%13)C%14=C%11N(CCCCCC)C%15=C%14C=C(N)C=C%15)C%16=O)=O)C%16=C%17)=C%17C8=O)=O)=C7)N6CCCCCC",
            ],
        ),
        # TODO: add cases with O=C, [HO] and rings
        (
            [
                "OC1=CC2=C(C=C(O)C(O)=C3)C3=C(C=C(OB(C4=CC=C(B(O)O)C=C4)O5)C5=C6)C6=C2C=C1O",
                "O=C1C(C(c2nc(C(c3nc(C(C(C(C4=O)=O)=O)=O)c4nc3C5=O)=O)c5nc2C1=O)=O)=O",
                "O=C1C(C(C(C2=C1N=C(C=C3C(N=C(C(C(C(C4=O)=O)=O)=O)C4=N3)=C5)C5=N2)=O)=O)=O",
            ],
            [
                # "OC1=CC2=C(C=C(O)C(O)=C3)C3=C(C=C(OB(C4=CC=C(B(O)O)C=C4)O5)C5=C6)C6=C2C=C1O",
                # "O=C1C(C(c2nc(C(c3nc(C(C(C(C4=O)=O)=O)=O)c4nc3C5=O)=O)c5nc2C1=O)=O)=O",
                # "O=C1C(C(C(C2=C1N=C(C=C3C(N=C(C(C(C(C4=O)=O)=O)=O)C4=N3)=C5)C5=N2)=O)=O)=O",
            ],
        ),
        # Not satisfying any rule TODO: add test cases
        (
            [
                "CCC",
                "O=C(/C(C(/C(C/1=O)=C/Nc2ccc(N)cc2)=O)=C\C)C1=C\C",  # pylint: disable=anomalous-backslash-in-string
            ],
            [],
        ),
    ],
)
def test_smarts_filter(smiles, expected):
    """Test the SMARTSFilter"""
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    molecule_filter = SMARTSFilter()
    filtered = molecule_filter.apply(mols)
    assert [Chem.MolToSmiles(mol) for mol in filtered] == [Chem.MolToSmiles(Chem.MolFromSmiles(exp)) for exp in expected], f"Expected {expected} but got {[Chem.MolToSmiles(mol) for mol in filtered]}"
