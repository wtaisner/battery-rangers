"""Filter that leaves molecules with a specific point group symmetry."""

import logging
from typing import Any

from pymatgen.core import Molecule
from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.features.symmetries import analyse_symmetry_point_group
from modules.core.features.utils import get_pymatgen_molecule_from_smiles
from modules.core.filters.generic_filter import GenericMoleculeFilter

logger = logging.getLogger(__name__)  # __name__ ensures the logger is specific to this module


class PointGroupSymmetryFilter(GenericMoleculeFilter):
    """Filter that leaves molecules with a specific point group symmetry.

    Args:
        translation_table_path (str): Path to the translation table for symmetry analysis. Defaults to "data/symmetries/symmetry_translation.csv".
        allowed_symmetries (set[str], optional): Set of allowed symmetries. If not set, defaults to predefined symmetries.
    """

    def __init__(
        self,
        translation_table_path: str = "data/symmetries/symmetry_translation.csv",
        allowed_symmetries: set[str] | None = None,
    ):
        super().__init__()
        self.translation_table_path = translation_table_path

        if allowed_symmetries is None:
            self.allowed_symmetries = {
                "Cs",
                "C2h",
                "D2h",
                "C3",
                "C3h",
                "C3v",
                "D3h",
                "D3d",
                "C4v",
                "D4h",
                "D4d",
                "D6h",
            }
        else:
            self.allowed_symmetries = allowed_symmetries

    # pylint: disable=arguments-differ
    def apply(self, molecules: list[Mol], return_point_group_symmetry: bool = False) -> tuple[Any, list[Mol]] | list[Mol]:
        """
        Apply the filter to a list of RDKit Mol objects.

        Args:
            molecules (list[Mol]): The list of RDKit Mol objects to filter.
            return_point_group_symmetry (bool): If True, return the point group symmetry of the molecules.
        Returns:
            if return_point_group_symmetry is True:
                list[tuple[str, Mol]]: A list of tuples containing the point group symmetry and the corresponding RDKit molecule.
            else:
                list[Mol]: The list of RDKit Mol objects that passed the filter.

        """
        smiles_dict = {}
        for mol in molecules:
            smiles_dict[Chem.MolToSmiles(mol)] = mol

        pymatgen_molecules = []
        for sml, mol in smiles_dict.items():
            try:
                pymatgen_molecules.append(get_pymatgen_molecule_from_smiles(sml))
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Error converting SMILES to pymatgen molecule: {sml}, Error: {str(e)}")
                smiles_dict[sml] = None
                continue

        # Filter out None values from smiles_dict
        smiles_dict = {k: v for k, v in smiles_dict.items() if v is not None}

        symmetry_groups, point_group_symmetrical_smiles = self._point_group_symmetry(
            pymatgen_molecules,
            list(smiles_dict.keys()),
            return_point_group_symmetry=True,
        )
        point_group_symmetrical_molecules = [v for k, v in smiles_dict.items() if k in point_group_symmetrical_smiles]

        if return_point_group_symmetry:
            return symmetry_groups, point_group_symmetrical_molecules

        return point_group_symmetrical_molecules

    def _point_group_symmetry(
        self,
        pymatgen_molecules: list[Molecule],
        smiles_lst: list[str],
        return_point_group_symmetry: bool = False,
    ) -> tuple[list[str], list[str]] | list[str]:
        """
        Filters out SMILES strings corresponding to molecules that do not exhibit
        a point group symmetry, based on pymatgen analysis.

        Args:
            pymatgen_molecules (List[Molecule]): List of pymatgen Molecule objects.
            smiles_lst (List[str]): List of SMILES strings corresponding to the molecules.
            return_point_group_symmetry (bool): If True, return the point group symmetry of the molecules.

        Returns:
            if return_point_group_symmetry is True:
                tuple[list[str], list[str]]: List of tuples containing the point group symmetry and the corresponding SMILES string.
            else:
                list[str]: List of SMILES strings for molecules with point group symmetry.
        """
        pointgroup_symmetrical = []
        symmetry_groups = []

        for i, mol in enumerate(pymatgen_molecules):
            try:
                _, _, pointgroup, _ = analyse_symmetry_point_group(mol, translation_table_path=self.translation_table_path)
            except AttributeError as e:  # pylint: disable=bare-except
                logger.error(f"Error analyzing symmetry point group: {e}")
                continue

            if pointgroup in self.allowed_symmetries:
                pointgroup_symmetrical.append(smiles_lst[i])
                symmetry_groups.append(pointgroup)
        if return_point_group_symmetry:
            return symmetry_groups, pointgroup_symmetrical
        return pointgroup_symmetrical


if __name__ == "__main__":
    # Example usage
    pgsf = PointGroupSymmetryFilter()
    molecules = [
        Chem.MolFromSmiles(
            "N#Cc1ccc(/C=C/C(/C=C/c2ccc(C#N)cc2)/C=C/c2ccc(-c3nc(-c4ccc(/C=C/C(/C=C/c5ccc(C#N)cc5)/C=C/c5ccc(C#N)cc5)cc4)nc(-c4ccc(/C=C/C(/C=C/c5ccc(C#N)cc5)/C=C/c5ccc(C#N)cc5)cc4)n3)cc2)cc1"
        )
    ]
    filtered_molecules = pgsf.apply(molecules)
    print(filtered_molecules)
