"""Filter that leaves molecules with a specific point group symmetry."""
from copy import deepcopy

from tqdm import tqdm

from modules.core.features.filters.generic_filter import GenericMoleculeFilter
from modules.core.features.symmetries import analyse_symmetry_point_group, translate_point_group_to_symmetry_description
from modules.core.features.utils import get_pymatgen_molecule_from_smiles


class PointGroupSymmetryFilter(GenericMoleculeFilter):
    """Filter that leaves molecules with a specific point group symmetry."""

    def apply(self, smiles: list[str], **kwargs) -> list[str]:
        """
        Apply the filter to a list of SMILES strings.

        Args:
            smiles (list[str]): The list of SMILES strings to filter.
        Returns:
            list[str]: The list of SMILES strings that passed the filter.
        """
        tmp_smiles = deepcopy(smiles)
        pymatgen_molecules = []
        bad_smiles = []
        for sml in tqdm(tmp_smiles, total=len(tmp_smiles), desc="Getting pymatgen molecules"):
            try:
                pymatgen_molecules.append(get_pymatgen_molecule_from_smiles(sml))
            except:  # pylint: disable=bare-except
                bad_smiles.append(sml)
                continue
        print("pymatgen modules:", len(pymatgen_molecules), "bad smiles:", len(bad_smiles))
        if len(pymatgen_molecules) != len(tmp_smiles):
            tmp_smiles = [x for x in tmp_smiles if x not in bad_smiles]

        point_group_symmetrical_smiles = self._point_group_symmetry(pymatgen_molecules, tmp_smiles)
        both_symmetrical_smiles = list(set(tmp_smiles).intersection(set(point_group_symmetrical_smiles)))
        return both_symmetrical_smiles

    @staticmethod
    def _point_group_symmetry(pymatgen_molecules, smiles_lst, translation_table_path: str = "/home/witoldt/repositories/battery-rangers/data/symmetries/symmetry_translation.csv"):
        pointgroup_symmetrical = []
        for i, mol in enumerate(pymatgen_molecules):
            _, _, pointgroup, _ = analyse_symmetry_point_group(mol, translation_table_path=translation_table_path)
            symm = translate_point_group_to_symmetry_description(pointgroup, translation_table_path=translation_table_path)
            if symm.startswith("no"):
                continue
            pointgroup_symmetrical.append(smiles_lst[i])
        return pointgroup_symmetrical
