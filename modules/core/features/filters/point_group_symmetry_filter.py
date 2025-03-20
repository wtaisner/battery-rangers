"""Filter that leaves molecules with a specific point group symmetry."""
import logging
from copy import deepcopy

from pymatgen.core import Molecule
from rdkit import Chem
from rdkit.Chem import Mol

from modules.core.features.filters.generic_filter import GenericMoleculeFilter
from modules.core.features.symmetries import analyse_symmetry_point_group, translate_point_group_to_symmetry_description
from modules.core.features.utils import get_pymatgen_molecule_from_smiles

logger = logging.getLogger(__name__)  # __name__ ensures the logger is specific to this module


class PointGroupSymmetryFilter(GenericMoleculeFilter):
    """Filter that leaves molecules with a specific point group symmetry."""

    def apply(self, molecules: list[Mol], **kwargs) -> list[Mol]:
        """
        Apply the filter to a list of RDKit Mol objects.

        Args:
            molecules (list[Mol]): The list of RDKit Mol objects to filter.
        Returns:
            list[Mol]: The list of RDKit Mol objects that passed the filter.
        """
        molecules_copy = deepcopy(molecules)
        pymatgen_molecules = []
        bad_smiles = []
        for mol in molecules_copy:
            try:
                pymatgen_molecules.append(get_pymatgen_molecule_from_smiles(Chem.MolToSmiles(mol)))
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Error converting SMILES to pymatgen molecule: {mol}, Error: {str(e)}")
                bad_smiles.append(mol)
                continue

        if len(pymatgen_molecules) != len(molecules_copy):
            molecules_copy = [x for x in molecules_copy if x not in bad_smiles]

        point_group_symmetrical_smiles = self._point_group_symmetry(pymatgen_molecules, molecules_copy)
        both_symmetrical_smiles = list(set(molecules_copy).intersection(set(point_group_symmetrical_smiles)))
        return both_symmetrical_smiles

    @staticmethod
    def _point_group_symmetry(pymatgen_molecules: list[Molecule], smiles_lst: list[str], translation_table_path: str = "data/symmetries/symmetry_translation.csv") -> list[str]:
        """
        Filters out SMILES strings corresponding to molecules that do not exhibit
        a point group symmetry, based on pymatgen analysis.

        Args:
            pymatgen_molecules (List[Molecule]): List of pymatgen Molecule objects.
            smiles_lst (List[str]): List of SMILES strings corresponding to the molecules.

        Returns:
            List[str]: List of SMILES strings for molecules with point group symmetry.
        """
        pointgroup_symmetrical = []

        for i, mol in enumerate(pymatgen_molecules):
            _, _, pointgroup, _ = analyse_symmetry_point_group(mol, translation_table_path=translation_table_path)
            symm = translate_point_group_to_symmetry_description(pointgroup, translation_table_path=translation_table_path)

            if not symm.startswith("no"):  # Skip molecules with no symmetry
                pointgroup_symmetrical.append(smiles_lst[i])

        return pointgroup_symmetrical
