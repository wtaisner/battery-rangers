"""Filter that leaves molecules with a specific point group symmetry."""
import logging
from copy import deepcopy

from pymatgen.core import Molecule
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdDistGeom import EmbedMolecule
from tqdm import tqdm

from modules.core.features.filters.generic_filter import GenericMoleculeFilter
from modules.core.features.symmetries import analyse_symmetry_point_group, translate_point_group_to_symmetry_description

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
        tmp_smiles = deepcopy(molecules)
        pymatgen_molecules = []
        bad_smiles = []
        for sml in tqdm(tmp_smiles, total=len(tmp_smiles), desc="Getting pymatgen molecules"):
            try:
                pymatgen_molecules.append(self._rdkit_mol_to_pymatgen_molecule(sml))
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Error converting SMILES to pymatgen molecule: {sml}, Error: {str(e)}")
                bad_smiles.append(sml)
                continue

        logger.debug(f"Pymatgen modules created: {len(pymatgen_molecules)} | Smiles that could not be converted: {len(bad_smiles)}")
        if len(pymatgen_molecules) != len(tmp_smiles):
            tmp_smiles = [x for x in tmp_smiles if x not in bad_smiles]

        point_group_symmetrical_smiles = self._point_group_symmetry(pymatgen_molecules, tmp_smiles)
        both_symmetrical_smiles = list(set(tmp_smiles).intersection(set(point_group_symmetrical_smiles)))
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

    @staticmethod
    def _rdkit_mol_to_pymatgen_molecule(molecule: Mol) -> Molecule:
        """
        Convert a RDKit Mol object to a pymatgen Molecule object.

        Args:
            molecule (Mol): The RDKit Mol object

        Returns:
            Molecule: A pymatgen Molecule object corresponding to the RDKit Mol object

        Raises:
            ValueError: If the RDKit Mol object is invalid or the molecule embedding fails.
        """
        try:
            rdkit_mol = Chem.AddHs(molecule)
            EmbedMolecule(rdkit_mol, randomSeed=42)
            conformer = rdkit_mol.GetConformer()
            coordinates = conformer.GetPositions()
            symbols = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]

            return Molecule(symbols, coordinates)

        except Exception as e:
            raise ValueError(f"Error converting RDKit Mol object to pymatgen molecule: {molecule}, Error: {str(e)}") from e
