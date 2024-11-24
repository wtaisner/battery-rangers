"""Filter that leaves molecules with a specific point group symmetry."""
import logging
from copy import deepcopy

from pymatgen.core import Molecule
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
from tqdm import tqdm

from modules.core.features.filters.generic_filter import GenericMoleculeFilter
from modules.core.features.symmetries import analyse_symmetry_point_group, translate_point_group_to_symmetry_description

logger = logging.getLogger(__name__)  # __name__ ensures the logger is specific to this module


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
                pymatgen_molecules.append(self._smiles_to_pymatgen_molecule(sml))
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
    def _point_group_symmetry(pymatgen_molecules: list[Molecule], smiles_lst: list[str]) -> list[str]:
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
            _, _, pointgroup, _ = analyse_symmetry_point_group(mol)
            symm = translate_point_group_to_symmetry_description(pointgroup)

            if not symm.startswith("no"):  # Skip molecules with no symmetry
                pointgroup_symmetrical.append(smiles_lst[i])

        return pointgroup_symmetrical

    @staticmethod
    def _smiles_to_pymatgen_molecule(smiles: str) -> Molecule:
        """
        Convert a SMILES string to a pymatgen Molecule object by first converting it
        to an RDKit molecule and then embedding it in 3D space.

        Args:
            smiles (str): The SMILES string of the molecule.

        Returns:
            Molecule: A pymatgen Molecule object corresponding to the SMILES string.

        Raises:
            ValueError: If the SMILES string is invalid or the molecule embedding fails.
        """
        try:
            cannonic_smiles = Chem.CanonSmiles(smiles)
            rdkit_mol = Chem.MolFromSmiles(cannonic_smiles)
            rdkit_mol = Chem.AddHs(rdkit_mol)
            EmbedMolecule(rdkit_mol, randomSeed=42)
            conformer = rdkit_mol.GetConformer()
            coordinates = conformer.GetPositions()
            symbols = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]

            return Molecule(symbols, coordinates)

        except Exception as e:
            raise ValueError(f"Error converting SMILES to pymatgen molecule: {smiles}, Error: {str(e)}") from e
