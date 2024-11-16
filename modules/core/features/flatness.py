"""Flatness feature calculation."""
import logging

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Mol, rdDepictor, rdDistGeom
from sklearn.linear_model import LinearRegression


def embed_molecule(mol: Mol, sample_size: int = 20, random_seed: int = 321) -> Chem.Mol:
    """
    Embed a molecule using RDKit.

    Args:
        mol: A molecule.
        sample_size:
        random_seed:

    Returns:
        The embedded molecule.
    """
    mol = Chem.AddHs(mol)

    rdDepictor.Compute2DCoords(mol)
    rdDistGeom.EmbedMolecule(mol)
    rdDistGeom.EmbedMultipleConfs(mol, sample_size, randomSeed=random_seed)

    return mol


def get_flatness_smiles(smiles: str, plot_visualization: bool = False, **kwargs) -> float:
    """
    Calculate the flatness of a molecule from a SMILES string.

    Args:
        smiles: A SMILES string.
        plot_visualization: Whether to plot the molecule and the fitted plane.
        **kwargs: Additional arguments for the get_flatness_mol function.

    Returns:
        The flatness of the molecule.
    """
    can_smi = Chem.CanonSmiles(smiles)
    mol = Chem.MolFromSmiles(can_smi)
    return get_flatness_mol(mol, plot_visualization, **kwargs)


def get_flatness_mol(mol: Chem.Mol, plot_visualization: bool = False, **kwargs) -> float:
    """
    Calculate the flatness of a molecule.

    Fit a plane to the molecule's 3D coordinates and calculate the RMSD of the
    molecule's atoms from the plane.

    Args:
        mol: A molecule.
        plot_visualization: Whether to plot the molecule and the fitted plane.
    Returns:
        The flatness of the molecule.
    """
    # TODO: this probably can be simplified to kwargs.get("sample_size", 20), or sth similar
    if "sample_size" in kwargs:
        embedded_mol = embed_molecule(mol, sample_size=kwargs["sample_size"])
    else:
        embedded_mol = embed_molecule(mol)

    logging.debug(f"Number of conformers: {embedded_mol.GetNumConformers()}")
    logging.debug(f"Is 3D: {embedded_mol.GetConformer().Is3D()}")
    logging.debug(f"Number of atoms: {embedded_mol.GetNumAtoms()}")

    conformers = embedded_mol.GetNumConformers()
    rmsds = np.zeros(conformers)
    for i in range(conformers):
        conf = embedded_mol.GetConformer(i)
        coords = conf.GetPositions()

        # Fit a plane to the coordinates
        model = LinearRegression()
        model.fit(coords[:, :2], coords[:, 2])

        # Calculate the RMSD of the atoms from the plane
        z_pred = model.predict(coords[:, :2])
        rmsds[i] = np.sqrt(np.mean((coords[:, 2] - z_pred) ** 2))

        if plot_visualization:
            __visualize_the_plane(coords, model)

    logging.info(f"RMSDs: {np.mean(rmsds)} +/- {np.std(rmsds)}")

    return np.mean(rmsds)


def __visualize_the_plane(coords: np.ndarray, model: LinearRegression) -> None:
    """
    Visualize the plane fitted to the molecule's 3D coordinates.

    Args:
        coords: The molecule's 3D coordinates.
        model: The fitted plane.
    Returns:
        None
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])

    x = np.linspace(min(coords[:, 0]), max(coords[:, 0]), 10)
    y = np.linspace(min(coords[:, 1]), max(coords[:, 1]), 10)

    x, y = np.meshgrid(x, y)

    z = model.intercept_ + model.coef_[0] * x + model.coef_[1] * y

    ax.plot_surface(x, y, z, alpha=0.2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set ticks to integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))

    # bbox_inches='tight' removes the white space around the image
    plt.tight_layout()
    plt.savefig("flatness.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    # smiles = 'Nc1ccc(-c2nc(-c3ccc(N)cc3)nc(-c3ccc(N4C(=O)c5ccc6c7c(ccc(c57)C4=O)C(=O)OC6=O)cc3)n2)cc1'
    SMILES = "N#Cc1c(F)c(F)c(C#N)c(F)c1F"

    f = get_flatness_smiles(SMILES, plot_visualization=True, sample_size=1)
    logging.info(f"Flatness: {f}")

    mol = Chem.MolFromSmiles(SMILES)
    Draw.MolToFile(mol, "mol.png")
