import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    from PIL import Image  # For type hinting and for the user to show/save the image
    from rdkit import Chem
    from rdkit.Chem import Draw

    def plot_molecule_with_smarts_highlight(smiles_string: str, smarts_pattern: str, size: tuple[int, int] = (400, 300)) -> Image.Image | None:
        """
        Generates a PIL Image of a molecule with SMARTS pattern matches highlighted.

        Args:
            smiles_string (str): The SMILES string of the molecule.
            smarts_pattern (str): The SMARTS pattern to highlight.
            size (tuple, optional): The dimensions (width, height) of the output image.
                                    Defaults to (400, 300).

        Returns:
            PIL.Image.Image | None: A PIL Image object of the molecule.
                                    - If SMARTS matches are found, they are highlighted.
                                    - If no SMARTS matches are found, the unhighlighted molecule is returned.
                                    - Returns None if the SMILES string or SMARTS pattern is invalid.
        """
        try:
            mol = Chem.MolFromSmiles(smiles_string)
            if not mol:
                print(f"Error: Could not parse SMILES string: '{smiles_string}'")
                return None

            pat = Chem.MolFromSmarts(smarts_pattern)
            if not pat:
                print(f"Error: Could not parse SMARTS pattern: '{smarts_pattern}'")
                return None

            matches = mol.GetSubstructMatches(pat)
            print(matches)
            atom_indices_to_highlight = []
            if matches:
                atom_indices_to_highlight = list(set(idx for match in matches for idx in match))

            if not mol.GetNumConformers():
                try:
                    Chem.rdDepictor.Compute2DCoords(mol)
                except Exception as e:
                    print(f"Warning: Could not generate 2D coordinates: {e}. Drawing may be suboptimal.")

            img = Draw.MolToImage(mol, size=size, highlightAtoms=atom_indices_to_highlight if atom_indices_to_highlight else None, includeAtomNumbers=False, bondLineWidth=1.5)

            return img

        except Exception as e:
            print(f"An unexpected error occurred in plot_molecule_with_smarts_highlight: {e}")
            # You might want to print the traceback for more details
            # import traceback
            # traceback.print_exc()
            return None

    return (plot_molecule_with_smarts_highlight,)


@app.cell
def _(plot_molecule_with_smarts_highlight):
    smiles = "O=C(N1CC(CCCCCCCC)CCCCCCCCCC)C2=CC(C3=CC=C(Br)S3)=C4C(C2=C5C1=O)=C(C(N(CC(CCCCCCCC)CCCCCCCCCC)C4=O)=O)C=C5C6=CC=C(S6)C(S7)=CC=C7C8=NC(C9=CC=C(C%10=CC=C(C%11=CC(C(N(CC(CCCCCCCC)CCCCCCCCCC)C%12=O)=O)=C(C%12=C(C%13=CC=C(Br)S%13)C=C%14C(N%15CC(CCCCCCCC)CCCCCCCCCC)=O)C%14=C%11C%15=O)S%10)S9)=NC(C%16=CC=C(C%17=CC=C(C%18=CC(C(N(CC(CCCCCCCC)CCCCCCCCCC)C%19=O)=O)=C(C%19=C(C%20=CC=C(Br)S%20)C=C%21C(N%22CC(CCCCCCCC)CCCCCCCCCC)=O)C%21=C%18C%22=O)S%17)S%16)=N8"
    smarts = "[HS]"

    plot_molecule_with_smarts_highlight(
        smiles_string=smiles,
        smarts_pattern=smarts,
        size=(800, 600),
    )
    return


if __name__ == "__main__":
    app.run()
