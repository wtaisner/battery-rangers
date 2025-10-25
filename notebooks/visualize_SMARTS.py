import marimo

__generated_with = "0.14.13"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import io
    from typing import List

    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import MolDraw2DCairo

    def plot_molecule_with_smarts_highlight(smiles_string: str, smarts_patterns: List[str], size: tuple[int, int] = (400, 300)) -> Image.Image | None:
        """
        Generates a PIL Image of a molecule with multiple SMARTS pattern matches highlighted.
        This version uses a robust method that sets atom properties on the molecule
        itself to force labeling, avoiding known bugs with MolDrawOptions.

        The molecule is drawn with all hydrogen atoms shown explicitly, and all
        non-hydrogen atoms are labeled with their atomic symbols. Highlights are
        extended to include hydrogen atoms attached to matched heavy atoms.

        Args:
            smiles_string (str): The SMILES string of the molecule.
            smarts_patterns (List[str]): A list of SMARTS pattern strings to highlight.
            size (tuple, optional): The dimensions (width, height) of the output image.

        Returns:
            PIL.Image.Image | None: A PIL Image object of the molecule.
                                    Returns None if the SMILES is invalid.
        """
        try:
            mol = Chem.MolFromSmiles(smiles_string)
            if not mol:
                print(f"Error: Could not parse SMILES string: '{smiles_string}'")
                return None

            mol = Chem.AddHs(mol)

            # Set a custom 'atomLabel' property to force labeling.
            for atom in mol.GetAtoms():
                if atom.GetSymbol() != "H":
                    atom.SetProp("atomLabel", atom.GetSymbol())

            # This will hold all heavy atom indices from all successful SMARTS matches.
            all_heavy_atoms_in_matches = set()

            # 1. Loop through each provided SMARTS pattern.
            for smarts_str in smarts_patterns:
                pat = Chem.MolFromSmarts(smarts_str)
                if not pat:
                    print(f"Warning: Skipping invalid SMARTS pattern: '{smarts_str}'")
                    continue  # Move to the next pattern in the list

                matches = mol.GetSubstructMatches(pat)
                print(f"Found {matches} matches for SMARTS '{smarts_str}' in the molecule.")
                if matches:
                    # Add all unique atom indices from these matches to our master set.
                    current_matches_set = {idx for match in matches for idx in match}
                    all_heavy_atoms_in_matches.update(current_matches_set)

            atom_indices_to_highlight = []
            if all_heavy_atoms_in_matches:
                # Create the final set of atoms to highlight, starting with the heavy atoms.
                final_atoms_to_highlight = set(all_heavy_atoms_in_matches)

                # Iterate through the matched heavy atoms to find their hydrogen neighbors.
                for atom_idx in all_heavy_atoms_in_matches:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == "H":
                            final_atoms_to_highlight.add(neighbor.GetIdx())

                # Convert the final set to a sorted list for the drawing function.
                atom_indices_to_highlight = sorted(list(final_atoms_to_highlight))

            try:
                Chem.rdDepictor.Compute2DCoords(mol)
            except Exception as e:
                print(f"Warning: Could not generate 2D coordinates: {e}. Drawing may be suboptimal.")

            # Use the low-level canvas for stability.
            drawer = MolDraw2DCairo(size[0], size[1])
            opts = drawer.drawOptions()

            opts.addAtomIndices = False
            opts.bondLineWidth = 1.5

            # Draw the molecule with the combined highlight list.
            drawer.DrawMolecule(mol, highlightAtoms=atom_indices_to_highlight)
            drawer.FinishDrawing()

            png_data = drawer.GetDrawingText()
            return Image.open(io.BytesIO(png_data))

        except Exception as e:
            print(f"An unexpected error occurred in plot_molecule_with_smarts_highlight: {e}")
            return None

    return Chem, Draw, plot_molecule_with_smarts_highlight


@app.cell
def _(plot_molecule_with_smarts_highlight):
    # 29, 33, 34
    smiles = r"C/C=C1/C(=O)C(=CNc2cccc3c(N)cccc23)C(=O)/C(=C/C)C1=O"
    # smiles = r"C/C=C1C(/C(C(/C(C\1=O)=C/Nc2c3cccc(N)c3ccc2)=O)=C\C)=O"
    # smiles = r"CC=c(c(=O)c(=CC)c(=O)c1=CNc2ccc(C(c3cc(N)ccc3C4=O)=O)c4c2)c1=O"

    # # 35, 36
    # smiles = "O=Cc1c(O)c(c(n2)sc3c2cc(sc(c4c(O)c(C=O)c(O)c(C=O)c4O)n5)c5c3)c(O)c(C=O)c1O"
    # smiles = "O=Cc1cc(c(n2)sc3c2cc(sc(c4cc(C=O)cc(C=O)c4)n5)c5c3)cc(C=O)c1"

    patterns = [
        # "Br",
        # "Cl",
        "[NH2]",
        # "C#N",
        "[CH]=O",
        # "c1nccc1",
        # "[#6](-c)-[#7r6]-[#6](-c)",
        # "[OH]",
        # "O1-B-O-c:c1",
        # "[#6]=[#8]", # C=O
        # "[#7]1~[#6]~[#6]~[#7]~[#6]~[#6]~1"
        "C=C-N"
        # "s~c~n"
    ]

    plot_molecule_with_smarts_highlight(
        smiles_string=smiles,
        smarts_patterns=patterns,
        size=(800, 600),
    )
    return (smiles,)


@app.cell(hide_code=True)
def _(Chem, Draw):
    from modules.core.filters.conjugation_filter import ConjugationFilter

    conjugation_filter = ConjugationFilter()

    def visualize_conjugation(smiles, mol_size=(300, 300)):
        """
        Rysuje cząsteczkę i podświetla jej układ sprzężony.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Błąd SMILES: {smiles}")
            return

        Chem.SanitizeMol(mol)

        conjugated_bonds = []
        conjugated_atoms = set()

        for bond in mol.GetBonds():
            if bond.GetIsConjugated():
                conjugated_bonds.append(bond.GetIdx())
                conjugated_atoms.add(bond.GetBeginAtomIdx())
                conjugated_atoms.add(bond.GetEndAtomIdx())

        # Konwersja setu na listę na potrzeby funkcji rysującej
        conjugated_atoms_list = list(conjugated_atoms)

        # Wyświetlanie obrazu (działa najlepiej w środowiskach typu Jupyter Notebook)
        img = Draw.MolToImage(mol, size=mol_size, highlightAtoms=conjugated_atoms_list, highlightBonds=conjugated_bonds, highlightColor=(0.8, 0.8, 0))  # kolor podświetlenia (żółty)

        return img

    return conjugation_filter, visualize_conjugation


@app.cell
def _(Chem, conjugation_filter, smiles, visualize_conjugation):
    is_conjugated = conjugation_filter.apply([Chem.MolFromSmiles(smiles)])

    print(f"Is conjugated: {len(is_conjugated)}")

    visualize_conjugation(smiles, mol_size=(600, 600))
    return


if __name__ == "__main__":
    app.run()
