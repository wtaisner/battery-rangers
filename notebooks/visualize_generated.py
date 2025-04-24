import marimo

__generated_with = "0.13.1"
app = marimo.App(width="full")


@app.cell
def _():
    import glob
    import os
    import random
    from typing import List, Optional, Tuple

    import pandas as pd
    from IPython.core.display import Image as IPyImage
    from rdkit import Chem, rdBase
    from rdkit.Chem import Draw

    # Suppress RDKit warnings (optional, can be noisy)
    rdBase.DisableLog("rdApp.warning")
    return Chem, Draw, IPyImage, List, Optional, Tuple, glob, os, pd, random


@app.cell
def _(Tuple):
    _INPUT_PATH_PATTERN: str = "data/raw/experts_merged.smi"
    PNG_SUB_IMG_SIZE: Tuple[int, int] = (500, 500)
    MOLS_PER_ROW: int = 5
    MAX_DISPLAY_MOLS: int = 25
    return MAX_DISPLAY_MOLS, MOLS_PER_ROW, PNG_SUB_IMG_SIZE


@app.cell
def _(
    Chem,
    Draw,
    IPyImage,
    List,
    MAX_DISPLAY_MOLS,
    MOLS_PER_ROW,
    Optional,
    PNG_SUB_IMG_SIZE,
    Tuple,
    glob,
    os,
    pd,
    random,
):
    def find_smiles_column(df: pd.DataFrame) -> Optional[str]:
        """
        Finds the SMILES column in a DataFrame, checking for 'smiles' or 'SMILES'.

        Args:
            df: The pandas DataFrame to search within.

        Returns:
            The name of the SMILES column ('smiles' or 'SMILES') if found, otherwise None.
        """
        if "smiles" in df.columns:
            return "smiles"
        elif "SMILES" in df.columns:
            return "SMILES"
        else:
            return None

    def smiles_to_mols(smiles_list: List[str]) -> Tuple[List[Chem.Mol], List[str]]:
        """
        Converts a list of SMILES strings to RDKit Mol objects.

        Args:
            smiles_list: A list of SMILES strings.

        Returns:
            A tuple containing:
            - A list of valid RDKit Mol objects with 'Original_SMILES' property set.
            - A list of SMILES strings that failed to parse or were invalid.
        """
        mols: List[Chem.Mol] = []
        invalid_smiles: List[str] = []
        for i, smi in enumerate(smiles_list):
            if not isinstance(smi, str) or not smi.strip():
                invalid_smiles.append(str(smi))
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mol.SetProp("_Name", f"Mol_{i}")
                mol.SetProp("Original_SMILES", smi)
                mols.append(mol)
            else:
                invalid_smiles.append(smi)
        return (mols, invalid_smiles)

    def generate_and_save_png_grid(mols: List[Chem.Mol], output_filepath: str, legends: Optional[List[str]] = None, mols_per_row: int = 5, sub_img_size: Tuple[int, int] = (300, 300)) -> Optional[str]:
        """
        Generates an RDKit molecule grid image, saves it ONLY as a PNG file,
        handling potential IPython object returns, and returns the filepath upon success.

        Args:
            mols: List of RDKit Mol objects to include in the grid.
            output_filepath: The full path where the PNG image file should be saved
                             (must end with .png).
            legends: Optional list of strings to display below each molecule.
                     Defaults to original SMILES if available.
            mols_per_row: Maximum number of molecules per row in the grid.
            sub_img_size: Tuple representing the size (width, height) of each sub-image.
                          Larger sizes result in higher resolution PNGs.

        Returns:
            - The output filepath as a string, if saving is successful.
            - None if no molecules are provided or saving fails.
        """
        if not mols:
            print("  ⚠️ No molecules provided for PNG grid generation.")
            return None
        if not output_filepath.lower().endswith(".png"):
            print(f"  ❌ Error: Output filepath '{output_filepath}' must end with .png")
            return None
        effective_legends = legends
        if effective_legends and len(effective_legends) != len(mols):
            print(f"  ⚠️ Legend count ({len(effective_legends)}) != mol count ({len(mols)}). Using default legends.")
            effective_legends = None
        if not effective_legends:
            try:
                effective_legends = [m.GetProp("Original_SMILES") for m in mols]
            except KeyError:
                print("  ⚠️ Could not find 'Original_SMILES' property for default legends. Omitting legends.")
                effective_legends = [""] * len(mols)
        generated_object = None
        try:
            generated_object = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=sub_img_size, legends=effective_legends, useSVG=False)
            if isinstance(generated_object, IPyImage) and hasattr(generated_object, "data"):
                png_data = generated_object.data
                if isinstance(png_data, bytes):
                    with open(output_filepath, "wb") as f:
                        f.write(png_data)
                    print(f"  ✅ PNG Image saved to: {output_filepath} (SubImgSize: {sub_img_size})")
                    return output_filepath
                else:
                    print(f"  ❌ Error: IPython Image object's .data attribute is not bytes (Type: {type(png_data)}). Cannot save.")
                    return None
            else:
                print("  ❌ Error: RDKit MolsToGridImage (PNG) did not return a recognized Image object with data attribute.")
                print(f"     Object type received: {type(generated_object)}")
                return None
        except Exception as e:
            print(f"  ❌ Error during PNG generation/saving for {output_filepath}: {type(e).__name__} - {e}")
            return None

    def process_csv_file(filepath: str):
        """
        Reads a CSV file, processes SMILES, generates and saves a molecule grid PNG.
        **Image rendering in the notebook output is disabled.**

        Args:
            filepath: Path to the input CSV file.
        """
        filename = os.path.basename(filepath)
        output_dir = os.path.dirname(filepath)
        base_filename = os.path.splitext(filename)[0]
        output_png_filename = f"{base_filename}_molecules.png"
        output_png_filepath = os.path.join(output_dir, output_png_filename)
        print(f"Processing: {filename}")
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                print(f"  ⚠️ Skipping: File {filename} is empty.")
                print("-" * 30)
                return
            smiles_col = find_smiles_column(df)
            if smiles_col is None:
                print("  ⚠️ Skipping: No 'smiles' or 'SMILES' column found.")
                print("-" * 30)
                return
            smiles_list = df[smiles_col].dropna().astype(str).tolist()
            if not smiles_list:
                print("  ⚠️ Skipping: No valid SMILES strings found after cleaning.")
                print("-" * 30)
                return
            mols, invalid_smiles = smiles_to_mols(smiles_list)
            if invalid_smiles:
                print(f"  ℹ️ Found {len(invalid_smiles)} invalid/empty SMILES strings. Examples: {invalid_smiles[:5]}")
            if not mols:
                print("  ❌ Skipping: Could not generate any valid molecules.")
                print("-" * 30)
                return
            num_mols = len(mols)
            print(f"  ℹ️ Found {num_mols} valid molecules.")
            mols_to_draw: List[Chem.Mol]
            if num_mols <= MAX_DISPLAY_MOLS:
                mols_to_draw = mols
            else:
                print(f"  ℹ️ Sampling {MAX_DISPLAY_MOLS} molecules randomly from {num_mols}.")
                mols_to_draw = random.sample(mols, MAX_DISPLAY_MOLS)
            try:
                legends = [m.GetProp("Original_SMILES") for m in mols_to_draw]
            except KeyError:
                legends = None
            png_filepath_or_none = generate_and_save_png_grid(mols=mols_to_draw, output_filepath=output_png_filepath, legends=legends, mols_per_row=MOLS_PER_ROW, sub_img_size=PNG_SUB_IMG_SIZE)
            if not png_filepath_or_none:
                print(f"  ⚠️ PNG image generation/saving failed for {filename}.")
        except FileNotFoundError:
            print(f"  ❌ Error: File not found at {filepath}")
        except pd.errors.EmptyDataError:
            print(f"  ⚠️ Skipping: File {filename} is empty.")
        except pd.errors.ParserError:
            print(f"  ❌ Error: Could not parse CSV file {filename}. Check format.")
        except Exception as e:
            print(f"  ❌ An unexpected error occurred while processing {filename}: {type(e).__name__} - {e}")
        finally:
            print("-" * 30)

    if "INPUT_PATH_PATTERN" not in globals():
        print("❌ Configuration Error: INPUT_PATH_PATTERN is not defined.")
        _INPUT_PATH_PATTERN = "*.csv"
    print(f"Searching for files matching pattern: '{_INPUT_PATH_PATTERN}'")
    file_list: List[str] = glob.glob(_INPUT_PATH_PATTERN, recursive=True)
    if not file_list:
        print(f"❌ No files found matching pattern: '{_INPUT_PATH_PATTERN}'")
        print("   Please check the INPUT_PATH_PATTERN variable in section 2.")
    else:
        print(f"✅ Found {len(file_list)} file(s):")
        file_list.sort()
        for f_path in file_list:
            print(f"  - {f_path}")
        print("-" * 30)
        if "process_csv_file" not in globals() or not callable(process_csv_file):
            print("❌ Execution Error: 'process_csv_file' function not defined or not callable.")
            print("   Please ensure the function definition in Cell 4 has been executed.")
        else:
            for filepath in file_list:
                process_csv_file(filepath)
        print("\nProcessing finished.")
    return


if __name__ == "__main__":
    app.run()
