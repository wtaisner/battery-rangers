import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import multiprocessing as mp
    import re
    import sqlite3

    import marimo as mo
    from tqdm import tqdm

    return mo, mp, re, sqlite3, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Chembl""")
    return


@app.cell
def _():
    # !wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35_sqlite.tar.gz
    # !tar -xvzf chembl_35_sqlite.tar.gz
    return


@app.cell(hide_code=True)
def _(mo):
    schema_url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35_schema.png"
    alt_text = "ChEMBL database schema"

    mo.image(schema_url, alt=alt_text)
    return


@app.cell
def _(pd, sqlite3):
    cnx = sqlite3.connect("data/chembl_35_sqlite/chembl_35.db")
    df_2 = pd.read_sql_query("SELECT * FROM main.compound_structures", cnx)
    df_2.shape
    return cnx, df_2


@app.cell
def _(df_2):
    df_2.head()
    return


@app.cell
def _(df_2, mp, pd, sf):
    def try_encoder(smiles):
        try:
            if smiles is not None:
                return sf.encoder(smiles)
            return None
        except sf.EncoderError:
            return None
        except Exception:
            return None

    def process_chunk(chunk):
        return chunk.apply(try_encoder)

    if "canonical_smiles" in df_2.columns:
        num_processes = 12
        chunk_size = (len(df_2) + num_processes - 1) // num_processes
        chunks = [df_2["canonical_smiles"].iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(num_processes)]
        with mp.Pool(processes=num_processes) as pool:
            processed_chunks = pool.map(process_chunk, chunks)
        df_2["selfies"] = pd.concat(processed_chunks)
        print(df_2.isna().sum())
        df_2.dropna(inplace=True)
        print(df_2.shape)
    else:
        print("The 'canonical_smiles' column is not found")
    return


@app.cell
def _(cnx, pd):
    fdf = pd.read_sql_query("SELECT * FROM main.compound_properties", cnx)  #  tabela z cechami do pretrenowania ew. modelu.
    fdf.columns
    return (fdf,)


@app.cell
def _(df_2, fdf, pd):
    merged_df = pd.merge(df_2, fdf, on="molregno", how="inner")
    merged_df.to_parquet("../data/chembl_35_sqlite/chembl_35.parquet")
    return


@app.cell
def _(pd):
    df_3 = pd.read_parquet("../data/chembl_35_sqlite/chembl_35.parquet")
    return (df_3,)


@app.cell
def _(df_3):
    df_3["canonical_smiles"].to_csv("../data/raw/chembl_35_smiles.txt", sep=" ", index=None, header=None)
    df_3["selfies"].to_csv("../data/raw/chembl_35_selfies.txt", sep=" ", index=None, header=None)
    return


@app.cell
def _(df_3, experts_selfies, json, pd, re, tqdm):
    def create_selfies_config_with_reference(original_df: pd.DataFrame, original_selfies_column: str, reference_df: pd.DataFrame, reference_selfies_column: str, output_json_path: str):
        """
        Generates a configuration JSON with SELFIES vocabulary and max string length.

        The vocabulary includes tokens from both the original and reference DataFrames.
        The max string length is determined SOLELY by the reference DataFrame.

        Args:
            original_df: The primary pandas DataFrame containing SELFIES strings.
            original_selfies_column: Column name in original_df with SELFIES.
            reference_df: The reference DataFrame (e.g., experts) with SELFIES.
            reference_selfies_column: Column name (can be integer like 0) in reference_df.
            output_json_path: Path to save the output JSON file.
        """
        if original_selfies_column not in original_df.columns:
            raise ValueError(f"Column '{original_selfies_column}' not found in the original DataFrame.")
        if reference_selfies_column not in reference_df.columns:
            raise ValueError(f"Column '{reference_selfies_column}' not found in the reference DataFrame.")
        token_pattern = re.compile("(\\[[^\\]]+\\])")
        combined_vocabulary_set = set(["[STOP]"])
        reference_max_token_len = 0
        print(f"Processing REFERENCE SELFIES from column '{reference_selfies_column}' for max token length and vocabulary...")
        for selfies_string in tqdm(reference_df[reference_selfies_column], desc="Processing Reference"):
            if pd.isna(selfies_string) or not isinstance(selfies_string, str):
                continue
            tokens = token_pattern.findall(selfies_string)
            reference_max_token_len = max(reference_max_token_len, len(tokens))
            combined_vocabulary_set.update(tokens)
        print(f"Max token length determined from reference data: {reference_max_token_len}")
        vocabulary_list = sorted(list(combined_vocabulary_set))
        output_data = {"vocabulary": vocabulary_list, "max_str_len": reference_max_token_len}
        print(f"\nCombined vocabulary size (Original + Reference): {len(vocabulary_list)} unique tokens.")
        try:
            with open(output_json_path, "w") as f:
                json.dump(output_data, f, indent=4)
            print(f"Configuration saved successfully to '{output_json_path}'")
        except IOError as e:
            print(f"Error saving JSON file: {e}")

    col_name = "selfies"
    output_file = "selfies_config.json"
    create_selfies_config_with_reference(df_3, col_name, experts_selfies, 0, output_file)
    print("\n--- Verifying saved JSON file ---")
    try:
        with open(output_file, "r") as f:
            loaded_config = json.load(f)
            print(json.dumps(loaded_config, indent=2))
            assert loaded_config["max_str_len"] == 24
            assert "[P]" in loaded_config["vocabulary"]
            assert "[Br]" in loaded_config["vocabulary"]
            assert "[F]" in loaded_config["vocabulary"]
            print("\nVerification checks passed.")
    except FileNotFoundError:
        print(f"File '{output_file}' not found.")
    except json.JSONDecodeError:
        print(f"File '{output_file}' is not valid JSON.")
    except AssertionError as e:
        print(f"Verification check failed: {e}")
    return


@app.cell
def _(Chem, df_3):
    atomic_numbers = set()
    bonds_numbers = []
    for smiles in df_3["canonical_smiles"]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            bonds_numbers.append(mol.GetNumBonds())
            if mol is not None:
                for atom in mol.GetAtoms():
                    atomic_numbers.add(atom.GetAtomicNum())
        except:
            pass
    return (bonds_numbers,)


@app.cell
def _(bonds_numbers):
    max(bonds_numbers)
    return


if __name__ == "__main__":
    app.run()
