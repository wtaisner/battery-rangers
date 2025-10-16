import marimo

__generated_with = "0.16.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import multiprocessing as mp
    import re
    import sqlite3

    import marimo as mo
    import pandas as pd
    import selfies as sf
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    return json, mo, mp, pd, re, sf, sqlite3, tqdm, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# ChEMBL35""")
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
    df = pd.read_sql_query("SELECT * FROM main.compound_structures", cnx)
    df.shape
    return cnx, df


@app.cell
def _():
    # see what tables are in main
    # tables_df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", cnx)
    # tables_df
    return


@app.cell
def _(df):
    # drop duplicates in canonical_smiles column
    df.drop_duplicates(subset=["canonical_smiles"], inplace=True)
    df.shape
    return


@app.cell
def _(df):
    df.head(2)
    return


@app.cell
def _(df, mp, pd, sf):
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

    if "canonical_smiles" in df.columns:
        num_processes = 12
        chunk_size = (len(df) + num_processes - 1) // num_processes
        chunks = [df["canonical_smiles"].iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(num_processes)]
        with mp.Pool(processes=num_processes) as pool:
            processed_chunks = pool.map(process_chunk, chunks)
        df["selfies"] = pd.concat(processed_chunks)
        print(df.isna().sum())
        df.dropna(inplace=True)
        print(df.shape)
    else:
        print("The 'canonical_smiles' column is not found")
    return


@app.cell
def _(cnx, pd):
    feature_df = pd.read_sql_query("SELECT * FROM main.compound_properties", cnx)  #  tabela z cechami do pretrenowania ew. modelu.
    feature_df.columns
    return (feature_df,)


@app.cell
def _(df, feature_df, pd):
    merged_df = pd.merge(df, feature_df, on="molregno", how="inner")
    merged_df.to_parquet("data/chembl_35_sqlite/chembl_35.parquet")
    merged_df.to_csv("data/chembl_35_sqlite/chembl_35.csv", index=None)
    return (merged_df,)


@app.cell
def _(merged_df):
    merged_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prepare CHEMBL for machine learning tasks""")
    return


@app.cell
def _(pd):
    df_3 = pd.read_parquet("data/chembl_35_sqlite/chembl_35.parquet")
    return (df_3,)


@app.cell
def _(df_3):
    df_3["canonical_smiles"].to_csv("data/chembl_35_sqlite/chembl_35_custom.smi", sep=" ", index=None, header=None)
    df_3["selfies"].to_csv("data/chembl_35_sqlite/chembl_35.slf", sep=" ", index=None, header=None)
    return


@app.cell
def _(df_3, train_test_split):
    def _():
        import os

        # --- Configuration ---
        # Assume df_3 is your loaded DataFrame with the data
        # Example: df_3 = pd.read_parquet("path/to/your/input_data.parquet")
        # Make sure df_3 has a column named 'selfies'
        # Define split proportions
        test_prop = 0.20  # 20% for test
        val_prop = 0.10  # 10% for validation
        # train_prop is implicitly 1.0 - test_prop - val_prop = 0.70

        random_seed = 23  # For reproducible splits
        output_dir = "data/chembl_35_sqlite/"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # --- Data Splitting ---

        print(f"Original DataFrame size: {len(df_3)}")

        # Step 1: Split into training (70%) and a temporary set (30% for val + test)
        train_df, temp_df = train_test_split(df_3, test_size=(val_prop + test_prop), random_state=random_seed)  # Calculate the size of the non-training part (0.10 + 0.20 = 0.30)

        # Step 2: Split the temporary set (30% of original) into validation (10% of original) and test (20% of original)
        # The test set size relative to the temporary set is test_prop / (val_prop + test_prop)
        # e.g., 0.20 / 0.30 = 2/3
        relative_test_size = test_prop / (val_prop + test_prop)

        val_df, test_df = train_test_split(temp_df, test_size=relative_test_size, random_state=random_seed)  # Use the same random state for deterministic split of the temp set

        # --- Verification ---
        print(f"Train DataFrame size:      {len(train_df)} ({len(train_df)/len(df_3):.2%})")
        print(f"Validation DataFrame size: {len(val_df)} ({len(val_df)/len(df_3):.2%})")
        print(f"Test DataFrame size:       {len(test_df)} ({len(test_df)/len(df_3):.2%})")
        print("-" * 30)

        # --- Saving Data ---

        # Save full DataFrames to Parquet
        print("Saving Parquet files...")
        train_df.to_parquet(os.path.join(output_dir, "chembl_35_train.parquet"))
        val_df.to_parquet(os.path.join(output_dir, "chembl_35_val.parquet"))
        test_df.to_parquet(os.path.join(output_dir, "chembl_35_test.parquet"))
        print("Parquet saving complete.")

        # Save full DataFrames to CSV (without index)
        print("Saving CSV files...")
        train_df.to_csv(os.path.join(output_dir, "chembl_35_train.csv"), index=None)
        val_df.to_csv(os.path.join(output_dir, "chembl_35_val.csv"), index=None)
        test_df.to_csv(os.path.join(output_dir, "chembl_35_test.csv"), index=None)
        print("CSV saving complete.")

        # Save only the 'selfies' column to .slf files (space-separated, no index, no header)
        print("Saving Selfies (.slf) files...")
        if "selfies" in train_df.columns:
            train_df["selfies"].to_csv(os.path.join(output_dir, "chembl_35_train.slf"), sep=" ", index=None, header=None)
            val_df["selfies"].to_csv(os.path.join(output_dir, "chembl_35_val.slf"), sep=" ", index=None, header=None)
            test_df["selfies"].to_csv(os.path.join(output_dir, "chembl_35_test.slf"), sep=" ", index=None, header=None)
            print("Selfies saving complete.")
        else:
            print("Warning: 'selfies' column not found in DataFrame. Skipping .slf file creation.")
        return print("\nAll files saved.")

    _()
    return


@app.cell
def _(pd, train_test_split):
    def _():
        import os

        # --- Configuration ---
        input_smi_file = "data/chembl_35_sqlite/chembl_35.smi"
        output_dir = "data/chembl_35_sqlite/"

        # Define split proportions
        test_prop = 0.20  # 20% for test
        val_prop = 0.10  # 10% for validation
        # train_prop is implicitly 1.0 - test_prop - val_prop = 0.70

        random_seed = 23  # For reproducible splits

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # --- Read Input SMILES File ---
        try:
            df = pd.read_csv(input_smi_file, sep=" ", header=None, on_bad_lines="warn")  # Read, warn about potential issues
            # Check if read correctly - sometimes SMILES files have extra info like IDs
            if df.shape[1] == 1:
                df.columns = ["canonical_smiles"]
            elif df.shape[1] > 1:
                print(f"Warning: Input SMILES file '{input_smi_file}' has {df.shape[1]} columns. Assuming the first column contains SMILES.")
                df = df.iloc[:, [0]]  # Select only the first column
                df.columns = ["canonical_smiles"]
            else:
                raise ValueError("Input SMILES file seems empty or incorrectly formatted.")

            print(f"Read {len(df)} lines from {input_smi_file}")

        except FileNotFoundError:
            print(f"Error: Input file not found at {input_smi_file}")
            exit()
        except Exception as e:
            print(f"Error reading input file {input_smi_file}: {e}")
            exit()

        # --- Data Splitting ---

        print(f"Original DataFrame size: {len(df)}")

        # Step 1: Split into training (70%) and a temporary set (30% for val + test)
        train_df, temp_df = train_test_split(df, test_size=(val_prop + test_prop), random_state=random_seed)  # 0.10 + 0.20 = 0.30

        # Step 2: Split the temporary set into validation (10% of original) and test (20% of original)
        relative_test_size = test_prop / (val_prop + test_prop)  # 0.20 / 0.30 = 2/3

        val_df, test_df = train_test_split(temp_df, test_size=relative_test_size, random_state=random_seed)  # Use the same random state

        # --- Verification ---
        print(f"Train SMILES count:      {len(train_df)} ({len(train_df)/len(df):.2%})")
        print(f"Validation SMILES count: {len(val_df)} ({len(val_df)/len(df):.2%})")
        print(f"Test SMILES count:       {len(test_df)} ({len(test_df)/len(df):.2%})")
        print("-" * 30)

        # --- Saving Data ---

        print("Saving SMILES (.smi) files...")
        # Save the 'canonical_smiles' column to .smi files (space-separated, no index, no header)
        train_df["canonical_smiles"].to_csv(os.path.join(output_dir, "chembl_35_train.smi"), sep=" ", index=None, header=None)
        val_df["canonical_smiles"].to_csv(os.path.join(output_dir, "chembl_35_val.smi"), sep=" ", index=None, header=None)
        test_df["canonical_smiles"].to_csv(os.path.join(output_dir, "chembl_35_test.smi"), sep=" ", index=None, header=None)

        print("SMILES saving complete.")
        return print("\nAll files saved.")

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generate a CHEMBL config for MolAIR""")
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


if __name__ == "__main__":
    app.run()
