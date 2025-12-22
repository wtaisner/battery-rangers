import marimo

__generated_with = "0.18.0"
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
    mo.md(
        r"""
    # ChEMBL35
    """
    )
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
def _(sf):
    def try_encoder(smiles):
        try:
            if smiles is not None:
                return sf.encoder(smiles)
            return None
        except sf.EncoderError:
            return None
        except Exception:
            return None

    return (try_encoder,)


@app.cell
def _(df, mp, pd, try_encoder):
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
    mo.md(
        r"""
    ## Prepare CHEMBL for machine learning tasks
    """
    )
    return


@app.cell
def _(pd):
    df_3 = pd.read_parquet("data/chembl_35_sqlite/chembl_35.parquet")
    return (df_3,)


@app.cell
def _(df_3, pd, re, tqdm):
    # ==============================================================================
    # NEW CODE BLOCK TO INSERT: Filter SELFIES by Token Length
    # ==============================================================================
    print("\nFiltering SELFIES by token length to remove extreme outliers...")

    # This is the most important parameter. 100 is a sane default for drug-like molecules.
    # This will fix the `max_str_len` issue and dramatically speed up training.
    MAX_SELFIES_TOKEN_LENGTH = 128
    token_pattern = re.compile("(\\[[^\\]]+\\])")

    def count_selfies_tokens(selfies_string):
        """Counts the number of SELFIES tokens in a string."""
        if pd.isna(selfies_string) or not isinstance(selfies_string, str):
            return 0
        return len(token_pattern.findall(selfies_string))

    # Create a new column with the token length
    # Note: We are still working with the original df_3 here
    print("Calculating token length for each SELFIES string in df_3...")
    df_3["selfies_len"] = [count_selfies_tokens(s) for s in tqdm(df_3["selfies"], desc="Counting Tokens")]

    # Perform the actual filtering to create our new, clean DataFrame
    df_filtered = df_3[df_3["selfies_len"] <= MAX_SELFIES_TOKEN_LENGTH].copy()

    # Report the results of filtering
    original_count = len(df_3)
    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count
    print(f"\nOriginal dataset size: {original_count}")
    print(f"Filtered dataset size: {filtered_count}")
    print(f"Molecules removed:     {removed_count} ({removed_count / original_count:.2%})")
    return (
        MAX_SELFIES_TOKEN_LENGTH,
        count_selfies_tokens,
        df_filtered,
        token_pattern,
    )


@app.cell
def _(count_selfies_tokens, pd, try_encoder):
    # see what is the length of substrates
    ctf = pd.read_csv("data/raw/experts_15_09_25_ctf_filtered.csv")
    ctf["selfies_length"] = ctf["smiles_substrate"].apply(lambda x: count_selfies_tokens(try_encoder(x)))
    ctf["selfies_length"].describe()
    return


@app.cell
def _(df_filtered):
    # df_filtered["canonical_smiles"].to_csv("data/chembl_35_sqlite/chembl_35_custom.smi", sep=" ", index=None, header=None)
    df_filtered["selfies"].to_csv("data/chembl_35_sqlite/chembl_35.slf", sep=" ", index=None, header=None)
    return


@app.cell
def _(json, pd, re, tqdm):
    def create_selfies_config(df: pd.DataFrame, selfies_column: str, max_len: int, output_path: str):
        """
        Generates a configuration JSON with SELFIES vocabulary and a defined max string length.
        Vocabulary is derived ONLY from the provided DataFrame (typically the training set).
        """
        print(f"Generating vocabulary from the '{selfies_column}' column of the training data...")

        # Use the same token pattern as before
        token_pattern = re.compile("(\\[[^\\]]+\\])")

        # Use a set for efficient collection of unique tokens
        vocabulary_set = set(["[STOP]"])  # Start with the essential STOP token

        for selfies_string in tqdm(df[selfies_column], desc="Building Vocabulary from Train Set"):
            if pd.isna(selfies_string) or not isinstance(selfies_string, str):
                continue
            tokens = token_pattern.findall(selfies_string)
            vocabulary_set.update(tokens)

        # Sort the vocabulary for consistency
        vocabulary_list = sorted(list(vocabulary_set))

        output_data = {"vocabulary": vocabulary_list, "max_str_len": max_len}  # Use the pre-defined max length from filtering

        print(f"\nFinal vocabulary size: {len(vocabulary_list)} unique tokens.")
        print(f"Max string length set to: {max_len}")

        try:
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=4)
            print(f"Configuration saved successfully to '{output_path}'")
        except IOError as e:
            print(f"Error saving JSON file: {e}")

    return (create_selfies_config,)


@app.cell
def _(
    MAX_SELFIES_TOKEN_LENGTH,
    create_selfies_config,
    df_3,
    df_filtered,
    token_pattern,
    train_test_split,
):
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
        train_df, temp_df = train_test_split(df_filtered, test_size=(val_prop + test_prop), random_state=random_seed)  # Calculate the size of the non-training part (0.10 + 0.20 = 0.30)

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

        # IMPORTANT: Generate the vocabulary using ONLY the train_df to prevent data leakage.
        # The max_str_len is already known from our filtering step in Step 1.
        output_file = "data/chembl35_vocab.json"
        create_selfies_config(train_df, "selfies", MAX_SELFIES_TOKEN_LENGTH, output_file)

        # Assuming you have train_df and val_df from your split
        train_vocab = set()
        for s in train_df["selfies"]:
            train_vocab.update(token_pattern.findall(s))

        val_vocab = set()
        for s in val_df["selfies"]:
            val_vocab.update(token_pattern.findall(s))

        # Find tokens in validation that are NOT in training
        oov_tokens = val_vocab - train_vocab

        print(f"Found {len(oov_tokens)} out-of-vocabulary tokens.")
        if len(oov_tokens) > 0:
            print("OOV Tokens:", oov_tokens)

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


if __name__ == "__main__":
    app.run()
