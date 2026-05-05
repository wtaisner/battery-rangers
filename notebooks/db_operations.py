import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import shutil
    import sqlite3
    import sys

    # Ensure the user has the selfies library installed
    try:
        import selfies as sf
    except ImportError:
        print("Error: The 'selfies' library is not installed.")
        print("Please install it by running: pip install selfies")
        sys.exit(1)

    # Database paths
    DB_PATH = "modules/bionemo/data/mol_db/substrate_properties.db"
    BACKUP_PATH = "modules/bionemo/data/mol_db/substrate_properties_backup.db"

    def is_valid_selfies(text):
        """
        Uses the official 'selfies' library to check if a string is a perfectly valid SELFIES string.
        """
        if not text:
            return 0

        # Fast heuristic: SELFIES strings always start with '[' and end with ']'
        # (Prevents running the slower decoder on obvious SMILES strings)
        if not (text.startswith("[") and text.endswith("]")):
            return 0

        try:
            # 1. The official library must be able to decode it into a SMILES string.
            # This raises sf.DecoderError if the SELFIES is syntactically invalid.
            _ = sf.decoder(text)

            # 2. Ensure no stray non-SELFIES characters are hiding in the string.
            # (e.g., "[C]c1ccccc1" might decode the "[C]", but leave SMILES artifacts).
            # sf.split_selfies perfectly extracts known valid SELFIES tokens.
            tokens = list(sf.split_selfies(text))
            if "".join(tokens) != text:
                return 0

            return 1
        except Exception:
            # Catch sf.DecoderError or any other library-specific exceptions
            return 0

    def main():
        if not os.path.exists(DB_PATH):
            print(f"Error: Database not found at {DB_PATH}")
            return

        # 1. Create a safety backup
        print(f"Creating a safety backup at {BACKUP_PATH}...")
        shutil.copy2(DB_PATH, BACKUP_PATH)

        # 2. Connect to the Database
        conn = sqlite3.connect(DB_PATH)

        # Register our Python validator into SQLite
        conn.create_function("IS_VALID_SELFIES", 1, is_valid_selfies)
        cursor = conn.cursor()

        try:
            # 3. Create a temporary table to store Row IDs of matches.
            # We do this so we only execute the Python validation function ONCE per row
            # instead of twice (which would be required if we did standard INSERT + DELETE).
            print("Evaluating rows using the `selfies` library (this may take a moment)...")
            cursor.execute("CREATE TEMP TABLE selfies_to_move (row_id INTEGER PRIMARY KEY)")

            # Scan the database and save the IDs of mistakenly placed SELFIES
            cursor.execute(
                """
                INSERT INTO selfies_to_move
                SELECT rowid FROM molecules WHERE IS_VALID_SELFIES(canon_smiles) = 1
            """
            )

            # Check how many we found
            cursor.execute("SELECT COUNT(*) FROM selfies_to_move")
            count = cursor.fetchone()[0]

            print(f"Found {count} rows where 'canon_smiles' contains a correct SELFIES string.")

            if count == 0:
                print("No SELFIES found in the 'canon_smiles' column. Exiting.")
                conn.close()
                return

            # 4. Move them to a new table
            new_table_name = "molecules_selfies_moved"
            print(f"Moving {count} rows to new table '{new_table_name}'...")

            # Spawn the new table by cloning the exact schema of the 'molecules' table
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {new_table_name} AS SELECT * FROM molecules WHERE 0")

            # Copy the mismatched rows into the new table using our Temp Table IDs
            cursor.execute(
                f"""
                INSERT INTO {new_table_name}
                SELECT * FROM molecules WHERE rowid IN (SELECT row_id FROM selfies_to_move)
            """
            )

            # Delete the mismatched rows from the original table using the Temp Table IDs
            cursor.execute(
                """
                DELETE FROM molecules WHERE rowid IN (SELECT row_id FROM selfies_to_move)
            """
            )

            conn.commit()
            print(f"Successfully moved rows and cleaned up the 'molecules' table.")

            # Reclaim unused physical hard-drive space freed up by the DELETE statement
            print("Vacuuming database to optimize and reclaim unused space...")
            cursor.execute("VACUUM")
            print("Done!")

        except sqlite3.Error as e:
            print(f"An SQLite error occurred: {e}")
            conn.rollback()  # Undo any partial changes if an error happens
        finally:
            conn.close()

    main()
    return


if __name__ == "__main__":
    app.run()
