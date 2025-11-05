"""Module for handling the molecule properties database."""
import datetime
import os
import sqlite3
from typing import Any, Dict, Optional


class MoleculeDB:
    """
    Class to handle a SQLite database for storing and retrieving molecule properties.
    It automatically connects, sets up the schema, and provides methods for
    adding data, performing fast lookups, and creating backups.
    """

    def __init__(self, db_path: str):
        """
        Initialize the database connector, connect, and set up the table.

        :param db_path: Path to the SQLite database file (e.g., 'data/molecules.db').
        """
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None

        # Connect and set up the database upon initialization
        self._connect()
        self._enable_wal_mode()
        self._create_table()

    def _connect(self):
        """Establish a connection to the SQLite database."""
        try:
            # The check_same_thread=False is important for use cases where you might
            # share the DB connection across different threads, common in RL envs.
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            raise

    def _enable_wal_mode(self):
        """
        Enable Write-Ahead Logging (WAL) for better concurrency.
        WAL allows multiple readers to operate while data is being written.
        """
        with self.connection:
            self.connection.execute("PRAGMA journal_mode=WAL;")

    def _create_table(self):
        """Create the 'molecules' table if it doesn't already exist."""
        create_table_sql = """
                           CREATE TABLE IF NOT EXISTS molecules
                           (
                               canon_smiles
                               TEXT
                               PRIMARY
                               KEY,
                               smarts_filter
                               INTEGER
                               NOT
                               NULL,
                               conjugation_filter
                               INTEGER
                               NOT
                               NULL,
                               flatness
                               REAL
                               NOT
                               NULL,
                               normalized_csm
                               REAL
                               NOT
                               NULL,
                               similarity
                               REAL
                               NOT
                               NULL,
                               steric_hindrance
                               INTEGER
                               NOT
                               NULL,
                               selfies
                               TEXT
                               NOT
                               NULL
                           ) \
                           """
        with self.connection:
            self.connection.execute(create_table_sql)

    def add_molecule(
        self,
        canon_smiles: str,
        smarts_filter: bool,
        conjugation_filter: bool,
        flatness: float,
        normalized_csm: float,
        similarity: float,
        steric_hindrance: bool,
        selfies: str,
    ) -> bool:
        """
        Add a new molecule and its properties to the database.
        If a molecule with the same canon_smiles already exists, it will be ignored.

        :return: True if a new row was inserted, False if it was ignored (already exists).
        """
        insert_sql = """
                     INSERT
                     OR IGNORE INTO molecules (
            canon_smiles, smarts_filter, conjugation_filter, flatness,
            normalized_csm, similarity, steric_hindrance, selfies
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
                     """
        data_tuple = (canon_smiles, int(smarts_filter), int(conjugation_filter), flatness, normalized_csm, similarity, int(steric_hindrance), selfies)

        with self.connection:
            cursor = self.connection.cursor()
            cursor.execute(insert_sql, data_tuple)
            return cursor.rowcount > 0

    def get_molecule_properties(self, canon_smiles: str) -> Optional[Dict[str, Any]]:
        """
        Looks up a molecule by its canonical SMILES and returns all its properties.
        This operation is highly optimized due to the primary key index.

        :param canon_smiles: The canonical SMILES string of the molecule to find.
        :return: A dictionary of the molecule's properties if found, otherwise None.
        """
        if not self.connection:
            print("Error: No active database connection.")
            return None

        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM molecules WHERE canon_smiles = ?", (canon_smiles,))
        result_tuple = cursor.fetchone()

        if result_tuple:
            # Get column names from the cursor description
            column_names = [description[0] for description in cursor.description]
            properties = dict(zip(column_names, result_tuple))

            # Convert integer columns back to booleans for consistency
            bool_columns = ["smarts_filter", "conjugation_filter", "steric_hindrance"]
            for col in bool_columns:
                if col in properties:
                    properties[col] = bool(properties[col])
            return properties
        else:
            # Molecule not found in the database
            return None

    def backup_db(self):
        """
        Creates a backup of the current database file.
        The backup is named 'backup_<YYYY-MM-DD>_<original_name>.db'
        and saved in the same directory.
        """
        if not self.connection:
            print("Error: No active connection to back up.")
            return

        today = datetime.date.today().strftime("%Y-%m-%d")
        db_dir, db_filename = os.path.split(self.db_path)
        backup_filename = f"backup_{today}_{db_filename}"
        backup_path = os.path.join(db_dir, backup_filename)

        print(f"Starting backup from '{self.db_path}' to '{backup_path}'...")
        try:
            backup_conn = sqlite3.connect(backup_path)
            with backup_conn:
                self.connection.backup(backup_conn)
            backup_conn.close()
            print("Backup completed successfully.")
        except sqlite3.Error as e:
            print(f"Backup failed: {e}")

    def close(self):
        """Close the database connection if it's open."""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("Database connection closed.")

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context and ensure the connection is closed."""
        self.close()


# --- Example Usage ---
if __name__ == "__main__":
    DB_FILE = "molecules_example.db"

    # Instantiate the database object once, outside your main loop
    db = MoleculeDB(DB_FILE)
    print(f"Database '{DB_FILE}' initialized.")

    # --- Population Step (done once) ---
    db.add_molecule(canon_smiles="CCO", smarts_filter=True, conjugation_filter=False, flatness=0.98, normalized_csm=1.2, similarity=0.85, steric_hindrance=False, selfies="[C][C][O]")
    db.add_molecule(canon_smiles="c1ccccc1", smarts_filter=True, conjugation_filter=True, flatness=1.0, normalized_csm=0.5, similarity=0.9, steric_hindrance=False, selfies="[c][c][c][c][c][c]")

    # --- Fast Lookup Step (done many times in your RL loop) ---
    print("\n--- Performing Lookups ---")

    # Case 1: Molecule exists
    mol_smiles_1 = "CCO"
    properties_1 = db.get_molecule_properties(mol_smiles_1)
    if properties_1:
        print(f"Found properties for '{mol_smiles_1}':")
        print(properties_1)
        assert properties_1["smarts_filter"] is True  # Note: The value is a proper boolean
    else:
        print(f"Molecule '{mol_smiles_1}' not found.")

    print("-" * 20)

    # Case 2: Molecule does not exist
    mol_smiles_2 = "C(F)(F)(F)C"
    properties_2 = db.get_molecule_properties(mol_smiles_2)
    if properties_2:
        print(f"Found properties for '{mol_smiles_2}':")
        print(properties_2)
    else:
        print(f"Molecule '{mol_smiles_2}' not found. (As expected)")

    # --- Backup and Close ---
    print("\n--- Backing up and closing ---")
    db.backup_db()
    db.close()  # Explicitly close when not using a 'with' statement```
