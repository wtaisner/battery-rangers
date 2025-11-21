"""Module for handling the molecule properties database."""
import datetime
import os
import sqlite3
from queue import Queue
from threading import Thread
from typing import Any, Dict, Optional


class MoleculeDB:
    """
    Class to handle a SQLite database for storing and retrieving molecule properties.
    It automatically connects, sets up the schema, and provides methods for
    adding data, performing fast lookups, and creating backups.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        self._connect()
        self._enable_wal_mode()
        self._create_table()

        self.write_queue = Queue()
        self.writer_thread = Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def _writer_loop(self):
        """The dedicated writer thread's main loop."""
        while True:
            # Block until an item is available in the queue
            item = self.write_queue.get()

            # Use a sentinel value (None) to signal the thread to exit
            if item is None:
                break

            # If it's not the sentinel, it's data to be written
            self._add_molecule_to_db(item)
            self.write_queue.task_done()

    def _add_molecule_to_db(self, properties: Dict[str, Any]):
        """The actual database insertion logic, only called by the writer thread."""
        insert_sql = """
                     INSERT \
                     OR IGNORE INTO molecules (
            canon_smiles, smarts_filter, conjugation_filter, flatness,
            normalized_csm, similarity, steric_hindrance, selfies
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
                     """
        data_tuple = (
            properties["canon_smiles"],
            int(properties["smarts_filter"]),
            int(properties["conjugation_filter"]),
            properties["flatness"],
            properties["normalized_csm"],
            properties["similarity"],
            int(properties["steric_hindrance"]),
            properties["selfies"],
        )
        try:
            with self.connection:
                self.connection.execute(insert_sql, data_tuple)
        except sqlite3.Error as e:
            print(f"Writer thread DB error: {e}")

    def add_molecule(self, **properties):
        """
        Public method to add a molecule. Instead of writing directly,
        it puts the properties dictionary onto the queue for the writer thread.
        """
        self.write_queue.put(properties)

    def close(self):
        """Gracefully shut down the writer thread and close the connection."""
        print("Closing database: waiting for writer queue to empty...")
        self.write_queue.join()  # Wait for all pending writes to complete
        self.write_queue.put(None)  # Send sentinel to stop the writer thread
        self.writer_thread.join()  # Wait for the thread to terminate

        if self.connection:
            self.connection.close()
            self.connection = None
            print("Database connection closed.")

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

    def get_molecule_properties(self, canon_smiles: str) -> Optional[Dict[str, Any]]:
        """
        Looks up a molecule by its canonical SMILES and returns all its properties.

        :param canon_smiles: The canonical SMILES string of the molecule to find.
        :return: A dictionary of the molecule's properties if found, otherwise None.
        """
        if not self.connection:
            print("Error: No active database connection.")
            return None

        result_tuple = None
        column_names = []

        # Retry loop to handle "Cursor needed to be reset" errors
        max_retries = 3
        for attempt in range(max_retries):
            cursor = None
            try:
                cursor = self.connection.cursor()
                cursor.execute("SELECT * FROM molecules WHERE canon_smiles = ?", (canon_smiles,))
                result_tuple = cursor.fetchone()

                # If found, grab column names immediately while cursor is valid
                if result_tuple:
                    column_names = [description[0] for description in cursor.description]

                # Success - break the retry loop
                break
            except sqlite3.InterfaceError:
                if attempt == max_retries - 1:
                    print(f"Warning: Failed to fetch properties for {canon_smiles} after retries.")
                    return None
            except Exception as e:
                print(f"Database error for {canon_smiles}: {e}")
                return None
            finally:
                if cursor:
                    cursor.close()

        if result_tuple:
            properties = dict(zip(column_names, result_tuple))

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
