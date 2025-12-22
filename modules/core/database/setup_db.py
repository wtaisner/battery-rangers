"""Module for handling the molecule properties database."""
import datetime
import os
import sqlite3
import time
from queue import Queue
from threading import Thread
from typing import Any, Dict, Optional


class MoleculeDB:
    """
    Class to handle a SQLite database for storing and retrieving molecule properties.

    Fixes implemented:
    1. Separate database connections for the main thread (Reader) and background thread (Writer).
    2. Retry logic for reads to handle concurrency edge cases.
    3. Explicit cursor lifecycle management to prevent memory/resource leaks.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

        # 1. Setup the Reader Connection (Main Thread)
        self.reader_connection = self._create_connection()
        self._initialize_db_schema()

        # 2. Setup the Writer Queue and Thread
        self.write_queue = Queue()
        self.writer_thread = Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def _create_connection(self) -> sqlite3.Connection:
        """Helper to create a properly configured SQLite connection."""
        # check_same_thread=False allows this specific connection object to be passed
        # around if you eventually use multi-threaded DataLoaders, though we try to avoid sharing it.
        # timeout=30 waits 30s for the lock to clear before raising an error.
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
            # Enable Write-Ahead Logging for better concurrency (Readers don't block Writers)
            conn.execute("PRAGMA journal_mode=WAL;")
            # synchronous=NORMAL is faster and safe enough for WAL mode
            conn.execute("PRAGMA synchronous=NORMAL;")
            return conn
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            raise

    def _initialize_db_schema(self):
        """Create the table using the reader connection."""
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS molecules (
                canon_smiles TEXT PRIMARY KEY,
                smarts_filter INTEGER NOT NULL,
                conjugation_filter INTEGER NOT NULL,
                flatness REAL NOT NULL,
                normalized_csm REAL NOT NULL,
                similarity REAL NOT NULL,
                steric_hindrance INTEGER NOT NULL,
                selfies TEXT NOT NULL
            )
        """
        with self.reader_connection:
            self.reader_connection.execute(create_table_sql)

    def _writer_loop(self):
        """
        The dedicated writer thread's main loop.
        CRITICAL FIX: This thread opens its OWN connection to the DB.
        """
        # Create a private connection for this thread
        writer_conn = self._create_connection()

        while True:
            # Block until an item is available
            item = self.write_queue.get()

            # Sentinel check to exit thread
            if item is None:
                break

            # Perform the write using the private connection
            self._write_to_db(writer_conn, item)
            self.write_queue.task_done()

        # Clean up the private connection when thread exits
        writer_conn.close()

    def _write_to_db(self, conn: sqlite3.Connection, properties: Dict[str, Any]):
        """Internal write logic using the specific writer connection."""
        insert_sql = """
            INSERT OR IGNORE INTO molecules (
                canon_smiles, smarts_filter, conjugation_filter, flatness,
                normalized_csm, similarity, steric_hindrance, selfies
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        # Safe casting
        try:
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

            with conn:
                conn.execute(insert_sql, data_tuple)

        except sqlite3.Error as e:
            print(f"Writer thread DB error for {properties.get('canon_smiles', 'UNKNOWN')}: {e}")
        except KeyError as e:
            print(f"Missing key in properties dict during write: {e}")

    def add_molecule(self, **properties):
        """
        Public API: Non-blocking add. Puts data into the queue.
        """
        self.write_queue.put(properties)

    def get_molecule_properties(self, canon_smiles: str) -> Optional[Dict[str, Any]]:
        """
        Robustly fetch molecule properties, handling potential concurrency noise.
        """
        if not self.reader_connection:
            print("Error: No active database connection.")
            return None

        result_tuple = None
        column_names = []

        # Retry loop to handle transient DB lock/state issues
        max_retries = 3

        for attempt in range(max_retries):
            cursor = None
            try:
                # Always create a fresh cursor
                cursor = self.reader_connection.cursor()
                cursor.execute("SELECT * FROM molecules WHERE canon_smiles = ?", (canon_smiles,))
                result_tuple = cursor.fetchone()

                if result_tuple:
                    column_names = [description[0] for description in cursor.description]

                # Success, exit retry loop
                break

            except sqlite3.OperationalError:
                # "Database is locked" - wait a bit and retry
                time.sleep(0.05 * (attempt + 1))
            except sqlite3.InterfaceError:
                # "Cursor needed to be reset" - logic flow issue, usually fixed by fresh cursor
                if attempt == max_retries - 1:
                    print(f"Warning: DB InterfaceError for {canon_smiles} after retries.")
            except Exception as e:
                print(f"Unexpected DB error for {canon_smiles}: {e}")
                return None
            finally:
                # CRITICAL: Always close the cursor
                if cursor:
                    cursor.close()

        # Parse result
        if result_tuple:
            properties = dict(zip(column_names, result_tuple))

            # Convert integers back to booleans
            bool_columns = ["smarts_filter", "conjugation_filter", "steric_hindrance"]
            for col in bool_columns:
                if col in properties:
                    properties[col] = bool(properties[col])
            return properties

        return None

    def backup_db(self):
        """Creates a backup using the reader connection."""
        if not self.reader_connection:
            print("Error: No active connection to back up.")
            return

        today = datetime.date.today().strftime("%Y-%m-%d")
        db_dir, db_filename = os.path.split(self.db_path)
        # Handle case where db_path is just a filename
        if not db_dir:
            db_dir = "."

        backup_filename = f"backup_{today}_{db_filename}"
        backup_path = os.path.join(db_dir, backup_filename)

        print(f"Starting backup from '{self.db_path}' to '{backup_path}'...")
        try:
            backup_conn = sqlite3.connect(backup_path)
            with backup_conn:
                self.reader_connection.backup(backup_conn)
            backup_conn.close()
            print("Backup completed successfully.")
        except sqlite3.Error as e:
            print(f"Backup failed: {e}")

    def close(self):
        """Gracefully shut down."""
        print("Closing database: waiting for writer queue to empty...")
        # 1. Wait for pending writes
        self.write_queue.join()

        # 2. Signal thread to stop
        self.write_queue.put(None)
        self.writer_thread.join()

        # 3. Close reader connection
        if self.reader_connection:
            self.reader_connection.close()
            self.reader_connection = None
            print("Database connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # For backward compatibility if 'connection' attribute was accessed directly
    @property
    def connection(self):
        return self.reader_connection
