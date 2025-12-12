"""A multiprocessing script to update a SQLite database with SMARTS and steric hindrance scores"""
import logging
import multiprocessing
import os
import sqlite3
from typing import List, Tuple

from rdkit import Chem, RDLogger
from tqdm import tqdm

# pylint: disable=import-error
from modules.core.filters.smarts_filter import SMARTSFilter
from modules.core.filters.steric_hindrance_filter import StericHindranceFilter

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_batch(batch_data: List[Tuple[int, str]]) -> List[Tuple[float, float, int]]:
    """
    Worker function to process a chunk of molecules.
    """
    # --- SILENCE RDKIT LOGS ---
    # We do this inside the worker to ensure every process is silenced.
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    # --------------------------

    # Initialize filters locally
    smarts_filter = SMARTSFilter()
    steric_filter = StericHindranceFilter()

    results = []

    for rowid, smiles in batch_data:
        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            results.append((0.0, 0.0, rowid))
            continue

        try:
            s_score = smarts_filter.get_reward(mol)
            st_score = steric_filter.get_reward(mol)
            results.append((s_score, st_score, rowid))
        except Exception:  # pylint: disable=broad-exception-caught
            results.append((0.0, 0.0, rowid))

    return results


def update_db_multiprocess(db_path: str, table_name: str = "molecules", num_workers: int = None):
    """
    Orchestrates the multiprocessing update.
    """
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return

    # 1. Read Data (Main Thread)
    logger.info(f"Reading data from {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if table exists and has columns
        cursor.execute(f"SELECT rowid, canon_smiles FROM {table_name}")
        all_rows = cursor.fetchall()  # List of (rowid, smiles)
        conn.close()  # Close connection before spawning processes to be safe
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Error reading database: {e}")
        conn.close()
        return

    total_records = len(all_rows)
    logger.info(f"Loaded {total_records} records. Preparing multiprocessing...")

    # 2. Chunk Data
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 5)  # TODO: Leave N cores for OS/Main thread

    logger.info(f"Using {num_workers} worker processes.")

    # Calculate chunk size (e.g., split total into 4x number of workers to keep queue moving)
    chunk_size = max(1, total_records // (num_workers * 4))

    # Create chunks generator
    batches = [all_rows[i : i + chunk_size] for i in range(0, total_records, chunk_size)]

    # 3. Parallel Processing
    results_to_update = []

    with multiprocessing.Pool(processes=num_workers) as pool:
        # imap_unordered is usually slightly faster if order doesn't strictly matter for processing,
        # but we need to collect everything. tqdm wrapper shows progress.
        for batch_results in tqdm(pool.imap_unordered(process_batch, batches), total=len(batches), desc="Processing Batches"):
            results_to_update.extend(batch_results)

    # 4. Batch Write (Main Thread)
    logger.info(f"Calculations complete. Writing {len(results_to_update)} updates to database...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Turn off synchronous writing for speed (optional, risky if power fails, but good for bulk updates)
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("BEGIN TRANSACTION")

        query = f"""
            UPDATE {table_name}
            SET smarts_filter = ?,
                steric_hindrance = ?
            WHERE rowid = ?
        """

        cursor.executemany(query, results_to_update)
        conn.commit()
        logger.info("Database update successful.")

    except Exception as e:  # pylint: disable=broad-exception-caught
        conn.rollback()
        logger.error(f"Error writing to database: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    # Update Substrate DB
    print("--- Updating Substrate DB ---")
    update_db_multiprocess("modules/bionemo/data/mol_db/substrate_properties.db")

    # Update Node DB
    print("\n--- Updating Node DB ---")
    update_db_multiprocess("modules/bionemo/data/mol_db/node_properties.db")
