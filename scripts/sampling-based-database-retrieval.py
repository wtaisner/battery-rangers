import argparse
import glob
import logging
import multiprocessing
import os
import sqlite3
from typing import List

import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def canonicalize_batch(smiles_batch: List[str]) -> List[str]:
    """Worker function to canonicalize a chunk of SMILES strings."""
    # Silence RDKit logs for workers
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    canonical_smiles = []
    for smiles in smiles_batch:
        if not isinstance(smiles, str) or not smiles.strip():
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles.append(Chem.MolToSmiles(mol))
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    return canonical_smiles


def main(
    glob_pattern: str,
    db_path: str,
    output_csv: str,
    table_name: str = "molecules",
    num_workers: int = None,
):
    """Finds CSVs, canonicalizes SMILES, and retrieves properties from the database."""
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return

    # 1. Browse and Extract Raw SMILES from CSVs
    logger.info(f"Searching for CSV files using pattern: {glob_pattern}")
    csv_files = glob.glob(glob_pattern, recursive=True)

    if not csv_files:
        logger.error("No CSV files found matching the pattern.")
        return

    logger.info(f"Found {len(csv_files)} CSV files. Extracting SMILES...")

    raw_smiles_set = set()
    for file in tqdm(csv_files, desc="Reading CSVs"):
        try:
            df = pd.read_csv(file, low_memory=False)
            # Support variations in column naming
            col_name = next((c for c in df.columns if c.upper() == "SMILES"), None)

            if col_name:
                raw_smiles_set.update(df[col_name].dropna().astype(str).tolist())
            else:
                logger.warning(f"No 'SMILES' column found in {file}")
        except Exception as e:
            logger.error(f"Failed to read {file}: {e}")

    logger.info(f"Extracted {len(raw_smiles_set)} raw unique SMILES strings.")

    # 2. Multiprocessing Canonicalization
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 2)

    logger.info(f"Canonicalizing SMILES using {num_workers} processes...")
    raw_smiles_list = list(raw_smiles_set)
    chunk_size = max(1, len(raw_smiles_list) // (num_workers * 4))
    batches = [raw_smiles_list[i : i + chunk_size] for i in range(0, len(raw_smiles_list), chunk_size)]

    unique_canon_smiles = set()
    with multiprocessing.Pool(processes=num_workers) as pool:
        for batch_results in tqdm(
            pool.imap_unordered(canonicalize_batch, batches),
            total=len(batches),
            desc="Canonicalizing",
        ):
            unique_canon_smiles.update(batch_results)

    logger.info(f"Resulted in {len(unique_canon_smiles)} unique canonical SMILES. Querying database...")

    # 3. Fast Database Query using Temporary Tables
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # Create a temporary table for our canonical SMILES
        cursor.execute("CREATE TEMPORARY TABLE temp_query_smiles (canon_smiles TEXT PRIMARY KEY)")

        # Insert unique canonical SMILES into the temp table
        smiles_tuples = [(s,) for s in unique_canon_smiles]
        cursor.executemany(
            "INSERT OR IGNORE INTO temp_query_smiles (canon_smiles) VALUES (?)",
            smiles_tuples,
        )

        # Perform an INNER JOIN to quickly grab all properties for matching SMILES
        # Using m.* grabs canon_smiles, smarts_filter, steric_hindrance, and any other columns
        query = f"""
            SELECT m.*
            FROM {table_name} m
            INNER JOIN temp_query_smiles t
            ON m.canon_smiles = t.canon_smiles
        """

        logger.info("Executing JOIN query against SQLite database...")
        results_df = pd.read_sql_query(query, conn)

        # 4. Save results
        logger.info(f"Match found for {len(results_df)} molecules. Saving to {output_csv}...")
        results_df.to_csv(output_csv, index=False)
        logger.info("Process completed successfully.")

    except Exception as e:
        logger.error(f"Database query failed: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate SMILES from CSVs and query DB for properties.")
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        required=True,
        help='Glob pattern for CSV files e.g. "data/sampling/**/*.csv"',
    )
    parser.add_argument("--db", dest="db_path", required=True, help="Path to the SQLite database.")
    parser.add_argument(
        "--out",
        dest="output_csv",
        required=True,
        help="Path to save the resulting CSV.",
    )
    parser.add_argument(
        "--table",
        dest="table_name",
        default="molecules",
        help="Table name in DB (default: molecules)",
    )
    parser.add_argument(
        "--workers",
        dest="num_workers",
        type=int,
        default=None,
        help="Number of CPU workers for canonicalization.",
    )

    args = parser.parse_args()

    main(
        glob_pattern=args.glob_pattern,
        db_path=args.db_path,
        output_csv=args.output_csv,
        table_name=args.table_name,
        num_workers=args.num_workers,
    )
