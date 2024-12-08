"""Generate report on how many molecules are passing the filters"""
import argparse
import logging
import os
from glob import glob
from multiprocessing import Pool

import pandas as pd

from modules.core.features.molecule_filter import MoleculeFilter

# Set up the logger for this specific module
logger = logging.getLogger(__name__)  # Logger specific to this module
handler = logging.StreamHandler()  # Handler for console output
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
# Set up the logger for the modules used in this script
logging.getLogger("modules.core.features.molecule_filter").setLevel(logging.CRITICAL)
logging.getLogger("modules.core.features.filters.flatness_filter").setLevel(logging.CRITICAL)
logging.getLogger("modules.core.features.filters.point_group_symmetry_filter").setLevel(logging.CRITICAL)


parser = argparse.ArgumentParser(description="Generate report on how many molecules are passing the filters.")

parser.add_argument(
    "--files",
    type=str,
    help="Path to the files to evaluate. Supports glob patterns.",
    default="../data/sampling/*/*.csv",
)
parser.add_argument(
    "--output_comparison",
    type=str,
    help="Path to the output file.",
    default="../data/sampling/filtering_comparison.csv",
)
parser.add_argument(
    "--output_filtered",
    type=str,
    help="Path to the output file with the filtered molecules.",
    default="../data/sampling/filtered_molecules.csv",
)
parser.add_argument(
    "--num_processes",
    type=int,
    help="Number of processes to use for multiprocessing.",
    default=20,
)


def process_file(file: str | os.PathLike) -> dict:
    """
    Function to process a single file and apply molecule filtering.
    """

    logger.info(f"Processing file: {file.split('/')[-1]}.")
    molecule_filter = MoleculeFilter()

    try:
        smiles = pd.read_csv(file)["SMILES"].drop_duplicates().values
    except:  # pylint: disable=bare-except
        smiles = pd.read_csv(file)["smiles"].drop_duplicates().values

    smiles_filtered = molecule_filter.apply(smiles)

    logger.info(f"Finished processing file: {file.split('/')[-1]}.")
    return {"filename": file.split("/")[-1], "num_total_molecules": len(smiles), "num_filtered_molecules": len(smiles_filtered), "smiles_after_filtering": smiles_filtered}


def main(args: argparse.Namespace):
    """Run the main script."""
    # Dictionary to store results
    result_dict = {"filenames": [], "num_total_molecules": [], "num_filtered_molecules": [], "smiles_after_filtering": []}
    # Get all files to evaluate
    files = glob(args.files)
    logger.info(f"Found {len(files)} files to evaluate.")
    # Use multiprocessing to process files
    with Pool(args.num_processes) as pool:
        # process_partial = partial(process_file)
        results = pool.map(process_file, files)

    # Aggregate results
    all_smiles_that_passed = set()
    for result in results:
        result_dict["filenames"].append(result["filename"])
        result_dict["num_total_molecules"].append(result["num_total_molecules"])
        result_dict["num_filtered_molecules"].append(result["num_filtered_molecules"])
        result_dict["smiles_after_filtering"].append(result["smiles_after_filtering"])
        all_smiles_that_passed |= set(result["smiles_after_filtering"])

    pd.DataFrame(result_dict).to_csv(args.output_comparison, index=False)
    pd.DataFrame({"smiles": list(all_smiles_that_passed)}).to_csv(args.output_filtered, index=False)


if __name__ == "__main__":
    main(parser.parse_args())
