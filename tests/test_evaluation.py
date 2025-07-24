"""Test the MoleculeGenerationEvaluator class."""
import pytest

from modules.generation.evaluation import MoleculeGenerationEvaluator


@pytest.fixture
def get_data():
    """Fixture to provide example data for testing."""
    generated_smiles_example = [
        "CCO",  # duplicate of training / generated
        "CCC",  # novel, valid
        "c1ccccc1",  # duplicate of training
        "invalid-smiles-string",  # invalid
        "CC(=O)O",  # duplicate of training
        "CCO",  # duplicate of training / generated
    ]

    training_smiles_example = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic Acid
        "C",  # Methane
        "CC",  # Ethane
        "c1ccccc1",  # Benzene (make one generated mol non-novel)
        "CC(C)C",  # Isobutane
    ]

    reference_smiles_example = [
        "O=C=O",  # Carbon Dioxide
        "c1ccc(C(=O)O)cc1",  # Benzoic Acid (canonical)
        "Nc1ccccc1",  # Aniline
    ]
    return generated_smiles_example, training_smiles_example, reference_smiles_example


def test_evaluate_molecule(get_data):
    """Test the evaluate method of the MoleculeGenerationEvaluator class."""
    generated_smiles, training_smiles, reference_smiles = get_data
    evaluator = MoleculeGenerationEvaluator(
        training_smiles=training_smiles,
        reference_smiles=reference_smiles,
        generated_smiles=generated_smiles,
    )

    results = evaluator.evaluate()

    assert results["uniqueness"] == 0.8
    assert results["validity"] == (5 / 6)
    assert results["novelty_wrt_training_set"] == 0.25
    assert results["novelty_wrt_reference_set"] == 1.0
