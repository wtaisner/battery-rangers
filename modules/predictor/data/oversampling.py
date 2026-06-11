"""Oversampling functions."""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder


def dist(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculates the Euclidean distance between two data points."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def get_rare_cases_relevance(y: np.ndarray, te: float) -> np.ndarray:
    """Identifies rare cases based on the relevance of the target variable values.

    Args:
        y (np.ndarray): Dataset target variable.
        te (float): Threshold for the relevance of the target variable.

    Returns:
        np.ndarray: Indices of rare cases.

    """
    relevance_func = create_inverse_density_relevance_function(y)

    # Determine the median of the target variable
    tilde_y = np.median(y)

    # Identify rare low and high cases
    relevances = relevance_func(y)
    rare_l_indices = np.where((relevances > te) & (y < tilde_y))[0]
    rare_h_indices = np.where((relevances > te) & (y > tilde_y))[0]

    rare_ids = np.concatenate((rare_l_indices, rare_h_indices), axis=0) if len(rare_l_indices) > 0 and len(rare_h_indices) > 0 else rare_l_indices if len(rare_l_indices) > 0 else rare_h_indices

    return rare_ids


def get_rare_cases_bins(y: np.ndarray, te: int) -> list[int]:
    """Identifies rare cases based number of bins.

    Args:
        y (np.ndarray): A dataset target variable.
        te (int): A number of bins.

    Returns:
        list[int]: Indices of rare cases.

    """
    rare_ids = []
    rng = np.random.default_rng(seed=42)
    bins = pd.cut(y, bins=te, labels=False)
    unique_bins, counts = np.unique(bins, return_counts=True)
    largest_bin = np.argmax(counts)
    largest_bin_size = counts[largest_bin]
    largest_bin_size *= 0.67
    largest_bin_size = int(largest_bin_size)
    for b in unique_bins:
        if b != largest_bin:
            small_ids = np.argwhere(bins == b).flatten()
            oversampling_count = largest_bin_size - len(small_ids)
            while oversampling_count >= len(small_ids):
                rare_ids.extend(small_ids)
                oversampling_count -= len(small_ids)
            if oversampling_count > 0:
                sampled_rare = rng.choice(small_ids, oversampling_count, replace=False)
                rare_ids.extend(sampled_rare)
    return rare_ids


def smoter(
    x: np.ndarray,
    y: np.ndarray,
    te: float,
    o: float,
    k: int,
    numeric_cols: list[int],
    oversampling_type: str = "SMOTER",
) -> tuple[np.ndarray, np.ndarray]:
    """Implements the SMOTER algorithm for regression with rare extreme values.

    Args:
        x (np.ndarray): The training data features.
        y (np.ndarray): The training data target variable.
        te (float): The threshold for relevance of the target variable values.
        o (float): The percentage of over-sampling (e.g., 200 for 200%).
        k (int): The number of nearest neighbors used in case generation.
        numeric_cols (list[int]): A list of column indices representing numeric attributes.
        oversampling_type (str): Type of oversampling to use (SMOTER or bins).

    Returns:
        tuple: A tuple containing the new training data features (x_resampled) and
               the new training data target variable (y_resampled).

    """
    if oversampling_type == "SMOTER":
        rare_ids = get_rare_cases_relevance(y, te)
    else:
        rare_ids = get_rare_cases_bins(y, te)

    # Generate synthetic cases for rare low and high values
    new_cases_x, new_cases_y = gen_synth_cases(rare_ids, o, k, numeric_cols, x, y)

    x_resampled = np.concatenate((x, new_cases_x), axis=0) if len(new_cases_x) > 0 else x
    y_resampled = np.concatenate((y, new_cases_y), axis=0) if len(new_cases_y) > 0 else y

    return x_resampled, y_resampled


def gen_synth_cases(
    rare_ids: np.ndarray | list[int],
    o: float,
    k: int,
    numeric_cols: list[int],
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic cases using the SMOTER approach.

    Args:
        rare_ids (np.ndarray | list[int]): Indices of rare cases to oversample.
        o (float): The percentage of over-sampling (e.g., 200 for 200%).
        k (int): The number of nearest neighbors to consider.
        numeric_cols (list[int]): A list of column indices representing numeric attributes.
        x (np.ndarray): The original training data features.
        y (np.ndarray): The original training data target variable.

    Returns:
        tuple: A tuple containing the synthetic features (x_synthetic) and the
               synthetic target variable (y_synthetic).

    """
    x_synthetic = []
    y_synthetic = []
    rng = np.random.default_rng(seed=42)
    ng = o / 100.0  # Number of new cases to generate for each existing case

    if len(rare_ids) == 0:
        return np.array([]), np.array([])

    # Calculate nearest neighbors using the original dataset for wider coverage
    categorical_cols = [i for i in range(x.shape[1]) if i not in numeric_cols]
    if len(categorical_cols) > 0:
        enc = OneHotEncoder(drop="first")
        enc_features = enc.fit_transform(x[:, categorical_cols])
        x_encoded = np.concatenate((x[:, numeric_cols], enc_features.toarray()), axis=1)
    else:
        x_encoded = x
    knn = NearestNeighbors(n_neighbors=2 * k)  # +1 to exclude the case itself
    knn.fit(x_encoded)

    for i in rare_ids:
        case_x = x[i]
        case_y = y[i]
        case_x_encoded = x_encoded[i]

        # Find k-Nearest Neighbors, but exclude the case itself for a valid neighborhood
        neighbors = knn.kneighbors(case_x_encoded.reshape(1, -1), return_distance=False)[0]  # [1:]  # Exclude self
        neighbors = pd.DataFrame(neighbors).drop_duplicates(keep="first").values.flatten()
        neighbors = np.array([n for n in neighbors if not np.all(case_x == x[n])])[:k]
        for _ in range(int(ng)):
            # Randomly choose neighbor
            x_index = rng.choice(neighbors)
            neighbor_x = x[x_index]
            neighbor_x_encoded = x_encoded[x_index]
            neighbor_y = y[x_index]

            new_x = np.copy(case_x)  # Initialize the new features with a copy of 'case_x'

            # Generate Attribute Values
            for a in range(x.shape[1]):  # Iterate through the columns
                if a in numeric_cols:
                    diff = case_x[a] - neighbor_x[a]
                    new_x[a] = case_x[a] + rng.uniform(low=0, high=1) * diff
                    if isinstance(case_x[a], int) and isinstance(neighbor_x[a], int):
                        new_x[a] = int(new_x[a])
                else:
                    new_x[a] = rng.choice([case_x[a], neighbor_x[a]])  # Randomly select

            # Decide the target value
            if len(categorical_cols) > 0:
                enc_features = enc.transform(new_x[categorical_cols].reshape(1, -1)).toarray().flatten()
                new_x_encoded = np.concatenate((new_x[numeric_cols], enc_features))
            else:
                new_x_encoded = new_x
            d1 = dist(new_x_encoded, case_x_encoded)
            d2 = dist(new_x_encoded, neighbor_x_encoded)
            new_y = (d2 * case_y + d1 * neighbor_y) / (d1 + d2)

            x_synthetic.append(new_x)
            y_synthetic.append(new_y)

    return np.array(x_synthetic), np.array(y_synthetic)


def create_inverse_density_relevance_function(y_values: np.ndarray):
    """Creates a relevance function that assigns higher relevance to values
    in regions of lower density.  This implements the idea that rare values
    are more relevant.

    Args:
        y_values (np.ndarray): The target variable values.

    Returns:
        callable: A function that takes a target value and returns its relevance.

    """
    kde = gaussian_kde(y_values)

    def relevance(y):
        """Calculates the relevance as the inverse of the probability density,
        normalized to the range [0, 1].
        """
        density = kde(y)
        if np.isscalar(density):
            if density == 0:
                return 1.0  # Handle zero density (rare case)
            density_value = float(density)
            return 1.0 / density_value
        density[density == 0] = np.finfo(float).eps  # Avoid division by zero
        return 1.0 / density

    # Normalize relevance values to [0, 1]
    relevance_values = relevance(y_values)
    min_relevance = np.min(relevance_values)
    max_relevance = np.max(relevance_values)

    def normalized_relevance(y):
        raw_relevance = relevance(y)
        normalized = (raw_relevance - min_relevance) / (max_relevance - min_relevance)
        return normalized

    return normalized_relevance


if __name__ == "__main__":
    df = pd.read_csv("../../../data/fingerprints_maccs/data_experts1.csv")
    df_features = df.drop(columns=["smiles", "capacity_max"])
    x = df.drop(columns=["smiles", "capacity_max"]).values
    y = df["capacity_max"].values
    print(df.duplicated(subset=[c for c in df.columns if c not in ["smiles", "capacity_max"]]))
    print(df.iloc[[9, 21]][["smiles", "capacity_max"]])
    print(df.columns)
    te = 0.5  # Relevance threshold
    o = 300  # Over-sampling percentage
    k = 5  # Number of nearest neighbors
    numeric_cols = [i for i in range(len(df_features.columns)) if df_features.columns[i] not in ["symmetry"]]  # Indices of numeric attributes
    numeric_cols = []
    # Create the relevance function
    relevance_func = create_inverse_density_relevance_function(y)

    # You can then pass this relevance_func to the smoter function:
    x_resampled, y_resampled = smoter(x, y, te, o, k, numeric_cols, oversampling_type="SMOTER")

    print("Original x shape:", x.shape)
    print("Original y shape:", y.shape)
    print("Resampled x shape:", x_resampled.shape)
    print("Resampled y shape:", y_resampled.shape)
    print("Resampled x:\n", x_resampled)
    print("Resampled y:\n", y_resampled)

    print(f"Duplicated rows: {pd.DataFrame(x_resampled).duplicated().sum()}")
