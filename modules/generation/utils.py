"""Utility functions for generation module."""
import math


def score_value_exponential(value: float, min_val: float = 2.0, max_val: float = 50.0, decay_rate: float = 0.1) -> float:
    """
    Scores a value based on its proximity to the range [min_val, max_val].

    - Score is 1.0 if value is within the range [min_val, max_val].
    - Score decreases exponentially based on distance outside the range.

    Args:
      value: The numerical value to score.
      min_val: The lower bound of the optimal range.
      max_val: The upper bound of the optimal range.
      decay_rate: Controls how quickly the score drops off with distance.
                  A higher value means a faster drop.

    Returns:
      The calculated score (between 0 and 1.0).
    """
    try:
        if min_val <= value <= max_val:
            return 1.0
        if value < min_val:
            distance = min_val - value
            # Exponential decay: score = exp(-k * distance)
            return math.exp(-decay_rate * distance)
        # value > max_val
        distance = value - max_val
        # Exponential decay: score = exp(-k * distance)
        return math.exp(-decay_rate * distance)
    except Exception as e:
        # Handle any unexpected errors gracefully
        return 0.0
