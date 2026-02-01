"""
Evaluation module for Chinese Chess positions.

Provides NNUE-based evaluation with material fallback.
"""

import numpy as np
from typing import Optional

from ..nnue.nue_network import NNUE
from ..features.feature_transformer import extract_features
from .material import material_evaluation


class Evaluator:
    """
    Position evaluator using NNUE network.

    Falls back to material evaluation if NNUE is not available.
    """

    def __init__(self, nnue: Optional[NNUE] = None):
        """
        Initialize evaluator.

        Args:
            nnue: NNUE network (if None, uses material evaluation)
        """
        self.nnue = nnue
        self.use_nnue = nnue is not None

    def evaluate(self, board: np.ndarray, perspective: int = 1) -> int:
        """
        Evaluate board position.

        Args:
            board: 10×9 board array
            perspective: Side to evaluate from (1=Red, -1=Black)

        Returns:
            Evaluation score in centipawns
            Positive = good for Red, Negative = good for Black
        """
        if self.use_nnue and self.nnue is not None:
            return self._evaluate_nnue(board, perspective)
        else:
            return self._evaluate_material(board, perspective)

    def _evaluate_nnue(self, board: np.ndarray, perspective: int) -> int:
        """
        Evaluate using NNUE network.

        Args:
            board: 10×9 board array
            perspective: Side to evaluate from

        Returns:
            Evaluation score in centipawns
        """
        # Extract features
        features = extract_features(board)

        # Get raw score from NNUE
        raw_score = self.nnue.forward(features)

        # Adjust for perspective
        if perspective == -1:
            raw_score = -raw_score

        # Clamp score to reasonable range
        score = max(-10000, min(10000, int(raw_score)))

        return score

    def _evaluate_material(self, board: np.ndarray, perspective: int) -> int:
        """
        Evaluate using material counts.

        Args:
            board: 10×9 board array
            perspective: Side to evaluate from

        Returns:
            Evaluation score in centipawns
        """
        score = material_evaluation(board)

        # Adjust for perspective
        if perspective == -1:
            score = -score

        return score

    def set_nnue(self, nnue: NNUE) -> None:
        """Set NNUE network."""
        self.nnue = nnue
        self.use_nnue = True

    def disable_nnue(self) -> None:
        """Disable NNUE and use material evaluation."""
        self.use_nnue = False


def create_evaluator(
    nnue_path: Optional[str] = None,
    input_size: int = 1260,
    hidden_size: int = 256
) -> Evaluator:
    """
    Create an evaluator.

    Args:
        nnue_path: Path to NNUE weights file (if None, uses material only)
        input_size: NNUE input size
        hidden_size: NNUE hidden size

    Returns:
        Configured Evaluator instance
    """
    nnue = None

    if nnue_path is not None:
        nnue = NNUE(input_size=input_size, hidden_size=hidden_size, random_init=False)
        try:
            nnue.load(nnue_path)
        except Exception as e:
            print(f"Warning: Failed to load NNUE from {nnue_path}: {e}")
            print("Using material evaluation instead.")
            nnue = None

    return Evaluator(nnue)


# Convenience function for quick evaluation
def quick_evaluate(board: np.ndarray) -> int:
    """
    Quick material evaluation for debugging.

    Args:
        board: 10×9 board array

    Returns:
        Material score (positive = Red advantage)
    """
    return material_evaluation(board)
