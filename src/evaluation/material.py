"""
Material evaluation for Chinese Chess.

Simple piece-value based evaluation for fallback and debugging.
"""

import numpy as np
from typing import Dict

from ..board.board_representation import (
    is_red, get_piece_type, EMPTY
)


# Piece values in centipawns
PIECE_VALUES = {
    1: 10000,  # General/King (infinite value, but use large number)
    2: 20,     # Advisor
    3: 30,     # Elephant
    4: 40,     # Horse
    5: 90,     # Rook
    6: 45,     # Cannon
    7: 10,     # Soldier
}


# Position bonuses (simplified piece-square tables)
# These tables give bonuses for pieces in certain positions
SOLDIER_BONUS_RED = [
    [0,  0,  0,  0,  0,  0,  0,  0,  0],  # Row 9 (back rank)
    [0,  0,  0,  0,  0,  0,  0,  0,  0],  # Row 8
    [0,  0,  0,  0,  0,  0,  0,  0,  0],  # Row 7
    [2,  4,  6,  8, 10,  8,  6,  4,  2],  # Row 6 (before river)
    [4,  8, 12, 16, 20, 16, 12,  8,  4],  # Row 5 (crossed river)
    [6, 12, 18, 24, 30, 24, 18, 12,  6],  # Row 4 (advanced)
    [8, 16, 24, 32, 40, 32, 24, 16,  8],  # Row 3
    [10, 20, 30, 40, 50, 40, 30, 20, 10],  # Row 2
    [12, 24, 36, 48, 60, 48, 36, 24, 12],  # Row 1
    [14, 28, 42, 56, 70, 56, 42, 28, 14],  # Row 0 (near promotion)
]

SOLDIER_BONUS_BLACK = [
    [14, 28, 42, 56, 70, 56, 42, 28, 14],  # Row 9 (near promotion)
    [12, 24, 36, 48, 60, 48, 36, 24, 12],  # Row 8
    [10, 20, 30, 40, 50, 40, 30, 20, 10],  # Row 7
    [8, 16, 24, 32, 40, 32, 24, 16,  8],  # Row 6
    [6, 12, 18, 24, 30, 24, 18, 12,  6],  # Row 5
    [4,  8, 12, 16, 20, 16, 12,  8,  4],  # Row 4 (crossed river)
    [2,  4,  6,  8, 10,  8,  6,  4,  2],  # Row 3
    [0,  0,  0,  0,  0,  0,  0,  0,  0],  # Row 2
    [0,  0,  0,  0,  0,  0,  0,  0,  0],  # Row 1
    [0,  0,  0,  0,  0,  0,  0,  0,  0],  # Row 0 (back rank)
]


def material_evaluation(board: np.ndarray) -> int:
    """
    Compute material evaluation.

    Positive score = Red advantage
    Negative score = Black advantage

    Args:
        board: 10×9 board array

    Returns:
        Material score in centipawns
    """
    score = 0

    for row in range(10):
        for col in range(9):
            piece = board[row, col]

            if piece == EMPTY:
                continue

            piece_type = abs(piece)
            piece_value = PIECE_VALUES.get(piece_type, 0)

            # Add position bonus for soldiers
            if piece_type == 7:  # Soldier
                if is_red(piece):
                    position_bonus = SOLDIER_BONUS_RED[row][col]
                else:
                    position_bonus = SOLDIER_BONUS_BLACK[row][col]
                piece_value += position_bonus

            # Red pieces add to score, Black subtract
            if is_red(piece):
                score += piece_value
            else:
                score -= piece_value

    return score


def count_pieces(board: np.ndarray) -> Dict[int, int]:
    """
    Count pieces on the board.

    Args:
        board: 10×9 board array

    Returns:
        Dictionary mapping piece values to counts
    """
    counts = {}

    for row in range(10):
        for col in range(9):
            piece = board[row, col]
            if piece != EMPTY:
                counts[piece] = counts.get(piece, 0) + 1

    return counts


def get_material_balance(board: np.ndarray) -> Dict[str, int]:
    """
    Get material balance for both sides.

    Args:
        board: 10×9 board array

    Returns:
        Dictionary with 'red' and 'black' material counts
    """
    red_material = 0
    black_material = 0

    for row in range(10):
        for col in range(9):
            piece = board[row, col]

            if piece == EMPTY:
                continue

            piece_type = abs(piece)
            piece_value = PIECE_VALUES.get(piece_type, 0)

            if is_red(piece):
                red_material += piece_value
            else:
                black_material += piece_value

    return {
        'red': red_material,
        'black': black_material,
        'difference': red_material - black_material
    }


def is_endgame(board: np.ndarray) -> bool:
    """
    Check if position is likely an endgame.

    Simple heuristic: endgame if total material < 2000 centipawns
    (roughly 2 rooks + some pieces remaining)

    Args:
        board: 10×9 board array

    Returns:
        True if endgame
    """
    total_material = 0

    for row in range(10):
        for col in range(9):
            piece = board[row, col]
            if piece != EMPTY and abs(piece) != 1:  # Exclude kings
                piece_type = abs(piece)
                total_material += PIECE_VALUES.get(piece_type, 0)

    return total_material < 2000
