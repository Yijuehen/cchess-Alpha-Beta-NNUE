"""
Feature Transformer for NNUE

Converts board positions to feature vectors for NNUE evaluation.
Uses HalfKP-inspired scheme adapted for Chinese Chess.

Feature space: 90 positions × 7 piece types × 2 colors = 1,260 features
"""

import numpy as np
from typing import Tuple

from ..board.board_representation import (
    ROWS, COLS, EMPTY,
    is_red, is_black, get_piece_type,
    pos_to_coords, coords_to_pos
)


# Feature dimensions
FEATURE_SIZE = 1260  # 90 squares × 7 piece types × 2 colors
NUM_SQUARES = 90
NUM_PIECE_TYPES = 7


def get_feature_index(pos: int, piece: int) -> int:
    """
    Convert (position, piece) to feature index.

    Feature encoding: pos * 14 + piece_type * 2 + color
    where color = 0 for Red, 1 for Black

    Args:
        pos: Position (0-89)
        piece: Piece value (positive=Red, negative=Black)

    Returns:
        Feature index (0-1259)
    """
    piece_type = abs(piece) - 1  # 0-6
    color = 0 if is_red(piece) else 1  # 0=Red, 1=Black
    return pos * 14 + piece_type * 2 + color


def extract_features(board: np.ndarray) -> np.ndarray:
    """
    Extract feature vector from board state.

    Creates a sparse binary feature vector where each feature
    indicates the presence of a (piece, position) combination.

    Args:
        board: 10×9 board array

    Returns:
        Feature vector of shape (1260,) with binary values
    """
    features = np.zeros(FEATURE_SIZE, dtype=np.float32)

    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row, col]

            if piece != EMPTY:
                pos = coords_to_pos(row, col)
                idx = get_feature_index(pos, piece)
                features[idx] = 1.0

    return features


def extract_features_sparse(board: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features in sparse format (indices, values).

    Returns only the active features (where pieces are present).

    Args:
        board: 10×9 board array

    Returns:
        Tuple of (indices, values) arrays
    """
    indices = []
    values = []

    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row, col]

            if piece != EMPTY:
                pos = coords_to_pos(row, col)
                idx = get_feature_index(pos, piece)
                indices.append(idx)
                values.append(1.0)

    return np.array(indices, dtype=np.int32), np.array(values, dtype=np.float32)


def update_features_incremental(
    features: np.ndarray,
    board: np.ndarray,
    move: int,
    captured_piece: int
) -> np.ndarray:
    """
    Update features incrementally after a move.

    Instead of recomputing all features, only update the features
    for the moving piece and captured piece.

    Args:
        features: Current feature vector
        board: Board state before the move
        move: Move being made
        captured_piece: Piece being captured (EMPTY if none)

    Returns:
        Updated feature vector
    """
    from ..movegen.move_generator import decode_move, make_move

    new_features = features.copy()
    from_pos, to_pos = decode_move(move)

    # Remove piece from old position
    from_row, from_col = pos_to_coords(from_pos)
    piece = board[from_row, from_col]
    old_idx = get_feature_index(from_pos, piece)
    new_features[old_idx] = 0.0

    # Add piece to new position
    to_row, to_col = pos_to_coords(to_pos)
    new_idx = get_feature_index(to_pos, piece)
    new_features[new_idx] = 1.0

    # Remove captured piece
    if captured_piece != EMPTY:
        captured_idx = get_feature_index(to_pos, captured_piece)
        new_features[captured_idx] = 0.0

    return new_features


def extract_features_halfkp(board: np.ndarray, perspective: int) -> np.ndarray:
    """
    Extract HalfKP features (King-Piece).

    HalfKP uses (king_position, piece_position, piece_type) features,
    which captures king-piece interactions.

    Feature space: 90 (king positions) × 90 (piece positions) × 7 (piece types) × 2 (colors)
    Simplified: We use a smaller subset for efficiency

    Args:
        board: 10×9 board array
        perspective: Side to evaluate from (1=Red, -1=Black)

    Returns:
        Feature vector
    """
    # Find king position for the perspective
    from ..board.board_representation import RED_GENERAL, BLACK_GENERAL

    king_piece = RED_GENERAL if perspective == 1 else BLACK_GENERAL
    king_pos = None

    for row in range(ROWS):
        for col in range(COLS):
            if board[row, col] == king_piece:
                king_pos = coords_to_pos(row, col)
                break

    if king_pos is None:
        # King not found (should not happen in valid games)
        return extract_features(board)

    # For HalfKP, we use features that combine king position with piece positions
    # Simplified version: offset piece features by king position
    features = np.zeros(FEATURE_SIZE, dtype=np.float32)

    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row, col]

            if piece != EMPTY:
                pos = coords_to_pos(row, col)
                # Combine king position with piece position
                # This is a simplification; true HalfKP would have separate feature space
                idx = (king_pos * NUM_SQUARES + pos) % FEATURE_SIZE
                features[idx] = 1.0

    return features


def batch_extract_features(boards: np.ndarray) -> np.ndarray:
    """
    Extract features from multiple boards efficiently.

    Args:
        boards: Array of shape (batch_size, 10, 9)

    Returns:
        Feature array of shape (batch_size, 1260)
    """
    batch_size = boards.shape[0]
    features = np.zeros((batch_size, FEATURE_SIZE), dtype=np.float32)

    for i in range(batch_size):
        features[i] = extract_features(boards[i])

    return features


def get_feature_importance(features: np.ndarray) -> np.ndarray:
    """
    Get feature importance scores (for analysis/debugging).

    Args:
        features: Feature vector

    Returns:
        Array of (index, value) tuples sorted by value
    """
    indexed = [(i, v) for i, v in enumerate(features) if v > 0]
    return sorted(indexed, key=lambda x: x[1], reverse=True)


def feature_to_piece_pos(idx: int) -> Tuple[int, int]:
    """
    Convert feature index back to (piece_type, color, position).

    Args:
        idx: Feature index (0-1259)

    Returns:
        Tuple of (position, piece_type, color)
    """
    pos = idx // 14
    remainder = idx % 14
    piece_type = (remainder // 2) + 1  # 1-7
    color = remainder % 2  # 0=Red, 1=Black
    return pos, piece_type, 0 if color == 0 else -1
