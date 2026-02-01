"""
Alpha-Beta search with iterative deepening.

Implements efficient game tree search with alpha-beta pruning.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from collections import defaultdict

from ..board.board_representation import ROWS, COLS, EMPTY, is_red
from ..movegen.move_generator import (
    Move, encode_move, decode_move,
    generate_moves, make_move, is_capture
)
from ..movegen.validators import is_legal, is_check
from ..evaluation.evaluator import Evaluator


# Search limits
MAX_DEPTH = 64
INF = 100000
MATE_SCORE = 100000


class SearchInfo:
    """Information about the current search."""

    def __init__(self):
        self.nodes = 0
        self.depth = 0
        self.score = 0
        self.best_move = None
        self.pv = []  # Principal variation
        self.time_ms = 0


def alphabeta(
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    evaluator: Evaluator,
    info: SearchInfo
) -> float:
    """
    Alpha-Beta search (negamax formulation).

    Args:
        board: Current board state
        depth: Remaining search depth
        alpha: Alpha value (lower bound)
        beta: Beta value (upper bound)
        maximizing: True if maximizing player
        evaluator: Position evaluator
        info: Search info struct

    Returns:
        Evaluation score
    """
    info.nodes += 1

    # Terminal conditions
    if depth == 0:
        return evaluator.evaluate(board, perspective=1 if maximizing else -1)

    # Generate legal moves
    color = 1 if maximizing else -1
    moves = generate_moves(board, color)
    legal_moves = [m for m in moves if is_legal(board, m, color)]

    if not legal_moves:
        # No legal moves - checkmate or stalemate
        if is_check(board, color):
            # Checkmate
            return -MATE_SCORE + (MAX_DEPTH - depth)
        else:
            # Stalemate
            return 0

    # Search moves
    best_score = -INF

    for move in legal_moves:
        new_board = make_move(board, move)

        # Negamax: recursively search with negated bounds and switched player
        score = -alphabeta(
            new_board,
            depth - 1,
            -beta,
            -alpha,
            not maximizing,
            evaluator,
            info
        )

        if score > best_score:
            best_score = score

        if score > alpha:
            alpha = score

        if alpha >= beta:
            break  # Beta cutoff

    return best_score


def alphabeta_root(
    board: np.ndarray,
    depth: int,
    evaluator: Evaluator,
    info: SearchInfo
) -> Tuple[float, Optional[Move]]:
    """
    Alpha-Beta search from root.

    Args:
        board: Current board state
        depth: Search depth
        evaluator: Position evaluator
        info: Search info struct

    Returns:
        Tuple of (score, best_move)
    """
    alpha = -INF
    beta = INF
    best_score = -INF
    best_move = None

    # Determine side to move
    # For simplicity, assume Red to move
    color = 1
    maximizing = True

    # Generate legal moves
    moves = generate_moves(board, color)
    legal_moves = [m for m in moves if is_legal(board, m, color)]

    if not legal_moves:
        # No legal moves
        if is_check(board, color):
            return -MATE_SCORE, None  # Checkmate
        else:
            return 0, None  # Stalemate

    # Order moves (simple version: captures first)
    legal_moves = order_moves(legal_moves, board)

    # Search each move
    for move in legal_moves:
        new_board = make_move(board, move)

        score = -alphabeta(
            new_board,
            depth - 1,
            -beta,
            -alpha,
            not maximizing,
            evaluator,
            info
        )

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

        # Update info
        if info.nodes % 1000 == 0:
            info.score = best_score

    return best_score, best_move


def order_moves(moves: List[Move], board: np.ndarray) -> List[Move]:
    """
    Order moves for better pruning.

    Simple ordering: captures first, then others.

    Args:
        moves: List of moves
        board: Current board state

    Returns:
        Ordered list of moves
    """
    captures = []
    non_captures = []

    for move in moves:
        if is_capture(board, move):
            captures.append(move)
        else:
            non_captures.append(move)

    # Sort captures by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    captures.sort(key=lambda m: _capture_value(m, board), reverse=True)

    return captures + non_captures


def _capture_value(move: Move, board: np.ndarray) -> int:
    """Get approximate capture value for move ordering."""
    from_pos, to_pos = decode_move(move)
    to_row, to_col = divmod(to_pos, 9)
    target = board[to_row, to_col]

    if target == EMPTY:
        return 0

    # Piece values
    values = {1: 10000, 2: 20, 3: 30, 4: 40, 5: 90, 6: 45, 7: 10}

    return values.get(abs(target), 0)


def iterative_deepening(
    board: np.ndarray,
    max_depth: int,
    evaluator: Evaluator,
    info: SearchInfo
) -> Tuple[float, Optional[Move]]:
    """
    Iterative deepening search.

    Searches with increasing depth, using the best move from each iteration
    to improve move ordering in the next.

    Args:
        board: Current board state
        max_depth: Maximum search depth
        evaluator: Position evaluator
        info: Search info struct

    Returns:
        Tuple of (score, best_move)
    """
    best_move = None
    best_score = 0

    for depth in range(1, max_depth + 1):
        info.depth = depth
        score, move = alphabeta_root(board, depth, evaluator, info)

        if move is not None:
            best_move = move
            best_score = score

        # Check for mate found
        if abs(score) >= MATE_SCORE - MAX_DEPTH:
            break

    info.score = best_score
    info.best_move = best_move

    return best_score, best_move


def search(
    board: np.ndarray,
    depth: int,
    evaluator: Evaluator
) -> Tuple[int, Optional[int]]:
    """
    Convenience function for searching a position.

    Args:
        board: Current board state
        depth: Search depth
        evaluator: Position evaluator

    Returns:
        Tuple of (score, best_move) where best_move is a position (0-89)
    """
    info = SearchInfo()
    score, move = iterative_deepening(board, depth, evaluator, info)

    if move is None:
        return score, None

    from_pos, to_pos = decode_move(move)
    return score, to_pos
