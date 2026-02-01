"""
Move Validation Module for Chinese Chess

Checks move legality including:
- King safety (cannot move into check)
- General facing rule (generals cannot face each other directly)
"""

import numpy as np
from typing import List, Optional

from .move_generator import (
    Move, encode_move, decode_move, make_move,
    generate_moves, is_capture
)
from ..board.board_representation import (
    ROWS, COLS, EMPTY,
    RED_GENERAL, BLACK_GENERAL,
    is_red, is_black,
    pos_to_coords, coords_to_pos
)


def is_legal(board: np.ndarray, move: Move, color: int) -> bool:
    """
    Check if a move is legal (doesn't leave king in check).

    Args:
        board: Current board state
        move: Move to check
        color: Side making the move (1=Red, -1=Black)

    Returns:
        True if move is legal, False otherwise
    """
    # Make the move
    new_board = make_move(board, move)

    # Find general's position
    general_pos = find_general(board, color)
    if general_pos is None:
        return False  # Should not happen

    # Check if general is under attack
    return not is_square_attacked(new_board, general_pos, -color)


def find_general(board: np.ndarray, color: int) -> Optional[int]:
    """
    Find the position of the general/king.

    Args:
        board: Board state
        color: Color to search for (1=Red, -1=Black)

    Returns:
        Position of general (0-89) or None if not found
    """
    general_piece = RED_GENERAL if color == 1 else BLACK_GENERAL

    for row in range(ROWS):
        for col in range(COLS):
            if board[row, col] == general_piece:
                return coords_to_pos(row, col)

    return None


def is_square_attacked(board: np.ndarray, square: int, by_color: int) -> bool:
    """
    Check if a square is attacked by pieces of the given color.

    Args:
        board: Board state
        square: Square to check (0-89)
        by_color: Color of attacking pieces (1=Red, -1=Black)

    Returns:
        True if square is attacked
    """
    row, col = pos_to_coords(square)

    # Check for rook/rook-style attacks (rook, cannon, general in palace)
    if is_attacked_by_rook(board, row, col, by_color):
        return True

    # Check for cannon attacks
    if is_attacked_by_cannon(board, row, col, by_color):
        return True

    # Check for horse attacks
    if is_attacked_by_horse(board, row, col, by_color):
        return True

    # Check for elephant attacks (elephants don't attack, but we check for blocking)
    # Elephants don't attack, skip

    # Check for advisor attacks
    if is_attacked_by_advisor(board, row, col, by_color):
        return True

    # Check for general attacks (including flying general)
    if is_attacked_by_general(board, row, col, by_color):
        return True

    # Check for soldier attacks
    if is_attacked_by_soldier(board, row, col, by_color):
        return True

    return False


def is_attacked_by_rook(board: np.ndarray, row: int, col: int, by_color: int) -> bool:
    """Check if square is attacked by a rook."""
    from .move_generator import generate_rook_moves

    # For each rook of the attacking color, check if it can reach the square
    rook_piece = 5 if by_color == 1 else -5

    for r in range(ROWS):
        for c in range(COLS):
            if board[r, c] == rook_piece:
                pos = coords_to_pos(r, c)
                moves = generate_rook_moves(board, pos, board[r, c])
                _, to_pos = zip(*[decode_move(m) for m in moves])
                if coords_to_pos(row, col) in to_pos:
                    return True

    return False


def is_attacked_by_cannon(board: np.ndarray, row: int, col: int, by_color: int) -> bool:
    """Check if square is attacked by a cannon."""
    from .move_generator import generate_cannon_moves

    cannon_piece = 6 if by_color == 1 else -6

    for r in range(ROWS):
        for c in range(COLS):
            if board[r, c] == cannon_piece:
                pos = coords_to_pos(r, c)
                moves = generate_cannon_moves(board, pos, board[r, c])
                _, to_pos = zip(*[decode_move(m) for m in moves])
                if coords_to_pos(row, col) in to_pos:
                    return True

    return False


def is_attacked_by_horse(board: np.ndarray, row: int, col: int, by_color: int) -> bool:
    """Check if square is attacked by a horse."""
    # Reverse horse move checking
    # A horse at (r, c) attacks (row, col) if:
    # (r, c) can reach (row, col) via a horse move and is not blocked

    horse_piece = 4 if by_color == 1 else -4

    # Horse positions that could attack (row, col)
    horse_positions = [
        (row - 2, col - 1, row - 1, col),
        (row - 2, col + 1, row - 1, col),
        (row + 2, col - 1, row + 1, col),
        (row + 2, col + 1, row + 1, col),
        (row - 1, col - 2, row, col - 1),
        (row + 1, col - 2, row, col - 1),
        (row - 1, col + 2, row, col + 1),
        (row + 1, col + 2, row, col + 1),
    ]

    for hr, hc, block_r, block_c in horse_positions:
        if 0 <= hr < ROWS and 0 <= hc < COLS:
            if board[hr, hc] == horse_piece:
                # Check for blocking
                if 0 <= block_r < ROWS and 0 <= block_c < COLS:
                    if board[block_r, block_c] == EMPTY:
                        return True

    return False


def is_attacked_by_advisor(board: np.ndarray, row: int, col: int, by_color: int) -> bool:
    """Check if square is attacked by an advisor."""
    advisor_piece = 2 if by_color == 1 else -2

    # Advisor attacks diagonally one step
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ar, ac = row + dr, col + dc

        if 0 <= ar < ROWS and 0 <= ac < COLS:
            if board[ar, ac] == advisor_piece:
                return True

    return False


def is_attacked_by_general(board: np.ndarray, row: int, col: int, by_color: int) -> bool:
    """
    Check if square is attacked by a general (including flying general rule).

    Generals attack orthogonally one step in the palace.
    Also implements flying general: generals cannot face each other directly.
    """
    general_piece = 1 if by_color == 1 else -1

    # Check for adjacent general attacks
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        gr, gc = row + dr, col + dc

        if 0 <= gr < ROWS and 0 <= gc < COLS:
            if board[gr, gc] == general_piece:
                return True

    # Check for flying general
    if is_flying_general(board):
        return True

    return False


def is_flying_general(board: np.ndarray) -> bool:
    """
    Check if generals are facing each other directly (flying general rule).

    The flying general rule states that the two generals cannot face each other
    on the same file with no pieces between them.

    Returns:
        True if generals are flying (illegal position)
    """
    # Find both generals
    red_general_pos = None
    black_general_pos = None

    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row, col]
            if piece == RED_GENERAL:
                red_general_pos = (row, col)
            elif piece == BLACK_GENERAL:
                black_general_pos = (row, col)

    if red_general_pos is None or black_general_pos is None:
        return False

    # Check if they're on the same column
    if red_general_pos[1] != black_general_pos[1]:
        return False

    # Check if there are pieces between them
    col = red_general_pos[1]
    min_row = min(red_general_pos[0], black_general_pos[0])
    max_row = max(red_general_pos[0], black_general_pos[0])

    for row in range(min_row + 1, max_row):
        if board[row, col] != EMPTY:
            return False  # Piece between them, not flying

    return True  # No pieces between, generals are facing


def is_attacked_by_soldier(board: np.ndarray, row: int, col: int, by_color: int) -> bool:
    """Check if square is attacked by a soldier."""
    soldier_piece = 7 if by_color == 1 else -7

    # Red soldiers attack upward and sideways (after crossing river)
    if by_color == 1:
        # From below (attacking upward)
        if row + 1 < ROWS and board[row + 1, col] == soldier_piece:
            return True
        # From sides (only if soldier has crossed river, i.e., row <= 4)
        if row <= 4:
            if col - 1 >= 0 and board[row, col - 1] == soldier_piece:
                return True
            if col + 1 < COLS and board[row, col + 1] == soldier_piece:
                return True
    else:
        # Black soldiers attack downward and sideways
        # From above (attacking downward)
        if row - 1 >= 0 and board[row - 1, col] == soldier_piece:
            return True
        # From sides (only if soldier has crossed river, i.e., row >= 5)
        if row >= 5:
            if col - 1 >= 0 and board[row, col - 1] == soldier_piece:
                return True
            if col + 1 < COLS and board[row, col + 1] == soldier_piece:
                return True

    return False


def generate_legal_moves(board: np.ndarray, color: int) -> List[Move]:
    """
    Generate all legal moves for the given color.

    Filters pseudo-legal moves to only those that don't leave the king in check.

    Args:
        board: Current board state
        color: Side to generate moves for (1=Red, -1=Black)

    Returns:
        List of legal moves
    """
    pseudo_moves = generate_moves(board, color)
    legal_moves = []

    for move in pseudo_moves:
        if is_legal(board, move, color):
            legal_moves.append(move)

    return legal_moves


def is_check(board: np.ndarray, color: int) -> bool:
    """
    Check if the given side's general is in check.

    Args:
        board: Board state
        color: Side to check (1=Red, -1=Black)

    Returns:
        True if in check
    """
    general_pos = find_general(board, color)
    if general_pos is None:
        return False

    return is_square_attacked(board, general_pos, -color)


def is_checkmate(board: np.ndarray, color: int) -> bool:
    """
    Check if the given side is in checkmate.

    Args:
        board: Board state
        color: Side to check (1=Red, -1=Black)

    Returns:
        True if checkmate
    """
    if not is_check(board, color):
        return False

    legal_moves = generate_legal_moves(board, color)
    return len(legal_moves) == 0


def is_stalemate(board: np.ndarray, color: int) -> bool:
    """
    Check if the given side is in stalemate.

    Args:
        board: Board state
        color: Side to check (1=Red, -1=Black)

    Returns:
        True if stalemate
    """
    if is_check(board, color):
        return False

    legal_moves = generate_legal_moves(board, color)
    return len(legal_moves) == 0
