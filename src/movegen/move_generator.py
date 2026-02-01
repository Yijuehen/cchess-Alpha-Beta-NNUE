"""
Move Generation Module for Chinese Chess

Generates pseudo-legal moves for all piece types following Chinese Chess rules:
- Palace restrictions (General, Advisor)
- River crossing (Elephant, Soldier)
- Blocking rules (Horse leg, Elephant eye)
- Jump captures (Cannon)
"""

import numpy as np
from typing import List, Tuple

from ..board.board_representation import (
    ROWS, COLS,
    EMPTY,
    RED_GENERAL, RED_ADVISOR, RED_ELEPHANT, RED_HORSE, RED_ROOK, RED_CANNON, RED_SOLDIER,
    BLACK_GENERAL, BLACK_ADVISOR, BLACK_ELEPHANT, BLACK_HORSE, BLACK_ROOK, BLACK_CANNON, BLACK_SOLDIER,
    is_red, is_black, get_piece_type,
    pos_to_coords, coords_to_pos
)


# Move encoding: 16-bit integer
# Upper 8 bits: from_pos (0-89)
# Lower 8 bits: to_pos (0-89)
Move = int


def encode_move(from_pos: int, to_pos: int) -> Move:
    """Encode move as 16-bit integer."""
    return (from_pos << 8) | to_pos


def decode_move(move: Move) -> Tuple[int, int]:
    """Decode move to (from_pos, to_pos)."""
    from_pos = (move >> 8) & 0xFF
    to_pos = move & 0xFF
    return from_pos, to_pos


def parse_move(move_str: str) -> Move:
    """Parse 4-digit move string to internal move encoding."""
    if len(move_str) != 4:
        raise ValueError(f"Move string must be 4 digits, got {move_str}")
    from_pos = int(move_str[:2])
    to_pos = int(move_str[2:])
    return encode_move(from_pos, to_pos)


def format_move(move: Move) -> str:
    """Format move as 4-digit string."""
    from_pos, to_pos = decode_move(move)
    return f"{from_pos:02d}{to_pos:02d}"


def generate_moves(board: np.ndarray, color: int) -> List[Move]:
    """
    Generate all pseudo-legal moves for the given color.

    Args:
        board: 10×9 board array
        color: 1 for Red, -1 for Black

    Returns:
        List of moves (encoded as 16-bit integers)
    """
    moves = []

    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row, col]

            # Skip empty squares and opponent's pieces
            if piece == EMPTY or (color == 1 and piece < 0) or (color == -1 and piece > 0):
                continue

            pos = coords_to_pos(row, col)
            piece_type = abs(piece)

            # Generate moves based on piece type
            if piece_type == 1:  # General/King
                moves.extend(generate_general_moves(board, pos, piece))
            elif piece_type == 2:  # Advisor
                moves.extend(generate_advisor_moves(board, pos, piece))
            elif piece_type == 3:  # Elephant
                moves.extend(generate_elephant_moves(board, pos, piece))
            elif piece_type == 4:  # Horse
                moves.extend(generate_horse_moves(board, pos, piece))
            elif piece_type == 5:  # Rook
                moves.extend(generate_rook_moves(board, pos, piece))
            elif piece_type == 6:  # Cannon
                moves.extend(generate_cannon_moves(board, pos, piece))
            elif piece_type == 7:  # Soldier
                moves.extend(generate_soldier_moves(board, pos, piece))

    return moves


def generate_rook_moves(board: np.ndarray, pos: int, piece: int) -> List[Move]:
    """
    Generate rook moves (orthogonal, any distance).

    Rooks move horizontally and vertically until blocked.
    """
    moves = []
    row, col = pos_to_coords(pos)

    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc

        while 0 <= new_row < ROWS and 0 <= new_col < COLS:
            target = board[new_row, new_col]

            if target == EMPTY:
                # Empty square, add move and continue
                moves.append(encode_move(pos, coords_to_pos(new_row, new_col)))
            elif is_red(piece) != is_red(target):
                # Capture opponent's piece
                moves.append(encode_move(pos, coords_to_pos(new_row, new_col)))
                break  # Can't move past captured piece
            else:
                # Blocked by own piece
                break

            new_row += dr
            new_col += dc

    return moves


def generate_cannon_moves(board: np.ndarray, pos: int, piece: int) -> List[Move]:
    """
    Generate cannon moves.

    Cannons move like rooks but capture by jumping over exactly one piece.
    """
    moves = []
    row, col = pos_to_coords(pos)

    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        jumped = False

        while 0 <= new_row < ROWS and 0 <= new_col < COLS:
            target = board[new_row, new_col]

            if not jumped:
                # Haven't jumped yet
                if target == EMPTY:
                    # Move to empty square
                    moves.append(encode_move(pos, coords_to_pos(new_row, new_col)))
                else:
                    # Found a piece to jump over
                    jumped = True
            else:
                # Already jumped, looking for capture
                if target != EMPTY:
                    if is_red(piece) != is_red(target):
                        # Capture opponent's piece
                        moves.append(encode_move(pos, coords_to_pos(new_row, new_col)))
                    break  # Can't jump over multiple pieces

            new_row += dr
            new_col += dc

    return moves


def generate_horse_moves(board: np.ndarray, pos: int, piece: int) -> List[Move]:
    """
    Generate horse moves (L-shape with blocking).

    Horse moves one step orthogonally then one step diagonally.
    Can be blocked if the first step is occupied ("hobbling the horse's leg").
    """
    moves = []
    row, col = pos_to_coords(pos)

    # Horse moves: (dr, dc) for first step, then (block_dr, block_dc) for blocking check
    # Final position is (row + dr + diag_dr, col + dc + diag_dc)
    horse_moves = [
        (-2, -1, -1, 0),  # Up-left
        (-2, 1, -1, 0),   # Up-right
        (2, -1, 1, 0),    # Down-left
        (2, 1, 1, 0),     # Down-right
        (-1, -2, 0, -1),  # Left-up
        (1, -2, 0, -1),   # Left-down
        (-1, 2, 0, 1),    # Right-up
        (1, 2, 0, 1),     # Right-down
    ]

    for dr, dc, block_dr, block_dc in horse_moves:
        new_row = row + dr
        new_col = col + dc
        block_row = row + block_dr
        block_col = col + block_dc

        # Check bounds
        if not (0 <= new_row < ROWS and 0 <= new_col < COLS):
            continue

        # Check for blocking piece ("horse leg")
        if board[block_row, block_col] != EMPTY:
            continue

        target = board[new_row, new_col]

        # Empty or capture opponent
        if target == EMPTY or is_red(piece) != is_red(target):
            moves.append(encode_move(pos, coords_to_pos(new_row, new_col)))

    return moves


def generate_elephant_moves(board: np.ndarray, pos: int, piece: int) -> List[Move]:
    """
    Generate elephant moves (diagonal 2 steps with blocking and river restriction).

    Elephants move exactly 2 steps diagonally.
    Can be blocked if the middle square is occupied ("blocking the elephant's eye").
    Cannot cross the river.
    """
    moves = []
    row, col = pos_to_coords(pos)

    # Elephant moves: 2 steps diagonal
    directions = [
        (-2, -2, -1, -1),  # Up-left
        (-2, 2, -1, 1),    # Up-right
        (2, -2, 1, -1),    # Down-left
        (2, 2, 1, 1),      # Down-right
    ]

    for dr, dc, eye_dr, eye_dc in directions:
        new_row = row + dr
        new_col = col + dc
        eye_row = row + eye_dr
        eye_col = col + eye_dc

        # Check bounds
        if not (0 <= new_row < ROWS and 0 <= new_col < COLS):
            continue

        # Red elephants cannot cross river (row 4-5 boundary)
        if is_red(piece) and new_row < 5:
            continue
        # Black elephants cannot cross river
        if is_black(piece) and new_row > 4:
            continue

        # Check for blocking piece ("elephant eye")
        if board[eye_row, eye_col] != EMPTY:
            continue

        target = board[new_row, new_col]

        # Empty or capture opponent
        if target == EMPTY or is_red(piece) != is_red(target):
            moves.append(encode_move(pos, coords_to_pos(new_row, new_col)))

    return moves


def generate_advisor_moves(board: np.ndarray, pos: int, piece: int) -> List[Move]:
    """
    Generate advisor moves (diagonal 1 step, palace restriction).

    Advisors move one step diagonally within the palace (3×3 area).
    """
    moves = []
    row, col = pos_to_coords(pos)

    # Diagonal directions
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dr, dc in directions:
        new_row = row + dr
        new_col = col + dc

        # Check bounds and palace restriction
        if not (0 <= new_row < ROWS and 0 <= new_col < COLS):
            continue

        # Palace restriction: columns 3-5
        if not (3 <= new_col <= 5):
            continue

        # Red palace: rows 7-9, Black palace: rows 0-2
        if is_red(piece) and not (7 <= new_row <= 9):
            continue
        if is_black(piece) and not (0 <= new_row <= 2):
            continue

        target = board[new_row, new_col]

        # Empty or capture opponent
        if target == EMPTY or is_red(piece) != is_red(target):
            moves.append(encode_move(pos, coords_to_pos(new_row, new_col)))

    return moves


def generate_general_moves(board: np.ndarray, pos: int, piece: int) -> List[Move]:
    """
    Generate general moves (orthogonal 1 step, palace restriction).

    Generals move one step orthogonally within the palace (3×3 area).
    """
    moves = []
    row, col = pos_to_coords(pos)

    # Orthogonal directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        new_row = row + dr
        new_col = col + dc

        # Check bounds and palace restriction
        if not (0 <= new_row < ROWS and 0 <= new_col < COLS):
            continue

        # Palace restriction: columns 3-5
        if not (3 <= new_col <= 5):
            continue

        # Red palace: rows 7-9, Black palace: rows 0-2
        if is_red(piece) and not (7 <= new_row <= 9):
            continue
        if is_black(piece) and not (0 <= new_row <= 2):
            continue

        target = board[new_row, new_col]

        # Empty or capture opponent
        if target == EMPTY or is_red(piece) != is_red(target):
            moves.append(encode_move(pos, coords_to_pos(new_row, new_col)))

    return moves


def generate_soldier_moves(board: np.ndarray, pos: int, piece: int) -> List[Move]:
    """
    Generate soldier moves (forward, then sideways after crossing river).

    Soldiers move one step forward.
    After crossing the river, they can also move sideways.
    Red soldiers cross from row 5 to 4, Black from row 4 to 5.
    """
    moves = []
    row, col = pos_to_coords(pos)

    if is_red(piece):
        # Red soldier moves up (decreasing row)
        # Forward
        new_row = row - 1
        if new_row >= 0:
            target = board[new_row, col]
            if target == EMPTY or is_red(target):
                moves.append(encode_move(pos, coords_to_pos(new_row, col)))

        # Sideways after crossing river (row <= 4 means crossed)
        if row <= 4:
            for dc in [-1, 1]:
                new_col = col + dc
                if 0 <= new_col < COLS:
                    target = board[row, new_col]
                    if target == EMPTY or is_red(target):
                        moves.append(encode_move(pos, coords_to_pos(row, new_col)))
    else:
        # Black soldier moves down (increasing row)
        # Forward
        new_row = row + 1
        if new_row < ROWS:
            target = board[new_row, col]
            if target == EMPTY or is_black(target):
                moves.append(encode_move(pos, coords_to_pos(new_row, col)))

        # Sideways after crossing river (row >= 5 means crossed)
        if row >= 5:
            for dc in [-1, 1]:
                new_col = col + dc
                if 0 <= new_col < COLS:
                    target = board[row, new_col]
                    if target == EMPTY or is_black(target):
                        moves.append(encode_move(pos, coords_to_pos(row, new_col)))

    return moves


def make_move(board: np.ndarray, move: Move) -> np.ndarray:
    """
    Apply a move to the board, returning a new board.

    Args:
        board: Current board state
        move: Move to make

    Returns:
        New board state after the move
    """
    new_board = board.copy()
    from_pos, to_pos = decode_move(move)

    from_row, from_col = pos_to_coords(from_pos)
    to_row, to_col = pos_to_coords(to_pos)

    # Move piece
    new_board[to_row, to_col] = new_board[from_row, from_col]
    new_board[from_row, from_col] = EMPTY

    return new_board


def unmake_move(board: np.ndarray, move: Move, captured_piece: int) -> np.ndarray:
    """
    Reverse a move on the board.

    Args:
        board: Board state after the move
        move: Move to reverse
        captured_piece: Piece that was captured (EMPTY if no capture)

    Returns:
        Board state before the move
    """
    new_board = board.copy()
    from_pos, to_pos = decode_move(move)

    from_row, from_col = pos_to_coords(from_pos)
    to_row, to_col = pos_to_coords(to_pos)

    # Move piece back
    new_board[from_row, from_col] = new_board[to_row, to_col]
    new_board[to_row, to_col] = captured_piece

    return new_board


def is_capture(board: np.ndarray, move: Move) -> bool:
    """Check if a move is a capture."""
    from_pos, to_pos = decode_move(move)
    to_row, to_col = pos_to_coords(to_pos)
    return board[to_row, to_col] != EMPTY
