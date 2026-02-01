"""
Board Visualization Module for Chinese Chess

Provides visual representation of the board state for debugging.
"""

import numpy as np
from typing import Optional

from .board_representation import (
    ROWS, COLS, PIECE_NAMES,
    is_red, is_black, get_piece_type
)


def print_board(board: np.ndarray, unicode: bool = True) -> None:
    """
    Print the board to console.

    Args:
        board: 10×9 NumPy array
        unicode: Whether to use Unicode characters for pieces
    """
    # Print column numbers
    print("   " + " ".join(str(i) for i in range(COLS)))

    for row in range(ROWS):
        # Print row number
        print(f"{row}  ", end="")

        for col in range(COLS):
            piece = board[row, col]

            if piece == 0:
                # Check if we're in the palace or river
                if _is_palace(row, col):
                    print("＋", end=" ")
                elif _is_river(row):
                    print("～", end=" ")
                else:
                    print("·", end=" ")
            else:
                if unicode:
                    piece_char = PIECE_NAMES.get(piece, "?")
                else:
                    # ASCII fallback
                    piece_type = get_piece_type(piece)
                    color = "R" if is_red(piece) else "B"
                    piece_char = f"{color}{piece_type}"

                print(piece_char, end=" ")

        print()  # Newline after each row

    print()  # Extra newline at the end


def _is_palace(row: int, col: int) -> bool:
    """Check if position is in the palace (3×3 area at top and bottom center)."""
    # Black palace (top)
    if row <= 2 and 3 <= col <= 5:
        return True
    # Red palace (bottom)
    if row >= 7 and 3 <= col <= 5:
        return True
    return False


def _is_river(row: int) -> bool:
    """Check if position is on the river (between row 4 and 5)."""
    return row == 4 or row == 5


def board_to_str(board: np.ndarray, unicode: bool = True) -> str:
    """
    Convert board to string representation.

    Args:
        board: 10×9 NumPy array
        unicode: Whether to use Unicode characters

    Returns:
        String representation of the board
    """
    lines = []

    # Header
    lines.append("   " + " ".join(str(i) for i in range(COLS)))

    for row in range(ROWS):
        line = f"{row}  "
        for col in range(COLS):
            piece = board[row, col]

            if piece == 0:
                if _is_palace(row, col):
                    line += "＋ "
                elif _is_river(row):
                    line += "～ "
                else:
                    line += "· "
            else:
                if unicode:
                    piece_char = PIECE_NAMES.get(piece, "?")
                else:
                    piece_type = get_piece_type(piece)
                    color = "R" if is_red(piece) else "B"
                    piece_char = f"{color}{piece_type}"
                line += f"{piece_char} "

        lines.append(line)

    return "\n".join(lines)


def board_to_ascii(board: np.ndarray) -> str:
    """
    Convert board to ASCII-only representation.

    Uses letters for pieces: K=King, A=Advisor, E=Elephant, H=Horse,
    R=Rook, C=Cannon, S=Soldier. Prefix with R for Red, B for Black.
    """
    lines = []

    # Header
    lines.append("   " + " ".join(str(i) for i in range(COLS)))

    piece_codes = {
        0: "  ",
        1: "RK", 2: "RA", 3: "RE", 4: "RH", 5: "RR", 6: "RC", 7: "RS",
        -1: "BK", -2: "BA", -3: "BE", -4: "BH", -5: "BR", -6: "BC", -7: "BS",
    }

    for row in range(ROWS):
        line = f"{row}  "
        for col in range(COLS):
            piece = board[row, col]
            piece_char = piece_codes.get(piece, "??")
            line += f"{piece_char} "
        lines.append(line)

    return "\n".join(lines)


def print_move(board_before: np.ndarray, board_after: np.ndarray,
               from_pos: int, to_pos: int) -> None:
    """
    Print a move with before and after board states.

    Args:
        board_before: Board state before move
        board_after: Board state after move
        from_pos: Starting position (0-89)
        to_pos: Ending position (0-89)
    """
    print("=" * 50)
    print(f"MOVE: {from_pos:02d} → {to_pos:02d}")
    print("=" * 50)
    print("BEFORE:")
    print_board(board_before)
    print("-" * 50)
    print("AFTER:")
    print_board(board_after)
    print("=" * 50)


def print_board_info(board: np.ndarray) -> None:
    """
    Print detailed information about the board state.

    Args:
        board: 10×9 NumPy array
    """
    from .board_representation import (
        RED_GENERAL, RED_ADVISOR, RED_ELEPHANT, RED_HORSE,
        RED_ROOK, RED_CANNON, RED_SOLDIER
    )

    print("=" * 50)
    print("BOARD ANALYSIS")
    print("=" * 50)

    # Count pieces
    red_pieces = {}
    black_pieces = {}

    for row in range(10):
        for col in range(9):
            piece = board[row, col]
            if piece > 0:
                piece_type = piece
                red_pieces[piece_type] = red_pieces.get(piece_type, 0) + 1
            elif piece < 0:
                piece_type = -piece
                black_pieces[piece_type] = black_pieces.get(piece_type, 0) + 1

    print("\nRED PIECES:")
    piece_names = {1: "General", 2: "Advisor", 3: "Elephant",
                   4: "Horse", 5: "Rook", 6: "Cannon", 7: "Soldier"}
    for ptype in sorted(red_pieces.keys()):
        count = red_pieces[ptype]
        name = piece_names.get(ptype, f"Piece{ptype}")
        print(f"  {name}: {count}")

    print("\nBLACK PIECES:")
    for ptype in sorted(black_pieces.keys()):
        count = black_pieces[ptype]
        name = piece_names.get(ptype, f"Piece{ptype}")
        print(f"  {name}: {count}")

    # Material count
    material_values = {1: 10000, 2: 20, 3: 30, 4: 40, 5: 90, 6: 45, 7: 10}

    red_material = sum(red_pieces.get(p, 0) * material_values[p]
                       for p in red_pieces)
    black_material = sum(black_pieces.get(p, 0) * material_values[p]
                         for p in black_pieces)

    print(f"\nMATERIAL:")
    print(f"  Red: {red_material}")
    print(f"  Black: {black_material}")
    print(f"  Difference: {red_material - black_material}")
    print("=" * 50)


def visualize_position(board: np.ndarray, highlight_positions: Optional[list] = None) -> str:
    """
    Create visual board with optional position highlighting.

    Args:
        board: 10×9 NumPy array
        highlight_positions: List of positions (0-89) to highlight with *

    Returns:
        String representation with highlighted positions
    """
    if highlight_positions is None:
        highlight_positions = []

    highlight_set = set(highlight_positions)
    lines = []

    lines.append("   " + " ".join(str(i) for i in range(COLS)))

    for row in range(ROWS):
        line = f"{row}  "
        for col in range(COLS):
            pos = row * COLS + col
            piece = board[row, col]

            if piece == 0:
                if pos in highlight_set:
                    line += "* "
                else:
                    line += "· "
            else:
                piece_char = PIECE_NAMES.get(piece, "?")
                if pos in highlight_set:
                    line += f"{piece_char}*"
                else:
                    line += f"{piece_char} "
                line += " " if len(piece_char) == 1 else ""

        lines.append(line)

    return "\n".join(lines)
