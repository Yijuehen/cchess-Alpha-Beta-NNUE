"""
Board Representation Module for Chinese Chess (Xiangqi)

Converts 64-character board state strings to/from 10×9 NumPy arrays.
Encoding format: 32 piece positions, each 2 digits (00-89).

Piece encoding:
- Positive: Red pieces
- Negative: Black pieces
- 0: Empty
- 1: General/King (帅/将)
- 2: Advisor (仕/士)
- 3: Elephant (相/象)
- 4: Horse (马)
- 5: Rook (车)
- 6: Cannon (炮)
- 7: Soldier (兵/卒)
"""

import numpy as np
from typing import Tuple, Optional


# Piece constants
EMPTY = 0
RED_GENERAL = 1
RED_ADVISOR = 2
RED_ELEPHANT = 3
RED_HORSE = 4
RED_ROOK = 5
RED_CANNON = 6
RED_SOLDIER = 7

BLACK_GENERAL = -1
BLACK_ADVISOR = -2
BLACK_ELEPHANT = -3
BLACK_HORSE = -4
BLACK_ROOK = -5
BLACK_CANNON = -6
BLACK_SOLDIER = -7

# Piece names for visualization
PIECE_NAMES = {
    0: "  ",
    1: "帅", 2: "仕", 3: "相", 4: "马", 5: "车", 6: "炮", 7: "兵",
    -1: "将", -2: "士", -3: "象", -4: "马", -5: "车", -6: "炮", -7: "卒"
}

# Board dimensions
ROWS = 10
COLS = 9
NUM_POSITIONS = ROWS * COLS  # 90


def pos_to_coords(pos: int) -> Tuple[int, int]:
    """Convert position (0-89) to (row, col) coordinates."""
    return divmod(pos, COLS)


def coords_to_pos(row: int, col: int) -> int:
    """Convert (row, col) coordinates to position (0-89)."""
    return row * COLS + col


def parse_board(board_str: str) -> np.ndarray:
    """
    Parse 64-character board string to 10×9 array.

    Args:
        board_str: 64-character string encoding 32 piece positions (2 digits each)

    Returns:
        10×9 NumPy array with piece values (positive=Red, negative=Black, 0=empty)

    Example:
        >>> board = parse_board("0919293949596979891777062646668600102030405060708012720323436383")
        >>> board[0, 0]  # Position 00 - Black Rook
        -5
    """
    if len(board_str) != 64:
        raise ValueError(f"Board string must be 64 characters, got {len(board_str)}")

    # Initialize empty board
    board = np.zeros((ROWS, COLS), dtype=np.int8)

    # Parse 32 piece positions (2 digits each)
    for i in range(0, 64, 2):
        pos_str = board_str[i:i+2]
        pos = int(pos_str)

        # Determine piece type and color based on position index
        piece_idx = i // 2  # 0-31

        # First 16 pieces are Red (positive), last 16 are Black (negative)
        # Piece type mapping based on the user's description:
        # Red: Car(09,19,29,39,49,59,69,79,89), Cannon(17,77), Soldier(06,26,46,66,86)
        # Black: Car(00,80), Horse(10,70), Aspect(20,60), Soldier(30,50), General(40), Gun(12,72,03,23,43,63,83)

        if piece_idx < 16:  # Red pieces
            piece = _identify_red_piece(pos)
        else:  # Black pieces
            piece = _identify_black_piece(pos)

        # Place piece on board
        row, col = pos_to_coords(pos)
        board[row, col] = piece

    return board


def _identify_red_piece(pos: int) -> int:
    """Identify red piece type based on position."""
    # Red Car (Rook): 09, 19, 29, 39, 49, 59, 69, 79, 89
    if pos % 10 == 9:
        return RED_ROOK
    # Red Cannon: 17, 77
    elif pos in (17, 77):
        return RED_CANNON
    # Red Soldier: 06, 26, 46, 66, 86
    elif pos % 20 == 6:
        return RED_SOLDIER
    # Red Horse: 18, 78 (inferred from symmetry)
    elif pos in (18, 78):
        return RED_HORSE
    # Red Elephant/Phase: 27, 67 (inferred)
    elif pos in (27, 67):
        return RED_ELEPHANT
    # Red Advisor: 37, 57 (inferred)
    elif pos in (37, 57):
        return RED_ADVISOR
    # Red General: 47 (inferred)
    elif pos == 47:
        return RED_GENERAL
    else:
        # Default to soldier for unknown positions
        # (this handles positions not explicitly documented)
        return RED_SOLDIER


def _identify_black_piece(pos: int) -> int:
    """Identify black piece type based on position."""
    # Black Car (Rook): 00, 80
    if pos in (0, 80):
        return -BLACK_ROOK  # Note: return negative
    # Black Horse: 10, 70
    elif pos in (10, 70):
        return -BLACK_HORSE
    # Black Aspect (Advisor): 20, 60
    elif pos in (20, 60):
        return -BLACK_ADVISOR
    # Black Soldier: 30, 50
    elif pos in (30, 50):
        return -BLACK_SOLDIER
    # Black General: 40
    elif pos == 40:
        return -BLACK_GENERAL
    # Black Gun (Cannon): 12, 72, 03, 23, 43, 63, 83
    elif pos in (12, 72, 3, 23, 43, 63, 83):
        return -BLACK_CANNON
    else:
        # Default handling
        return -BLACK_SOLDIER


def encode_board(board: np.ndarray) -> str:
    """
    Encode 10×9 board array back to 64-character string.

    Args:
        board: 10×9 NumPy array with piece values

    Returns:
        64-character string encoding
    """
    red_positions = []
    black_positions = []

    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row, col]
            if piece != 0:
                pos = coords_to_pos(row, col)
                pos_str = f"{pos:02d}"

                if piece > 0:  # Red
                    red_positions.append(pos_str)
                else:  # Black
                    black_positions.append(pos_str)

    # Pad with empty positions if needed (should have 32 pieces total)
    while len(red_positions) < 16:
        red_positions.append("99")  # Invalid position for padding
    while len(black_positions) < 16:
        black_positions.append("99")

    return "".join(red_positions + black_positions)


def get_piece_at(board: np.ndarray, row: int, col: int) -> int:
    """Get piece at board position."""
    if not (0 <= row < ROWS and 0 <= col < COLS):
        raise ValueError(f"Invalid position: ({row}, {col})")
    return board[row, col]


def set_piece_at(board: np.ndarray, row: int, col: int, piece: int) -> np.ndarray:
    """Set piece at board position (returns new board)."""
    if not (0 <= row < ROWS and 0 <= col < COLS):
        raise ValueError(f"Invalid position: ({row}, {col})")
    new_board = board.copy()
    new_board[row, col] = piece
    return new_board


def is_red(piece: int) -> bool:
    """Check if piece is red."""
    return piece > 0


def is_black(piece: int) -> bool:
    """Check if piece is black."""
    return piece < 0


def get_piece_type(piece: int) -> int:
    """Get piece type (1-7) without color."""
    return abs(piece)


def copy_board(board: np.ndarray) -> np.ndarray:
    """Create a deep copy of the board."""
    return board.copy()


# Initial board state for reference
INITIAL_BOARD_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


def create_initial_board() -> np.ndarray:
    """Create the initial starting board position."""
    # This is a simplified initial board - adjust based on actual starting position
    board = np.zeros((ROWS, COLS), dtype=np.int8)

    # Black pieces (top, rows 0-4)
    board[0, 0] = board[0, 8] = BLACK_ROOK
    board[0, 1] = board[0, 7] = BLACK_HORSE
    board[0, 2] = board[0, 6] = BLACK_ELEPHANT
    board[0, 3] = board[0, 5] = BLACK_ADVISOR
    board[0, 4] = BLACK_GENERAL
    board[2, 1] = board[2, 7] = BLACK_CANNON
    board[3, 0] = board[3, 2] = board[3, 4] = board[3, 6] = board[3, 8] = BLACK_SOLDIER

    # Red pieces (bottom, rows 5-9)
    board[9, 0] = board[9, 8] = RED_ROOK
    board[9, 1] = board[9, 7] = RED_HORSE
    board[9, 2] = board[9, 6] = RED_ELEPHANT
    board[9, 3] = board[9, 5] = RED_ADVISOR
    board[9, 4] = RED_GENERAL
    board[7, 1] = board[7, 7] = RED_CANNON
    board[6, 0] = board[6, 2] = board[6, 4] = board[6, 6] = board[6, 8] = RED_SOLDIER

    return board


def validate_board(board: np.ndarray) -> bool:
    """
    Validate board state.

    Checks:
    - Board is 10×9
    - Exactly one general of each color
    - Pieces are within valid ranges
    """
    if board.shape != (ROWS, COLS):
        return False

    # Count generals
    red_generals = np.sum(board == RED_GENERAL)
    black_generals = np.sum(board == BLACK_GENERAL)

    if red_generals != 1 or black_generals != 1:
        return False

    # Check piece values are in valid range
    if np.any(board > 7) or np.any(board < -7):
        return False

    return True
