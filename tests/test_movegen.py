"""
Tests for move generation.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.board.board_representation import create_initial_board
from src.movegen.move_generator import (
    encode_move, decode_move, parse_move, format_move,
    generate_moves, make_move, is_capture
)


def test_move_encoding():
    """Test move encoding/decoding."""
    # Test encode/decode roundtrip
    from_pos = 70
    to_pos = 62
    move = encode_move(from_pos, to_pos)
    decoded_from, decoded_to = decode_move(move)

    assert decoded_from == from_pos
    assert decoded_to == to_pos


def test_move_string_formatting():
    """Test move string parsing/formatting."""
    move_str = "7062"
    move = parse_move(move_str)
    formatted = format_move(move)

    assert formatted == move_str


def test_generate_initial_moves():
    """Test move generation from initial position."""
    board = create_initial_board()

    # Red to move (color = 1)
    moves = generate_moves(board, 1)

    # Should have some legal moves from initial position
    assert len(moves) > 0

    # All moves should be legal format
    for move in moves:
        from_pos, to_pos = decode_move(move)
        assert 0 <= from_pos <= 89
        assert 0 <= to_pos <= 89


def test_make_move():
    """Test making a move."""
    board = create_initial_board()

    # Get initial piece count
    initial_count = np.sum(board != 0)

    # Make a move (should be legal from initial position)
    moves = generate_moves(board, 1)
    if moves:
        move = moves[0]
        new_board = make_move(board, move)

        # Piece count should stay the same (non-capture)
        assert np.sum(new_board != 0) == initial_count

        # The from position should be empty
        from_pos, to_pos = decode_move(move)
        from_row, from_col = divmod(from_pos, 9)
        assert new_board[from_row, from_col] == 0


def test_capture_detection():
    """Test capture detection."""
    board = create_initial_board()

    # From initial position, no captures available immediately
    moves = generate_moves(board, 1)

    for move in moves:
        # Most initial moves should not be captures
        # (this is a weak test, but checks the function runs)
        is_cap = is_capture(board, move)
        assert isinstance(is_cap, bool)
