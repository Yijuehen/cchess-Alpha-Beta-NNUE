"""
Integration tests for the complete engine.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.xiangqi_engine import create_engine
from src.board.board_representation import create_initial_board


def test_engine_creation():
    """Test engine can be created."""
    engine = create_engine()
    assert engine is not None
    assert engine.board is not None


def test_initial_position():
    """Test engine with initial position."""
    engine = create_engine()

    # Should have legal moves
    moves = engine.get_legal_moves()
    assert len(moves) > 0

    # Should not be in check
    assert not engine.is_check()
    assert not engine.is_checkmate()


def test_search():
    """Test basic search."""
    engine = create_engine()
    engine.set_search_depth(2)

    score, move = engine.search(depth=2)

    # Should return a move
    assert move is not None
    assert len(move) == 4

    # Score should be within reasonable bounds
    assert -100000 < score < 100000


def test_evaluate():
    """Test position evaluation."""
    engine = create_engine()

    score = engine.evaluate()

    # Initial position should be roughly balanced
    # (close to 0, but not necessarily exactly 0)
    assert -1000 < score < 1000


def test_make_move():
    """Test making moves."""
    engine = create_engine()

    # Get legal moves
    moves = engine.get_legal_moves()
    assert len(moves) > 0

    # Make first legal move
    success = engine.make_move(moves[0])
    assert success

    # Should now be Black's turn
    assert engine.side_to_move == -1


def test_analyze():
    """Test position analysis."""
    engine = create_engine()

    results = engine.analyze(depth=2)

    # Should return list of moves
    assert isinstance(results, list)
    assert len(results) > 0

    # Each result should be (move, score) tuple
    for move, score in results:
        assert isinstance(move, str)
        assert isinstance(score, int)


def test_print_board():
    """Test board printing (just check it doesn't crash)."""
    engine = create_engine()

    # These should not raise exceptions
    engine.print_board()
    board_str = engine.board_to_string()
    assert isinstance(board_str, str)


def test_search_info():
    """Test search info retrieval."""
    engine = create_engine()
    engine.search(depth=2)

    info = engine.get_search_info()

    assert 'nodes' in info
    assert 'depth' in info
    assert 'score' in info
    assert info['nodes'] > 0
