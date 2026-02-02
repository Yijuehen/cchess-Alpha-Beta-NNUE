"""
Main Xiangqi Engine interface.

Coordinates board representation, move generation, evaluation, and search.
"""

import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

from ..board.board_representation import (
    parse_board, encode_board,
    create_initial_board, validate_board,
    is_red, is_black
)
from ..board.visualization import print_board, board_to_str
from ..movegen.move_generator import (
    Move, encode_move, decode_move, parse_move, format_move,
    generate_moves, make_move, is_capture
)
from ..movegen.validators import (
    is_legal, generate_legal_moves,
    is_check, is_checkmate, is_stalemate
)
from ..evaluation.evaluator import Evaluator, create_evaluator
from ..nnue.nue_network import NNUE
from ..search.alphabeta import (
    search, iterative_deepening, SearchInfo
)
from ..utils.logger import get_logger

logger = get_logger("engine")


class XiangqiEngine:
    """
    Chinese Chess engine combining Alpha-Beta search with NNUE evaluation.
    """

    def __init__(
        self,
        nnue_path: Optional[str] = None,
        input_size: int = 1260,
        hidden_size: int = 256
    ):
        """
        Initialize engine.

        Args:
            nnue_path: Path to NNUE weights file (optional)
            input_size: NNUE input size
            hidden_size: NNUE hidden size
        """
        # Initialize evaluator
        self.evaluator = create_evaluator(nnue_path, input_size, hidden_size)

        # Initialize board to starting position
        self.board = create_initial_board()

        # Search state
        self.search_info = SearchInfo()
        self.depth = 4  # Default search depth

        # Game state
        self.side_to_move = 1  # Red starts
        self.board_history = []  # Store board states for undo

        logger.info(
            f"XiangqiEngine initialized: nnue={nnue_path is not None}, "
            f"depth={self.depth}"
        )

    def set_position(self, board_str: str) -> None:
        """
        Set position from board string.

        Args:
            board_str: 64-character board encoding
        """
        try:
            self.board = parse_board(board_str)

            if not validate_board(self.board):
                raise ValueError("Invalid board position")

            logger.debug(f"Position set: {board_str[:16]}...")
        except Exception as e:
            logger.error(f"Failed to set position: {e}")
            raise

    def get_position(self) -> str:
        """
        Get current position as board string.

        Returns:
            64-character board encoding
        """
        return encode_board(self.board)

    def reset(self) -> None:
        """Reset to initial position."""
        self.board = create_initial_board()
        self.side_to_move = 1
        self.board_history = []  # Clear history

    def set_fen(self, fen: str) -> None:
        """
        Set position from FEN string (not yet implemented).

        Args:
            fen: FEN string
        """
        raise NotImplementedError("FEN support not yet implemented")

    def get_fen(self) -> str:
        """
        Get current position as FEN string (not yet implemented).

        Returns:
            FEN string
        """
        raise NotImplementedError("FEN support not yet implemented")

    def make_move(self, move_str: str) -> bool:
        """
        Make a move.

        Args:
            move_str: 4-character move string (e.g., "7062")

        Returns:
            True if move was legal and made, False otherwise
        """
        try:
            move = parse_move(move_str)

            # Check if move is legal
            if not is_legal(self.board, move, self.side_to_move):
                logger.debug(f"Illegal move attempted: {move_str}")
                return False

            # Save state for undo (board copy and side_to_move)
            self.board_history.append((self.board.copy(), self.side_to_move))

            # Make the move
            self.board = make_move(self.board, move)

            # Switch sides
            self.side_to_move = -self.side_to_move

            logger.debug(f"Move made: {move_str}, side={self.side_to_move}")
            return True

        except Exception as e:
            logger.warning(f"Move failed: {move_str}, error={e}")
            return False

    def undo_move(self, move_str: str = None) -> bool:
        """
        Undo the last move.

        Args:
            move_str: Move to undo (optional, not used but kept for compatibility)

        Returns:
            True if successful, False if no history to undo
        """
        if not self.board_history:
            logger.debug("No moves to undo")
            return False

        # Restore previous state
        self.board, self.side_to_move = self.board_history.pop()
        logger.debug(f"Undo performed, side={self.side_to_move}")
        return True

    def search(self, depth: Optional[int] = None) -> Tuple[int, str]:
        """
        Search for best move.

        Args:
            depth: Search depth (uses engine default if None)

        Returns:
            Tuple of (score, best_move_string)
        """
        if depth is None:
            depth = self.depth

        logger.info(f"Search started: depth={depth}, side={self.side_to_move}")

        # Clear search info
        self.search_info = SearchInfo()

        # Search - PASS side_to_move to ensure engine moves correct color
        score, best_move = iterative_deepening(
            self.board,
            depth,
            self.evaluator,
            self.search_info,
            self.side_to_move  # CRITICAL: Pass current side to move
        )

        if best_move is None:
            logger.warning("No legal moves found")
            return score, ""

        move_str = format_move(best_move)
        logger.info(f"Search result: move={move_str}, score={score}")

        return score, move_str

    def analyze(self, depth: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Analyze current position and return top moves.

        Args:
            depth: Analysis depth

        Returns:
            List of (move_string, score) tuples sorted by score
        """
        if depth is None:
            depth = self.depth

        results = []

        # Generate legal moves
        moves = generate_legal_moves(self.board, self.side_to_move)

        for move in moves:
            # Make move
            new_board = make_move(self.board, move)

            # Evaluate resulting position
            score = -self.evaluator.evaluate(
                new_board,
                perspective=self.side_to_move
            )

            move_str = format_move(move)
            results.append((move_str, int(score)))

        # Sort by score (descending for Red, ascending for Black)
        reverse = self.side_to_move == 1
        results.sort(key=lambda x: x[1], reverse=reverse)

        return results

    def evaluate(self) -> int:
        """
        Evaluate current position.

        Returns:
            Evaluation score in centipawns
        """
        return self.evaluator.evaluate(self.board, perspective=self.side_to_move)

    def is_check(self) -> bool:
        """Check if current side is in check."""
        return is_check(self.board, self.side_to_move)

    def is_checkmate(self) -> bool:
        """Check if current position is checkmate."""
        return is_checkmate(self.board, self.side_to_move)

    def is_stalemate(self) -> bool:
        """Check if current position is stalemate."""
        return is_stalemate(self.board, self.side_to_move)

    def get_legal_moves(self) -> List[str]:
        """
        Get all legal moves for current side.

        Returns:
            List of move strings
        """
        moves = generate_legal_moves(self.board, self.side_to_move)
        return [format_move(m) for m in moves]

    def print_board(self) -> None:
        """Print current board to console."""
        print_board(self.board)

    def board_to_string(self) -> str:
        """
        Get board as string.

        Returns:
            Board string representation
        """
        return board_to_str(self.board)

    def load_nnue(self, nnue_path: str) -> None:
        """
        Load NNUE weights.

        Args:
            nnue_path: Path to NNUE weights file
        """
        nnue = NNUE()
        nnue.load(nnue_path)
        self.evaluator.set_nnue(nnue)

    def set_search_depth(self, depth: int) -> None:
        """
        Set default search depth.

        Args:
            depth: Search depth (plies)
        """
        if depth < 1 or depth > 64:
            raise ValueError("Depth must be between 1 and 64")
        self.depth = depth

    def get_search_info(self) -> dict:
        """
        Get information about last search.

        Returns:
            Dictionary with search statistics
        """
        return {
            'nodes': self.search_info.nodes,
            'depth': self.search_info.depth,
            'score': self.search_info.score,
            'best_move': format_move(self.search_info.best_move) if self.search_info.best_move else None,
            'time_ms': self.search_info.time_ms,
        }

    def play(self, moves: List[str]) -> None:
        """
        Play a sequence of moves.

        Args:
            moves: List of move strings
        """
        for move_str in moves:
            if not self.make_move(move_str):
                raise ValueError(f"Illegal move: {move_str}")

    def perft(self, depth: int) -> int:
        """
        Performance test: count nodes to given depth.

        Args:
            depth: Depth to search

        Returns:
            Number of nodes
        """
        return _perft_recursive(self.board, depth, self.side_to_move)


def _perft_recursive(board: np.ndarray, depth: int, color: int) -> int:
    """Helper for perft test."""
    if depth == 0:
        return 1

    moves = generate_legal_moves(board, color)
    nodes = 0

    for move in moves:
        new_board = make_move(board, move)
        nodes += _perft_recursive(new_board, depth - 1, -color)

    return nodes


def create_engine(nnue_path: Optional[str] = None) -> XiangqiEngine:
    """
    Create a Xiangqi engine.

    Args:
        nnue_path: Optional path to NNUE weights

    Returns:
        Configured XiangqiEngine instance
    """
    return XiangqiEngine(nnue_path=nnue_path)
