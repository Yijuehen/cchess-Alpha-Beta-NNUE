#!/usr/bin/env python3
"""
Benchmark script for the Xiangqi engine.

Tests search performance and plays test games.
"""

import argparse
import time
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.xiangqi_engine import create_engine
from src.board.board_representation import create_initial_board


def benchmark_search(depth: int = 8, time_limit: float = 5.0):
    """Benchmark search performance."""
    print("=" * 60)
    print(f"Search Benchmark (depth={depth})")
    print("=" * 60)

    engine = create_engine()
    engine.set_search_depth(depth)

    # Test on initial position
    start_time = time.time()

    score, move = engine.search(depth=depth)

    elapsed = time.time() - start_time
    info = engine.get_search_info()

    print(f"Best move: {move}")
    print(f"Score: {score}")
    print(f"Nodes searched: {info['nodes']}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Nodes per second: {info['nodes'] / elapsed:.0f}")

    return info['nodes'] / elapsed


def benchmark_perft(max_depth: int = 5):
    """Benchmark move generation (perft test)."""
    print("=" * 60)
    print(f"Perft Benchmark (max depth={max_depth})")
    print("=" * 60)

    engine = create_engine()

    for depth in range(1, max_depth + 1):
        start = time.time()
        nodes = engine.perft(depth)
        elapsed = time.time() - start

        print(f"Depth {depth}: {nodes:,} nodes ({elapsed:.3f}s, {nodes/elapsed:.0f} nps)")


def play_self_game(depth: int = 4, max_moves: int = 50):
    """Play a self-game to test the engine."""
    print("=" * 60)
    print(f"Self-Game (depth={depth}, max_moves={max_moves})")
    print("=" * 60)

    engine = create_engine()
    engine.reset()

    for move_num in range(1, max_moves + 1):
        print(f"\nMove {move_num}:")

        # Get best move
        score, move = engine.search(depth=depth)

        if not move:
            print("No legal moves - game over")
            break

        print(f"  Score: {score}")
        print(f"  Move: {move}")

        # Make move
        engine.make_move(move)

        # Print board
        engine.print_board()

        # Check for game end
        if engine.is_checkmate():
            print("Checkmate!")
            break
        elif engine.is_stalemate():
            print("Stalemate!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Xiangqi engine"
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Search benchmark
    search_parser = subparsers.add_parser('search', help='Benchmark search')
    search_parser.add_argument(
        '--depth',
        type=int,
        default=8,
        help='Search depth'
    )

    # Perft benchmark
    perft_parser = subparsers.add_parser('perft', help='Perft test')
    perft_parser.add_argument(
        '--max-depth',
        type=int,
        default=5,
        help='Maximum depth'
    )

    # Self-game
    game_parser = subparsers.add_parser('game', help='Play self-game')
    game_parser.add_argument(
        '--depth',
        type=int,
        default=4,
        help='Search depth'
    )
    game_parser.add_argument(
        '--max-moves',
        type=int,
        default=50,
        help='Maximum moves'
    )

    args = parser.parse_args()

    if args.command == 'search':
        nps = benchmark_search(depth=args.depth)
        print(f"\nResult: {nps:.0f} nodes/second")

    elif args.command == 'perft':
        benchmark_perft(max_depth=args.max_depth)

    elif args.command == 'game':
        play_self_game(depth=args.depth, max_moves=args.max_moves)

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
