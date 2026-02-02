#!/usr/bin/env python3
"""
Simple Chinese Chess game player against NNUE engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.xiangqi_engine import create_engine


def main():
    # Load engine with trained NNUE
    print("Loading NNUE model...")
    engine = create_engine(nnue_path='models/nnue_weights.pkl')

    # Reset to starting position
    engine.reset()

    print("\n" + "=" * 60)
    print("Chinese Chess - Human (Red) vs NNUE Engine (Black)")
    print("=" * 60)
    print("\nMove format: 4-digit code (e.g., 7062)")
    print("Commands: 'q' = quit, 'board' = show board")
    print()

    while True:
        # Show board
        engine.print_board()
        print()

        # Check if game is over
        if engine.is_checkmate():
            winner = "Black" if engine.side_to_move == 1 else "Red"
            print(f"Checkmate! {winner} wins!")
            break

        # Get legal moves
        moves = engine.get_legal_moves()
        if not moves:
            winner = "Black" if engine.side_to_move == 1 else "Red"
            print(f"No legal moves! {winner} wins!")
            break

        # Show whose turn it is
        player = "Red (You)" if engine.side_to_move == 1 else "Black (Engine)"
        print(f"\n{player}'s turn")

        # Show available moves (first 15)
        print(f"Legal moves: {' '.join(moves[:15])}")
        if len(moves) > 15:
            print(f"            ... and {len(moves) - 15} more")

        # Get player's move
        if engine.side_to_move == 1:  # Red's turn (human)
            move = input("\nEnter your move: ").strip()

            if move.lower() == 'q':
                print("\nGame ended by user.")
                break

            if move.lower() == 'board':
                continue

            if move not in moves:
                print(f"Invalid move! Valid moves: {moves[:5]}...")
                continue

            # Make human's move
            engine.make_move(move)
            print(f"You played: {move}")

            if engine.is_check():
                print("Check!")

        else:  # Black's turn (engine)
            print("\nEngine is thinking...")

            # Engine searches for best move
            score, best_move = engine.search(depth=4)

            if best_move is None:
                print("Engine has no moves - You win!")
                break

            # Make engine's move
            engine.make_move(best_move)
            print(f"Engine played: {best_move} (score: {score:.2f})")

            if engine.is_check():
                print("Check!")

        print("-" * 60)

    print("\nThanks for playing!")


if __name__ == '__main__':
    main()
