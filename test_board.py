#!/usr/bin/env python3
"""Test board initialization."""

from src.board.board_representation import create_initial_board, RED_ROOK, BLACK_ROOK

board = create_initial_board()

print("Row 0 (Black back rank - should be negative):")
for col in range(9):
    val = board[0, col]
    print(f"  [{0},{col}] = {val} (type: {type(val).__name__})")

print("\nRow 9 (Red back rank - should be positive):")
for col in range(9):
    val = board[9, col]
    print(f"  [{9},{col}] = {val} (type: {type(val).__name__})")

print("\nExpected:")
print("Row 0: [-5, -4, -3, -2, -1, -2, -3, -4, -5]")
print("Row 9: [5, 4, 3, 2, 1, 2, 3, 4, 5]")
