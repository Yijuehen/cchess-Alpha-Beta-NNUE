#!/usr/bin/env python3
"""
Chinese Chess game with GUI using Tkinter.

Play against the NNUE engine with a visual interface.
"""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, scrolledtext

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.xiangqi_engine import create_engine


class ChineseChessGUI:
    """Graphical interface for Chinese Chess."""

    # Colors
    COLOR_BOARD = "#DEB887"  # Light brown
    COLOR_LINE = "#8B4513"   # Dark brown
    COLOR_RED = "#DC143C"    # Red for Red pieces
    COLOR_BLACK = "#000000"  # Black for Black pieces
    COLOR_SELECTED = "#90EE90"  # Light green for selected square
    COLOR_LAST_MOVE = "#87CEEB"  # Light blue for last move
    COLOR_BG = "#F5F5DC"    # Beige background

    # Piece symbols (Chinese characters)
    # Mapping matches piece constants: 1=GENERAL, 2=ADVISOR, 3=ELEPHANT, 4=HORSE, 5=ROOK, 6=CANNON, 7=SOLDIER
    PIECES = {
        1: "帥", 2: "仕", 3: "相", 4: "馬", 5: "車", 6: "炮", 7: "兵",
        -1: "將", -2: "士", -3: "象", -4: "馬", -5: "車", -6: "炮", -7: "卒"
    }

    # Board dimensions
    CELL_SIZE = 70
    MARGIN = 50
    BOARD_WIDTH = CELL_SIZE * 8
    BOARD_HEIGHT = CELL_SIZE * 9

    def __init__(self, nnue_path='models/nnue_weights.pkl'):
        self.root = tk.Tk()
        self.root.title("Chinese Chess - NNUE Engine")
        self.root.geometry("1000x800")
        self.root.configure(bg=self.COLOR_BG)

        # Initialize engine
        self.engine = create_engine(nnue_path=nnue_path)
        self.engine.reset()

        # Game state
        self.selected_pos = None
        self.legal_moves = []
        self.move_history = []
        self.game_over = False

        # Setup GUI
        self.setup_ui()
        self.draw_board()

    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - Board
        board_frame = tk.Frame(main_frame, bg=self.COLOR_BG)
        board_frame.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(
            board_frame,
            width=self.BOARD_WIDTH + 2 * self.MARGIN,
            height=self.BOARD_HEIGHT + 2 * self.MARGIN,
            bg=self.COLOR_BOARD,
            highlightthickness=2,
            highlightbackground=self.COLOR_LINE
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_board_click)

        # Right side - Info and Controls
        info_frame = tk.Frame(main_frame, bg=self.COLOR_BG, width=300)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        # Title
        title = tk.Label(
            info_frame,
            text="Chinese Chess",
            font=("Arial", 24, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_LINE
        )
        title.pack(pady=10)

        # Status label
        self.status_label = tk.Label(
            info_frame,
            text="Your turn (Red)",
            font=("Arial", 16),
            bg=self.COLOR_BG,
            fg=self.COLOR_RED
        )
        self.status_label.pack(pady=10)

        # Engine score label
        self.score_label = tk.Label(
            info_frame,
            text="Score: 0.00",
            font=("Arial", 12),
            bg=self.COLOR_BG
        )
        self.score_label.pack(pady=5)

        # Control buttons
        button_frame = tk.Frame(info_frame, bg=self.COLOR_BG)
        button_frame.pack(pady=10)

        tk.Button(
            button_frame,
            text="New Game",
            command=self.new_game,
            width=12,
            height=2
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="Undo Move",
            command=self.undo_move,
            width=12,
            height=2
        ).pack(side=tk.LEFT, padx=5)

        # Move history
        history_label = tk.Label(
            info_frame,
            text="Move History:",
            font=("Arial", 12, "bold"),
            bg=self.COLOR_BG
        )
        history_label.pack(pady=(20, 5), anchor="w")

        self.history_text = scrolledtext.ScrolledText(
            info_frame,
            width=35,
            height=20,
            font=("Courier", 10)
        )
        self.history_text.pack(fill=tk.BOTH, expand=True)

        # Legend
        legend_frame = tk.Frame(info_frame, bg=self.COLOR_BG)
        legend_frame.pack(pady=10, fill=tk.X)

        tk.Label(
            legend_frame,
            text="You: Red (先手)",
            fg=self.COLOR_RED,
            bg=self.COLOR_BG,
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=10)

        tk.Label(
            legend_frame,
            text="Engine: Black (後手)",
            fg=self.COLOR_BLACK,
            bg=self.COLOR_BG,
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=10)

    def draw_board(self):
        """Draw the board and pieces."""
        self.canvas.delete("all")

        # Draw grid lines
        for i in range(10):  # Rows (horizontal lines)
            y = self.MARGIN + i * self.CELL_SIZE
            self.canvas.create_line(
                self.MARGIN, y,
                self.MARGIN + 8 * self.CELL_SIZE, y,
                fill=self.COLOR_LINE, width=2
            )

        for i in range(9):  # Columns (vertical lines)
            x = self.MARGIN + i * self.CELL_SIZE
            # Top half (above river)
            self.canvas.create_line(
                x, self.MARGIN,
                x, self.MARGIN + 4 * self.CELL_SIZE,
                fill=self.COLOR_LINE, width=2
            )
            # Bottom half (below river)
            self.canvas.create_line(
                x, self.MARGIN + 5 * self.CELL_SIZE,
                x, self.MARGIN + 9 * self.CELL_SIZE,
                fill=self.COLOR_LINE, width=2
            )

        # Draw palace diagonals (top)
        self.canvas.create_line(
            self.MARGIN + 3 * self.CELL_SIZE, self.MARGIN,
            self.MARGIN + 5 * self.CELL_SIZE, self.MARGIN + 2 * self.CELL_SIZE,
            fill=self.COLOR_LINE, width=2
        )
        self.canvas.create_line(
            self.MARGIN + 5 * self.CELL_SIZE, self.MARGIN,
            self.MARGIN + 3 * self.CELL_SIZE, self.MARGIN + 2 * self.CELL_SIZE,
            fill=self.COLOR_LINE, width=2
        )

        # Draw palace diagonals (bottom)
        self.canvas.create_line(
            self.MARGIN + 3 * self.CELL_SIZE, self.MARGIN + 7 * self.CELL_SIZE,
            self.MARGIN + 5 * self.CELL_SIZE, self.MARGIN + 9 * self.CELL_SIZE,
            fill=self.COLOR_LINE, width=2
        )
        self.canvas.create_line(
            self.MARGIN + 5 * self.CELL_SIZE, self.MARGIN + 7 * self.CELL_SIZE,
            self.MARGIN + 3 * self.CELL_SIZE, self.MARGIN + 9 * self.CELL_SIZE,
            fill=self.COLOR_LINE, width=2
        )

        # Draw river text
        river_y = self.MARGIN + 4.5 * self.CELL_SIZE
        self.canvas.create_text(
            self.MARGIN + 2 * self.CELL_SIZE, river_y,
            text="楚 河",
            font=("KaiTi", 20, "bold"),
            fill=self.COLOR_LINE
        )
        self.canvas.create_text(
            self.MARGIN + 6 * self.CELL_SIZE, river_y,
            text="漢 界",
            font=("KaiTi", 20, "bold"),
            fill=self.COLOR_LINE
        )

        # Highlight last move
        if self.move_history:
            last_move = self.move_history[-1]
            from_pos, to_pos = self._decode_move(last_move)
            self._highlight_square(from_pos, self.COLOR_LAST_MOVE)
            self._highlight_square(to_pos, self.COLOR_LAST_MOVE)

        # Highlight selected square
        if self.selected_pos is not None:
            self._highlight_square(self.selected_pos, self.COLOR_SELECTED)

        # Highlight legal moves
        for move in self.legal_moves:
            _, to_pos = self._decode_move(move)
            self._draw_move_dot(to_pos)

        # Draw pieces
        board = self.engine.board
        for row in range(10):
            for col in range(9):
                piece = board[row, col]
                if piece != 0:
                    pos = row * 9 + col
                    self._draw_piece(row, col, piece)

    def _draw_piece(self, row, col, piece):
        """Draw a piece at the given position."""
        x = self.MARGIN + col * self.CELL_SIZE
        y = self.MARGIN + row * self.CELL_SIZE

        # Convert to Python int for reliable comparison
        piece_val = int(piece)

        # Piece circle
        radius = self.CELL_SIZE // 2 - 5
        color = self.COLOR_RED if piece_val > 0 else self.COLOR_BLACK

        self.canvas.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            fill="#FFF8DC", outline=color, width=3
        )

        # Piece character
        symbol = self.PIECES[piece_val]
        self.canvas.create_text(
            x, y,
            text=symbol,
            font=("KaiTi", 28, "bold"),
            fill=color
        )

    def _highlight_square(self, pos, color):
        """Highlight a square."""
        row = pos // 9
        col = pos % 9
        x = self.MARGIN + col * self.CELL_SIZE
        y = self.MARGIN + row * self.CELL_SIZE

        self.canvas.create_rectangle(
            x - self.CELL_SIZE // 2 + 2,
            y - self.CELL_SIZE // 2 + 2,
            x + self.CELL_SIZE // 2 - 2,
            y + self.CELL_SIZE // 2 - 2,
            fill=color, outline=""
        )

    def _draw_move_dot(self, pos):
        """Draw a small dot indicating a legal move."""
        row = pos // 9
        col = pos % 9
        x = self.MARGIN + col * self.CELL_SIZE
        y = self.MARGIN + row * self.CELL_SIZE

        self.canvas.create_oval(
            x - 8, y - 8,
            x + 8, y + 8,
            fill="#32CD32", outline=""
        )

    def _decode_move(self, move_str):
        """Decode a 4-character move string to from_pos and to_pos."""
        from_pos = int(move_str[:2])
        to_pos = int(move_str[2:])
        return from_pos, to_pos

    def _encode_move(self, from_pos, to_pos):
        """Encode from_pos and to_pos to a 4-character move string."""
        return f"{from_pos:02d}{to_pos:02d}"

    def on_board_click(self, event):
        """Handle board click events."""
        if self.game_over:
            return

        # Only respond on human's turn (Red)
        if self.engine.side_to_move != 1:
            return

        # Convert pixel to position
        col = round((event.x - self.MARGIN) / self.CELL_SIZE)
        row = round((event.y - self.MARGIN) / self.CELL_SIZE)

        if not (0 <= row < 10 and 0 <= col < 9):
            return

        pos = row * 9 + col

        # If no piece selected, try to select one
        if self.selected_pos is None:
            piece = int(self.engine.board[row, col])
            if piece > 0:  # Red piece
                self.selected_pos = pos
                # Get legal moves for this piece
                all_moves = self.engine.get_legal_moves()
                self.legal_moves = [m for m in all_moves if m[:2] == f"{pos:02d}"]
                self.draw_board()
            return

        # If piece already selected
        if pos == self.selected_pos:
            # Deselect
            self.selected_pos = None
            self.legal_moves = []
            self.draw_board()
            return

        # Try to make move
        move = self._encode_move(self.selected_pos, pos)

        if move in self.legal_moves:
            # Valid move
            self.make_human_move(move)
        else:
            # Invalid move, check if clicking on another piece
            piece = int(self.engine.board[row, col])
            if piece > 0:  # Red piece
                self.selected_pos = pos
                all_moves = self.engine.get_legal_moves()
                self.legal_moves = [m for m in all_moves if m[:2] == f"{pos:02d}"]
                self.draw_board()
            else:
                # Deselect
                self.selected_pos = None
                self.legal_moves = []
                self.draw_board()

    def make_human_move(self, move):
        """Make the human's move."""
        # Make move
        self.engine.make_move(move)

        # Update history
        move_num = len(self.move_history) // 2 + 1
        self.move_history.append(move)
        self._add_to_history(move_num, move, "Red")

        # Clear selection
        self.selected_pos = None
        self.legal_moves = []

        # Redraw
        self.draw_board()

        # Check game state
        if self._check_game_over():
            return

        # Update status
        self.status_label.config(
            text="Engine thinking...",
            fg=self.COLOR_BLACK
        )
        self.root.update()

        # Engine's turn
        self.root.after(100, self.make_engine_move)

    def make_engine_move(self):
        """Make the engine's move."""
        # CRITICAL: Verify it's Black's turn (engine plays Black)
        if self.engine.side_to_move != -1:
            print(f"ERROR: Engine called but it's not Black's turn! side_to_move={self.engine.side_to_move}")
            self.status_label.config(text="Error: Not engine's turn", fg=self.COLOR_RED)
            return

        # Search for best move (engine searches for Black's best move)
        score, best_move = self.engine.search(depth=3)

        if best_move is None or best_move == "":
            self.game_over = True
            self.status_label.config(text="You win!", fg=self.COLOR_RED)
            messagebox.showinfo("Game Over", "Engine has no moves - You win!")
            return

        # Verify the move is valid
        from_pos = int(best_move[:2])
        from_row, from_col = from_pos // 9, from_pos % 9
        piece = int(self.engine.board[from_row, from_col])

        # Engine should only move Black pieces (negative values)
        if piece >= 0:
            print(f"ERROR: Engine trying to move Red piece! piece={piece} at ({from_row},{from_col})")
            print(f"Move: {best_move}, side_to_move={self.engine.side_to_move}")
            self.status_label.config(text="Engine error - try New Game", fg=self.COLOR_RED)
            return

        # Make move
        if not self.engine.make_move(best_move):
            print(f"ERROR: Engine failed to make move: {best_move}")
            self.status_label.config(text="Move failed - try New Game", fg=self.COLOR_RED)
            return

        # Update history
        move_num = len(self.move_history) // 2 + 1
        self.move_history.append(best_move)
        self._add_to_history(move_num, best_move, "Black")

        # Update score
        self.score_label.config(text=f"Score: {score:.2f}")

        # Redraw
        self.draw_board()

        # Check game state
        if not self._check_game_over():
            # Update status
            self.status_label.config(
                text="Your turn (Red)",
                fg=self.COLOR_RED
            )

            # Check status
            if self.engine.is_check():
                self.status_label.config(text="Check! Your turn", fg=self.COLOR_RED)

    def _add_to_history(self, move_num, move, player):
        """Add move to history display."""
        from_pos, to_pos = self._decode_move(move)
        from_row, from_col = from_pos // 9, from_pos % 9
        to_row, to_col = to_pos // 9, to_pos % 9

        # Convert to algebraic notation
        col_names = "abcdefghi"
        move_str = f"{move_num}. {player}: ({from_col},{from_row}) -> ({to_col},{to_row})"

        self.history_text.insert(tk.END, move_str + "\n")
        self.history_text.see(tk.END)

    def _check_game_over(self):
        """Check if game is over."""
        if self.engine.is_checkmate():
            self.game_over = True
            winner = "Black" if self.engine.side_to_move == 1 else "Red"
            if winner == "Red":
                self.status_label.config(text="Checkmate! You win!", fg=self.COLOR_RED)
                messagebox.showinfo("Game Over", "Checkmate! You win!")
            else:
                self.status_label.config(text="Checkmate! Engine wins!", fg=self.COLOR_BLACK)
                messagebox.showinfo("Game Over", "Checkmate! Engine wins!")
            return True

        # Check for stalemate (no legal moves)
        moves = self.engine.get_legal_moves()
        if not moves:
            self.game_over = True
            winner = "Black" if self.engine.side_to_move == 1 else "Red"
            if winner == "Red":
                self.status_label.config(text="No moves! You win!", fg=self.COLOR_RED)
                messagebox.showinfo("Game Over", "No legal moves! You win!")
            else:
                self.status_label.config(text="No moves! Engine wins!", fg=self.COLOR_BLACK)
                messagebox.showinfo("Game Over", "No legal moves! Engine wins!")
            return True

        return False

    def new_game(self):
        """Start a new game."""
        self.engine.reset()
        self.selected_pos = None
        self.legal_moves = []
        self.move_history = []
        self.game_over = False

        self.history_text.delete(1.0, tk.END)
        self.status_label.config(text="Your turn (Red)", fg=self.COLOR_RED)
        self.score_label.config(text="Score: 0.00")

        self.draw_board()

    def undo_move(self):
        """Undo the last two moves (human + engine)."""
        try:
            if len(self.move_history) == 0:
                return  # No moves to undo

            # Determine how many moves to undo (always undo pairs if possible)
            moves_to_undo = 2 if len(self.move_history) >= 2 else 1

            for _ in range(moves_to_undo):
                if self.move_history:
                    self.move_history.pop()
                if not self.engine.undo_move():
                    break  # Stop if undo fails

            # CRITICAL: After undoing, it should always be Red's turn (human)
            # The engine moves automatically after the human, so undoing brings us back to human's turn
            self.engine.side_to_move = 1

            # Rebuild history display from scratch (safer than deleting lines)
            self.history_text.delete(1.0, tk.END)
            for i, move in enumerate(self.move_history):
                move_num = i // 2 + 1
                player = "Red" if i % 2 == 0 else "Black"
                self._add_to_history_direct(move_num, move, player)

            # Clear selection
            self.selected_pos = None
            self.legal_moves = []

            # Redraw
            self.draw_board()

            # Always set to Red's turn after undo
            self.status_label.config(text="Your turn (Red)", fg=self.COLOR_RED)

            if self.game_over:
                self.game_over = False

        except Exception as e:
            # Log error and show message instead of freezing
            print(f"Undo error: {e}")
            self.status_label.config(text="Undo failed - try New Game", fg=self.COLOR_BLACK)

    def _add_to_history_direct(self, move_num, move, player):
        """Add move to history display without auto-scroll (for rebuild)."""
        from_pos, to_pos = self._decode_move(move)
        from_row, from_col = from_pos // 9, from_pos % 9
        to_row, to_col = to_pos // 9, to_pos % 9

        move_str = f"{move_num}. {player}: ({from_col},{from_row}) -> ({to_col},{to_row})"
        self.history_text.insert(tk.END, move_str + "\n")

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def main():
    """Main entry point."""
    print("Loading NNUE model...")
    print("If model file is not found, the engine will use material evaluation only.")
    print()

    try:
        game = ChineseChessGUI(nnue_path='models/nnue_weights.pkl')
        game.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease train the model first:")
        print("  python scripts/train_model.py --config config/test_config.yaml")
        print("\nOr use config/full_config.yaml for full training")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
