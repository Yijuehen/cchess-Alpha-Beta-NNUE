# Chinese Chess Engine with Alpha-Beta + NNUE

A Python-based Chinese Chess (Xiangqi) engine combining traditional Alpha-Beta search with NNUE (Neural Network User Evaluation) for fast, accurate position evaluation.

## Features

- **Complete Move Generation**: All 7 piece types with Chinese Chess-specific rules
  - Palace restrictions (General, Advisor)
  - River crossing (Elephant, Soldier)
  - Blocking rules (Horse leg, Elephant eye)
  - Jump captures (Cannon)
  - Flying General rule

- **Alpha-Beta Search**:
  - Iterative deepening
  - Move ordering with MVV-LVA
  - Quiescence search support

- **NNUE Evaluation**:
  - Shallow neural network (1260 → 256 → 1)
  - SCReLU activation for quantization-friendly inference
  - Feature extraction from board positions
  - Material evaluation fallback

- **Training Pipeline**:
  - Train on CSV game data
  - Multiple loss functions (MSE, Huber, Scaled MSE)
  - Validation and early stopping
  - Checkpointing

## Installation

```bash
# Clone or navigate to the project
cd cchess-Alpha-Beta+NNUE

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
cchess-Alpha-Beta+NNUE/
├── data/
│   └── chess.csv              # Training data (1.9M positions)
├── src/
│   ├── board/                 # Board representation
│   ├── movegen/               # Move generation
│   ├── features/              # Feature extraction for NNUE
│   ├── nnue/                  # NNUE network
│   ├── training/              # Training pipeline
│   ├── search/                # Alpha-Beta search
│   ├── evaluation/            # Position evaluation
│   └── engine/                # Main engine interface
├── scripts/
│   ├── train_model.py         # Training script
│   └── benchmark.py           # Performance benchmarking
├── tests/                     # Unit tests
├── models/                    # Trained models
└── requirements.txt
```

## Quick Start

### Basic Engine Usage

```python
from src.engine.xiangqi_engine import create_engine

# Create engine (without NNUE for now)
engine = create_engine()

# Set position
engine.set_position("0919293949596979891777062646668600102030405060708012720323436383")

# Search for best move
score, move = engine.search(depth=4)
print(f"Best move: {move}, Score: {score}")

# Analyze position
analysis = engine.analyze(depth=4)
for move, score in analysis[:5]:
    print(f"{move}: {score}")
```

### Training NNUE Model

```bash
# Train on full dataset
python scripts/train_model.py \
    --data data/chess.csv \
    --output models/nnue_weights.pkl \
    --epochs 10 \
    --batch-size 32

# Quick test with limited samples
python scripts/train_model.py \
    --data data/chess.csv \
    --output models/nnue_test.pkl \
    --max-samples 10000 \
    --epochs 3
```

### Benchmarking

```bash
# Search performance
python scripts/benchmark.py search --depth 8

# Move generation (perft)
python scripts/benchmark.py perft --max-depth 5

# Self-game
python scripts/benchmark.py game --depth 4 --max-moves 20
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_board.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Data Format

The CSV file contains Chinese Chess game positions with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| chessboradStatus | 64-character board encoding | `0919293949596979891777062646668600102030405060708012720323436383` |
| moveStatus | 4-digit move encoding | `7062` |
| chessboradStatusAfterMoving | Board after move | `0919293949596979891747062646668600102030405060708012720323436383` |
| predictAction | Predicted next action | `None` |
| result | Game result | `0`=Red wins, `1`=Black wins, `2`=Draw, `None`=incomplete |

## Board Encoding

The 64-character string encodes 32 piece positions (2 digits each):

```
Positions: 00-89 (row × 9 + col)

Red pieces:
- 09,19,29,39,49,59,69,79,89: Rooks
- 17,77: Cannons
- 06,26,46,66,86: Soldiers
- 18,78: Horses
- 27,67: Elephants
- 37,57: Advisors
- 47: General

Black pieces:
- 00,80: Rooks
- 10,70: Horses
- 20,60: Advisors
- 30,50: Soldiers
- 40: General
- 12,72,03,23,43,63,83: Cannons
```

## API Reference

### XiangqiEngine

Main engine class for playing Chinese Chess.

```python
engine = XiangqiEngine(nnue_path="models/nnue_weights.pkl")

# Position management
engine.set_position(board_str)
engine.reset()
position = engine.get_position()

# Move making
success = engine.make_move("7062")
engine.undo_move("7062")

# Search
score, move = engine.search(depth=4)
analysis = engine.analyze(depth=4)

# Evaluation
score = engine.evaluate()
is_check = engine.is_check()
is_mate = engine.is_checkmate()

# Legal moves
moves = engine.get_legal_moves()

# Display
engine.print_board()
board_str = engine.board_to_string()
```

### Training

```python
from src.training.trainer import train_model

network = train_model(
    csv_path="data/chess.csv",
    output_path="models/nnue_weights.pkl",
    input_size=1260,
    hidden_size=256,
    learning_rate=0.01,
    epochs=10,
    batch_size=32,
    max_samples=100000,  # None for all data
    validation_ratio=0.1,
    loss_fn="mse"
)
```

## Performance

Target performance metrics:

- **Search Speed**: >100,000 nodes/second (Python)
- **Training**: Converges within 10 epochs on full dataset
- **Evaluation**: <1ms per position with NNUE

Note: Performance is limited by Python's interpreter. For production use, consider:
- Numba JIT compilation for hot paths
- Cython for critical modules
- Port to C++ for 10-100x speedup

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  XiangqiEngine                      │
│  (Main API: position setup, search, analysis)       │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐    ┌─────────────────┐
│  Alpha-Beta      │    │  NNUE           │
│  Search Engine   │◄───┤  Evaluator      │
│  - Iterative     │    │  - Feature      │
│    Deepening     │    │    Transformer  │
│  - Move Ordering │    │  - Shallow Net  │
│  - Quiescence    │    │  - SCReLU       │
└────────┬─────────┘    └─────────────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌────────┐ ┌───────┐
│ Board  │ │ Move  │
│ Rep.   │ │ Gen.  │
└────────┘ └───────┘
```

## Limitations

- **Performance**: Python is slower than C++ engines
- **Opening Book**: No opening book support
- **Endgame Tablebases**: No tablebase support
- **UCI Protocol**: Basic implementation only

## Future Enhancements

1. **Performance**:
   - Numba JIT compilation
   - Cython for critical paths
   - Bitboard representation

2. **Features**:
   - Opening book
   - Endgame tablebases
   - UCI protocol support
   - Web interface

3. **Training**:
   - Deeper network architecture
   - Reinforcement learning from self-play
   - Multi-GPU training support

## References

- [Pikafish](https://github.com/official-stockfish/nnue-pytorch) - NNUE training methodology
- [Stockfish NNUE](https://github.com/official-stockfish/Stockfish) - Original NNUE implementation
- Chinese Chess rules and piece movement

## License

This project is for educational and research purposes.

## Contributing

Contributions welcome! Areas for improvement:
- Performance optimization
- Additional training data
- Feature enhancements
- Bug fixes and testing

## Acknowledgments

- NNUE methodology from Stockfish and Stockfish NNUE-pytorch
- Chinese Chess rules and piece movement guidelines
- Alpha-Beta search algorithm foundations
