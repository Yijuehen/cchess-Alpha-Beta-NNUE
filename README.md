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

- **Logging System**:
  - Structured logging with multiple levels (DEBUG, INFO, WARNING, ERROR)
  - File logging with automatic rotation (10MB per file, 5 backups)
  - Console and JSON format options
  - Module-specific loggers

- **Background Tasks**:
  - Train models in background threads
  - Progress tracking and status monitoring
  - Async search operations
  - Batch position analysis

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

### Playing Against the Engine

**Command-Line Interface:**
```bash
# Play against the NNUE engine in terminal
python play_game.py
```

**Graphical Interface (GUI):**
```bash
# Play with a visual board interface
python play_game_gui.py
```

The GUI features:
- Visual Chinese Chess board with traditional pieces
- Click-to-move interface
- Move validation and highlighting
- Engine score display
- Move history panel
- Undo move functionality
- Check/checkmate detection

**You play Red (先手)** - pieces at the bottom, move first
**Engine plays Black (後手)** - pieces at the top

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

## Configuration Files

You can use YAML configuration files to manage training parameters instead of specifying them on the command line.

### Example Configuration Files

Two example configurations are provided:

**`config/test_config.yaml`** - Quick testing configuration
```yaml
# Quick test with limited samples
data:
  csv_path: "data/chess.csv"
  output_path: "models/test_model.pkl"

training:
  max_samples: 1000
  epochs: 1
  batch_size: 32
  learning_rate: 0.01

model:
  input_size: 1260
  hidden_size: 256

loss:
  type: "mse"  # Options: mse, huber, scaled_mse

validation:
  ratio: 0.1

checkpointing:
  enabled: false
  directory: "models/checkpoints"

logging:
  level: "INFO"
  log_dir: "logs/training"
```

**`config/full_config.yaml`** - Full production training
```yaml
# Complete training on all data
data:
  csv_path: "data/chess.csv"
  output_path: "models/nnue_weights.pkl"

training:
  max_samples: null  # null = use all samples
  epochs: 10
  batch_size: 32
  learning_rate: 0.01

# ... same structure as test_config.yaml
```

### Using Configuration Files

**Train with config file:**
```bash
# Use test configuration
python scripts/train_model.py --config config/test_config.yaml

# Use full configuration
python scripts/train_model.py --config config/full_config.yaml
```

**Override config values with command-line arguments:**
```bash
# Use config but override specific parameters
python scripts/train_model.py \
    --config config/test_config.yaml \
    --epochs 5 \
    --max-samples 5000
```

**Background training with config:**
```bash
# Start background training with config
python scripts/train_background.py \
    --config config/full_config.yaml \
    --task-id training_1 \
    --wait
```

### Creating Custom Configurations

You can create your own configuration files by copying and modifying the examples:

```bash
# Copy test config
cp config/test_config.yaml config/my_config.yaml

# Edit parameters
nano config/my_config.yaml  # or use your preferred editor

# Use your config
python scripts/train_model.py --config config/my_config.yaml
```

### Configuration File Priority

When both config file and command-line arguments are provided:
1. Command-line arguments take precedence (highest priority)
2. Config file values are used as defaults
3. Built-in defaults are used if neither is provided

Example:
```bash
# Config has epochs=10, but CLI overrides to 5
python scripts/train_model.py \
    --config config/full_config.yaml \
    --epochs 5  # This value will be used instead of config's 10
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

---

## Logging

The engine uses a structured logging system with multiple levels and automatic file rotation.

### Configuration

```python
from src.utils.logger import setup_logging, get_logger

# Setup logging for your application
setup_logging(
    name="my_app",
    log_dir="logs",
    level="INFO",
    console=True,
    file=True,
    json_format=False
)

# Get a logger for your module
logger = get_logger("my_module")
logger.info("Application started")
```

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about progress
- **WARNING**: Warning messages for non-critical issues
- **ERROR**: Error messages for failures
- **CRITICAL**: Critical errors that prevent operation

### Usage Examples

```python
from src.utils.logger import get_logger

logger = get_logger("training")

logger.info("Training started")
logger.debug(f"Epoch {epoch}: loss={loss:.4f}")
logger.warning("Skipping invalid sample")
logger.error("Failed to save model", exc_info=True)
```

### Log Files

Logs are automatically written to `logs/` directory:
- `logs/training/training.log` - Training logs
- `logs/benchmark/benchmark.log` - Benchmark logs
- `logs/background_training/background_training.log` - Background training logs

Features:
- Automatic rotation at 10MB per file
- Keeps 5 backup files
- UTF-8 encoding for Chinese characters

### Viewing Logs

```bash
# View latest logs
tail -f logs/training/training.log

# Search for errors
grep "ERROR" logs/training/training.log

# View with JSON parser (if using JSON format)
jq '.' logs/training/training.log
```

---

## Background Tasks

Run long-running operations in background threads with progress tracking.

### Background Training

Train models without blocking the main thread:

```python
from src.training.background_trainer import (
    train_model_background,
    check_training_status,
    wait_for_training
)

# Start training in background
task_id = train_model_background(
    csv_path="data/chess.csv",
    output_path="models/nnue.pkl",
    task_id="my_training_job",
    epochs=10,
    batch_size=32
)

# Check status
status = check_training_status(task_id)
print(f"Status: {status.status.value}")
print(f"Progress: {status.progress:.1%}")
print(f"Elapsed: {status.elapsed_time():.1f}s")

# Wait for completion (optional)
model = wait_for_training(task_id, timeout=None)
```

### Command Line Tools

**Start Background Training:**
```bash
python scripts/train_background.py \
    --data data/chess.csv \
    --output models/nnue.pkl \
    --task-id training_1 \
    --epochs 10

# The training runs in background
# You can check its status separately
```

**Check Task Status:**
```bash
python scripts/background_status.py --task-id training_1

# Output:
# Task: training_1
# Status: running
# Progress: 45.0%
# Elapsed: 123.5s
```

**Wait for Completion:**
```bash
python scripts/background_status.py \
    --task-id training_1 \
    --wait
```

### Background Batch Analysis

Analyze multiple positions in background:

```python
from src.engine.background_engine import BackgroundEngine

engine = BackgroundEngine()

# Analyze multiple positions
positions = [
    "0919293949596979891777062646668600102030405060708012720323436383",
    # ... more positions
]

task_id = engine.analyze_batch_background(
    positions=positions,
    depth=4,
    task_id="batch_analysis_1"
)

# Check results later
status = engine.get_task_status("batch_analysis_1")
results = status.result  # List of analysis results
```

### Task States

Tasks can have the following states:
- **pending**: Task is queued
- **running**: Task is currently executing
- **completed**: Task finished successfully
- **failed**: Task failed with an error
- **cancelled**: Task was cancelled

### Management Functions

```python
from src.utils.background import get_background_manager

manager = get_background_manager()

# List all tasks
tasks = manager.list_tasks()

# Cleanup completed tasks
manager.cleanup_task(task_id)

# Wait for specific timeout
result = manager.wait_for_task(task_id, timeout=60.0)
```

---

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
