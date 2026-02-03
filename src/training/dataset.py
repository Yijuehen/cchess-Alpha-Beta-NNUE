"""
Dataset module for loading CSV training data.

Provides efficient data loading for the 1.9M position dataset.
"""

import numpy as np
import csv
from typing import Iterator, Tuple, Optional
from pathlib import Path


class ChessDataset:
    """
    Dataset for Chinese Chess positions from CSV file.

    CSV columns:
    - chessboradStatus: 64-character board encoding
    - moveStatus: 4-digit move encoding
    - chessboradStatusAfterMoving: Board after move
    - predictAction: Predicted next action
    - result: 0=Red wins, 1=Black wins, 2=Draw, None=incomplete
    """

    def __init__(
        self,
        csv_path: str,
        has_result_only: bool = False,
        shuffle: bool = True
    ):
        """
        Initialize dataset.

        Args:
            csv_path: Path to CSV file
            has_result_only: Only include samples with game results
            shuffle: Shuffle the data
        """
        self.csv_path = Path(csv_path)
        self.has_result_only = has_result_only
        self.shuffle = shuffle
        self._length = None

    def _load_row(self, row: list) -> Optional[Tuple[str, int]]:
        """
        Load a single row and extract (board_str, result).

        Args:
            row: CSV row as list

        Returns:
            Tuple of (board_str, result) or None if filtered out
        """
        if len(row) < 5:
            return None

        board_str = row[0]
        result_str = row[4]

        # Filter samples without results if needed
        if self.has_result_only and result_str == 'None':
            return None

        # Parse result
        if result_str == 'None':
            result = None
        else:
            result = int(result_str)

        return board_str, result

    def __iter__(self) -> Iterator[Tuple[str, Optional[int]]]:
        """Iterate over dataset."""
        with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)

            for row in reader:
                data = self._load_row(row)
                if data is not None:
                    yield data

    def __len__(self) -> int:
        """Get dataset length (cached)."""
        if self._length is None:
            count = 0
            for _ in self:
                count += 1
            self._length = count
        return self._length


class BatchDataLoader:
    """
    Batch data loader for training with data augmentation support.

    Yields batches of (features, targets) for training.
    """

    def __init__(
        self,
        csv_path: str,
        batch_size: int = 32,
        has_result_only: bool = True,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        augment: bool = False,
        augment_prob: float = 0.5,
        target_scale: float = 100.0
    ):
        """
        Initialize data loader.

        Args:
            csv_path: Path to CSV file
            batch_size: Batch size
            has_result_only: Only use samples with results
            max_samples: Maximum number of samples to use
            shuffle: Shuffle data
            augment: Apply data augmentation (vertical flip)
            augment_prob: Probability of augmentation per sample
            target_scale: Scale factor for targets (converts ±10000 to ±scale)
        """
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.has_result_only = has_result_only
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.augment = augment
        self.augment_prob = augment_prob
        self.target_scale = target_scale

        # Import augmentation module if needed
        if self.augment:
            from .data_augmentation import DataAugmentor, flip_board_vertical
            self.augmentor = DataAugmentor(vertical_flip_prob=augment_prob)
            self.flip_board_vertical = flip_board_vertical

    def result_to_score(self, result: int, perspective: int = 1) -> float:
        """
        Convert result to evaluation score (scaled for stable training).

        Args:
            result: 0=Red wins, 1=Black wins, 2=Draw
            perspective: 1=Red, -1=Black

        Returns:
            Score in centipawns, then scaled by target_scale
            (e.g., with target_scale=100, returns ±100 instead of ±10000)
        """
        if result == 2:  # Draw
            raw_score = 0.0
        elif result == 0:  # Red wins
            raw_score = 10000.0 if perspective == 1 else -10000.0
        else:  # Black wins
            raw_score = -10000.0 if perspective == 1 else 10000.0

        # Scale target for stable training (prevents gradient explosion)
        return raw_score / self.target_scale

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over batches.

        Yields:
            Tuple of (features, targets) arrays
        """
        from ..board.board_representation import parse_board
        from ..features.feature_transformer import extract_features

        batch_features = []
        batch_targets = []
        sample_count = 0

        with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) < 5:
                    continue

                board_str = row[0]
                result_str = row[4]

                # Skip samples without results
                if self.has_result_only and result_str == 'None':
                    continue

                # Parse result
                result = int(result_str) if result_str != 'None' else None
                if result is None:
                    continue

                # Convert to score
                target = self.result_to_score(result)

                # Parse board and extract features
                try:
                    board = parse_board(board_str)

                    # Apply augmentation if enabled
                    if self.augment and np.random.random() < self.augment_prob:
                        import random
                        # Flip board (Red <-> Black)
                        board = self.flip_board_vertical(board)
                        # Flip target score
                        target = -target

                    features = extract_features(board)
                except Exception as e:
                    # Skip invalid positions
                    continue

                batch_features.append(features)
                batch_targets.append(target)
                sample_count += 1

                # Yield batch when full
                if len(batch_features) >= self.batch_size:
                    yield (
                        np.array(batch_features, dtype=np.float32),
                        np.array(batch_targets, dtype=np.float32)
                    )
                    batch_features = []
                    batch_targets = []

                # Check max samples
                if self.max_samples and sample_count >= self.max_samples:
                    break

        # Yield final batch if it's complete
        if batch_features and len(batch_features) >= self.batch_size:
            yield (
                np.array(batch_features[:self.batch_size], dtype=np.float32),
                np.array(batch_targets[:self.batch_size], dtype=np.float32)
            )
        # Note: We drop incomplete batches to avoid dimension mismatch
        # If you get "no batches yielded" error, reduce batch_size or increase data


def create_train_validation_split(
    csv_path: str,
    validation_ratio: float = 0.1,
    shuffle: bool = True
) -> Tuple[str, str]:
    """
    Split CSV data into training and validation sets.

    Creates temporary CSV files for train and validation.

    Args:
        csv_path: Path to original CSV
        validation_ratio: Ratio of validation data
        shuffle: Shuffle before splitting

    Returns:
        Tuple of (train_csv_path, val_csv_path)
    """
    import tempfile
    import random

    # Read all rows
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header if present
        for row in reader:
            rows.append(row)

    if shuffle:
        random.shuffle(rows)

    # Split
    split_idx = int(len(rows) * (1 - validation_ratio))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    # Create temporary files
    train_fd, train_path = tempfile.mkstemp(suffix='.csv')
    val_fd, val_path = tempfile.mkstemp(suffix='.csv')

    # Write train data
    with open(train_fd, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(train_rows)

    # Write validation data
    with open(val_fd, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(val_rows)

    return train_path, val_path
