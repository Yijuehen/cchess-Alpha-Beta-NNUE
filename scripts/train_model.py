#!/usr/bin/env python3
"""
Training script for NNUE model.

Trains a neural network evaluation function on Chinese Chess position data.
"""

import argparse
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train NNUE model on Chinese Chess data"
    )

    # Data arguments
    parser.add_argument(
        '--data',
        type=str,
        default='data/chess.csv',
        help='Path to training CSV file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='models/nnue_weights.pkl',
        help='Path to save trained model'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to use (for quick testing)'
    )

    # Model architecture
    parser.add_argument(
        '--input-size',
        type=int,
        default=1260,
        help='NNUE input size'
    )

    parser.add_argument(
        '--hidden-size',
        type=int,
        default=256,
        help='NNUE hidden layer size'
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Learning rate'
    )

    parser.add_argument(
        '--loss',
        type=str,
        default='mse',
        choices=['mse', 'huber', 'scaled_mse'],
        help='Loss function'
    )

    parser.add_argument(
        '--validation-ratio',
        type=float,
        default=0.1,
        help='Validation split ratio'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models/checkpoints',
        help='Directory to save checkpoints'
    )

    parser.add_argument(
        '--no-checkpoints',
        action='store_true',
        help='Disable checkpoint saving'
    )

    args = parser.parse_args()

    # Create checkpoint directory
    checkpoint_dir = None if args.no_checkpoints else args.checkpoint_dir

    # Train model
    print("=" * 60)
    print("NNUE Training")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Architecture: {args.input_size} → {args.hidden_size} → 1")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Loss function: {args.loss}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    print("=" * 60)
    print()

    try:
        network = train_model(
            csv_path=args.data,
            output_path=args.output,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            validation_ratio=args.validation_ratio,
            loss_fn=args.loss,
            checkpoint_dir=checkpoint_dir,
            verbose=True
        )

        print()
        print("=" * 60)
        print("Training completed successfully!")
        print(f"Model saved to: {args.output}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
