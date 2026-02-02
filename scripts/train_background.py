#!/usr/bin/env python3
"""
Train NNUE model in background.

Starts training as a background task that can be monitored separately.
Supports configuration files and command-line arguments.
"""

import argparse
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.background_trainer import (
    train_model_background,
    check_training_status,
    wait_for_training
)
from src.utils.logger import setup_logging, get_logger


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train NNUE model in background",
        epilog="Example: python scripts/train_background.py --config config/test_config.yaml"
    )

    parser.add_argument('--config', help='Path to YAML configuration file')
    parser.add_argument('--data', help='Path to training CSV')
    parser.add_argument('--output', help='Path to save model')
    parser.add_argument('--task-id', default='training_job_1', help='Task ID')
    parser.add_argument('--wait', action='store_true', help='Wait for completion')
    parser.add_argument('--poll-interval', type=float, default=5.0, help='Poll interval (seconds)')

    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--max-samples', type=int, help='Max samples to use')
    parser.add_argument('--validation-ratio', type=float, help='Validation ratio')

    # Logging
    parser.add_argument('--log-level', help='Log level')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")

    # Merge config with CLI args (CLI args take precedence)
    def get_value(key, default=None):
        # Check CLI args first (non-None values)
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            return cli_val

        # Then check config
        if config:
            # Map config keys to arg names
            config_map = {
                'data': ('data', 'csv_path'),
                'output': ('data', 'output_path'),
                'epochs': ('training', 'epochs'),
                'batch_size': ('training', 'batch_size'),
                'learning_rate': ('training', 'learning_rate'),
                'max_samples': ('training', 'max_samples'),
                'validation_ratio': ('validation', 'ratio'),
                'log_level': ('logging', 'level'),
            }

            if key in config_map:
                section, cfg_key = config_map[key]
                if section in config and cfg_key in config[section]:
                    return config[section][cfg_key]

        return default

    # Get all values with defaults
    data = get_value('data', 'data/chess.csv')
    output = get_value('output', 'models/nnue_weights.pkl')
    task_id = args.task_id
    wait = args.wait
    poll_interval = args.poll_interval
    epochs = get_value('epochs', 10)
    batch_size = get_value('batch_size', 32)
    learning_rate = get_value('learning_rate', 0.01)
    max_samples = get_value('max_samples', None)
    validation_ratio = get_value('validation_ratio', 0.1)
    log_level = get_value('log_level', 'INFO')

    # Setup logging
    setup_logging(
        name="background_training",
        log_dir="logs/training",
        level=log_level
    )
    logger = get_logger("background_training")

    logger.info("=" * 60)
    logger.info("Background NNUE Training")
    logger.info("=" * 60)
    logger.info(f"Task ID: {task_id}")
    logger.info(f"Data: {data}")
    logger.info(f"Output: {output}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Max samples: {max_samples if max_samples else 'All'}")
    logger.info("=" * 60)

    try:
        # Start training
        train_model_background(
            csv_path=data,
            output_path=output,
            task_id=task_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_samples=max_samples,
            validation_ratio=validation_ratio
        )

        logger.info(f"Training started in background: {task_id}")
        print(f"\nTraining task submitted: {task_id}")
        print(f"Check status with: python scripts/background_status.py --task-id {task_id}")

        if wait:
            logger.info("Waiting for training to complete...")

            while True:
                status = check_training_status(task_id)

                if status.status.value in ['completed', 'failed', 'cancelled']:
                    break

                progress = status.progress * 100
                logger.info(f"Progress: {progress:.1f}%")
                print(f"Progress: {progress:.1f}%")

                time.sleep(poll_interval)

            # Get final result
            result = wait_for_training(task_id)

            if result is not None:
                logger.info("Training completed successfully!")
                print("\nTraining completed successfully!")
            else:
                logger.error("Training failed")
                print("\nTraining failed")
                return 1

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
