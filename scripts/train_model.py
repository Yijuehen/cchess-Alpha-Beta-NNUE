#!/usr/bin/env python3
"""
Training script for NNUE model.

Trains a neural network evaluation function on Chinese Chess position data.
Supports configuration files and command-line arguments.
"""

import argparse
from pathlib import Path
import yaml

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import train_model
from src.utils.logger import setup_logging, get_logger


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_with_args(config, args):
    """Merge config file with command-line arguments (CLI args take precedence)."""
    result = {}

    # Start with config file values
    if config:
        # Flatten nested config structure
        if 'data' in config:
            result.update({f'data_{k}': v for k, v in config['data'].items()})
        if 'training' in config:
            result.update({k: v for k, v in config['training'].items()})
        if 'model' in config:
            result.update(config['model'])
        if 'loss' in config:
            result['loss'] = config['loss']['type']
        if 'validation' in config:
            result['validation_ratio'] = config['validation']['ratio']
        if 'checkpointing' in config:
            result['no_checkpoints'] = not config['checkpointing']['enabled']
            if config['checkpointing']['enabled']:
                result['checkpoint_dir'] = config['checkpointing']['directory']
        if 'logging' in config:
            result['log_level'] = config['logging']['level']
            result['log_dir'] = config['logging']['log_dir']

    # Override with CLI args (convert argparse Namespace to dict, filtering out None values)
    args_dict = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}

    # Map CLI arg names to internal names
    arg_mapping = {
        'data': 'csv_path',
        'output': 'output_path',
    }

    for cli_key, internal_key in arg_mapping.items():
        if cli_key in args_dict:
            args_dict[internal_key] = args_dict.pop(cli_key)

    result.update(args_dict)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train NNUE model on Chinese Chess data",
        epilog="Example: python scripts/train_model.py --config config/test_config.yaml"
    )

    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file (overrides defaults, overridden by CLI args)'
    )

    # Data arguments
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to training CSV file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save trained model'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to use (for quick testing)'
    )

    # Data augmentation arguments
    parser.add_argument(
        '--augment',
        action='store_true',
        default=None,
        help='Enable data augmentation (vertical flip)'
    )

    parser.add_argument(
        '--no-augment',
        action='store_true',
        default=None,
        help='Disable data augmentation'
    )

    parser.add_argument(
        '--augment-prob',
        type=float,
        default=None,
        help='Probability of applying augmentation per sample (0.0-1.0)'
    )

    parser.add_argument(
        '--target-scale',
        type=float,
        default=None,
        help='Scale factor for targets (default: 100.0, converts ±10000 to ±100)'
    )

    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=None,
        help='Maximum gradient norm for clipping (default: 1.0)'
    )

    # Model architecture
    parser.add_argument(
        '--input-size',
        type=int,
        default=None,
        help='NNUE input size'
    )

    parser.add_argument(
        '--hidden-size',
        type=int,
        default=None,
        help='NNUE hidden layer size'
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Training batch size'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate'
    )

    parser.add_argument(
        '--loss',
        type=str,
        default=None,
        choices=['mse', 'huber', 'scaled_mse'],
        help='Loss function'
    )

    parser.add_argument(
        '--validation-ratio',
        type=float,
        default=None,
        help='Validation split ratio'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory to save checkpoints'
    )

    parser.add_argument(
        '--no-checkpoints',
        action='store_true',
        default=None,
        help='Disable checkpoint saving'
    )

    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        default=None,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Log directory'
    )

    args = parser.parse_args()

    # Load configuration
    config = None
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")

    # Merge config with CLI args (CLI args take precedence)
    merged = merge_config_with_args(config, args)

    # Apply defaults for any missing values
    defaults = {
        'csv_path': 'data/chess.csv',
        'output_path': 'models/nnue_weights.pkl',
        'input_size': 1260,
        'hidden_size': 256,
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.01,
        'loss': 'mse',
        'validation_ratio': 0.1,
        'checkpoint_dir': 'models/checkpoints',
        'no_checkpoints': False,
        'log_level': 'INFO',
        'log_dir': 'logs/training',
        'max_samples': None,
        'augment': False,
        'augment_prob': 0.5,
        'target_scale': 100.0,
        'max_grad_norm': 1.0,
    }

    for key, value in defaults.items():
        if key not in merged or merged[key] is None:
            merged[key] = value

    # Setup logging
    setup_logging(
        name="training",
        log_dir=merged['log_dir'],
        level=merged['log_level']
    )
    logger = get_logger("training")

    # Create checkpoint directory
    checkpoint_dir = None if merged['no_checkpoints'] else merged['checkpoint_dir']

    # Log configuration
    logger.info("=" * 60)
    logger.info("NNUE Training Started")
    logger.info("=" * 60)
    logger.info(f"Config source: {'Config file' if config else 'Command line/defaults'}")
    logger.info(f"Data: {merged['csv_path']}")
    logger.info(f"Output: {merged['output_path']}")
    logger.info(f"Architecture: {merged['input_size']} → {merged['hidden_size']} → 1")
    logger.info(f"Epochs: {merged['epochs']}")
    logger.info(f"Batch size: {merged['batch_size']}")
    logger.info(f"Learning rate: {merged['learning_rate']}")
    logger.info(f"Loss function: {merged['loss']}")
    logger.info(f"Max samples: {merged['max_samples'] if merged['max_samples'] else 'All'}")
    logger.info(f"Data augmentation: {'Enabled' if merged['augment'] else 'Disabled'}")
    if merged['augment']:
        logger.info(f"Augmentation probability: {merged['augment_prob']}")
    logger.info("=" * 60)

    print()
    print("Starting training...")
    print(f"Data: {merged['csv_path']}")
    print(f"Output: {merged['output_path']}")
    print()

    try:
        network = train_model(
            csv_path=merged['csv_path'],
            output_path=merged['output_path'],
            input_size=merged['input_size'],
            hidden_size=merged['hidden_size'],
            learning_rate=merged['learning_rate'],
            target_scale=merged['target_scale'],
            max_grad_norm=merged['max_grad_norm'],
            epochs=merged['epochs'],
            batch_size=merged['batch_size'],
            max_samples=merged['max_samples'],
            validation_ratio=merged['validation_ratio'],
            loss_fn=merged['loss'],
            checkpoint_dir=checkpoint_dir,
            augment=merged['augment'],
            augment_prob=merged['augment_prob'],
            verbose=True
        )

        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {merged['output_path']}")
        logger.info("=" * 60)

        print()
        print("=" * 60)
        print("Training completed successfully!")
        print(f"Model saved to: {merged['output_path']}")
        print(f"Logs: {merged['log_dir']}/training.log")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        print("\nTraining interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
