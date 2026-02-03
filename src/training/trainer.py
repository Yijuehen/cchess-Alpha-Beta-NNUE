"""
Training script for NNUE model.

Trains NNUE network on Chinese Chess position data.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Callable
from tqdm import tqdm

from ..nnue.nue_network import NNUE, NNUETrainer
from ..utils.logger import get_logger
from .dataset import BatchDataLoader, create_train_validation_split
from .loss import mse_loss, get_loss_function

logger = get_logger("training")


class Trainer:
    """
    High-level trainer for NNUE network.

    Handles training loop, validation, and checkpointing.
    """

    def __init__(
        self,
        network: NNUE,
        learning_rate: float = 0.01,
        loss_fn: str = 'mse',
        device: str = 'cpu',
        target_scale: float = 100.0,
        max_grad_norm: float = 1.0
    ):
        """
        Initialize trainer.

        Args:
            network: NNUE network to train
            learning_rate: Learning rate
            loss_fn: Loss function name
            device: Device to use ('cpu' or 'cuda')
            target_scale: Scale factor for training targets
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.network = network
        self.trainer = NNUETrainer(network, learning_rate)
        self.loss_fn_name = loss_fn
        self.loss_fn, self.loss_grad_fn = get_loss_function(loss_fn)
        self.target_scale = target_scale
        self.max_grad_norm = max_grad_norm
        self.logger = get_logger("training")

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        self.logger.info(
            f"Initialized trainer: lr={learning_rate}, "
            f"loss_fn={loss_fn}, device={device}, "
            f"target_scale={target_scale}, max_grad_norm={max_grad_norm}"
        )

    def compute_validation_loss(
        self,
        val_features: np.ndarray,
        val_targets: np.ndarray
    ) -> float:
        """Compute validation loss."""
        self.logger.debug(
            f"Computing validation loss: {len(val_features)} samples"
        )
        predictions = self.network.forward_batch(val_features)
        loss = self.loss_fn(predictions, val_targets)
        self.logger.debug(f"Validation loss: {loss:.4f}")
        return loss

    def train(
        self,
        train_csv_path: str,
        val_csv_path: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 32,
        validation_ratio: float = 0.1,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = 5,
        verbose: bool = True,
        augment: bool = False,
        augment_prob: float = 0.5
    ) -> dict:
        """
        Train the network.

        Args:
            train_csv_path: Path to training CSV
            val_csv_path: Path to validation CSV (if None, will split from train)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_ratio: Validation split ratio (if no val CSV)
            max_train_samples: Max training samples to use
            max_val_samples: Max validation samples to use
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
            verbose: Print progress
            augment: Apply data augmentation (vertical flip)
            augment_prob: Probability of applying augmentation per sample

        Returns:
            Training history dictionary
        """
        # Create validation split if needed
        if val_csv_path is None:
            if verbose:
                self.logger.info("Creating train/validation split...")
            train_csv_path, val_csv_path = create_train_validation_split(
                train_csv_path, validation_ratio
            )
            self.logger.info(f"Split created: train={train_csv_path}, val={val_csv_path}")

        # Create data loaders with augmentation and target scaling
        train_loader = BatchDataLoader(
            train_csv_path,
            batch_size=batch_size,
            has_result_only=True,
            max_samples=max_train_samples,
            augment=augment,
            augment_prob=augment_prob,
            target_scale=self.target_scale
        )

        val_loader = BatchDataLoader(
            val_csv_path,
            batch_size=min(batch_size, 16),  # Use smaller batch for validation
            has_result_only=True,
            max_samples=max_val_samples,
            augment=False,  # Don't augment validation data
            target_scale=self.target_scale
        )

        if augment:
            self.logger.info(f"Data augmentation enabled: vertical flip (p={augment_prob})")

        # Load validation data into memory
        if verbose:
            self.logger.info("Loading validation data...")

        val_features_list = []
        val_targets_list = []
        for batch_features, batch_targets in val_loader:
            val_features_list.append(batch_features)
            val_targets_list.append(batch_targets)

        if val_features_list:
            val_features = np.vstack(val_features_list)
            val_targets = np.concatenate(val_targets_list)
            self.logger.info(f"Validation data loaded: {len(val_features)} samples")
        else:
            val_features = None
            val_targets = None
            self.logger.warning("No validation data available - validation will be skipped")

        # Training loop
        self.logger.info("=" * 60)
        self.logger.info(f"Starting training: {epochs} epochs")
        self.logger.info(f"Learning rate: {self.trainer.learning_rate}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Loss function: {self.loss_fn_name}")
        self.logger.info("=" * 60)

        patience_counter = 0

        for epoch in range(epochs):
            # Training
            epoch_losses = []
            self.logger.info(f"Epoch {epoch+1}/{epochs} started")
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader

            for batch_idx, (batch_features, batch_targets) in enumerate(train_iterator):
                loss = self.trainer.update_step(
                    batch_features,
                    batch_targets,
                    max_grad_norm=self.max_grad_norm
                )
                epoch_losses.append(loss)

                # Log batch progress periodically
                if batch_idx > 0 and batch_idx % 100 == 0:
                    self.logger.debug(f"  Batch {batch_idx}: loss={loss:.4f}")

            # Compute average training loss
            train_loss = np.mean(epoch_losses)
            self.train_losses.append(train_loss)

            self.logger.info(f"Epoch {epoch+1}/{epochs} completed: train_loss={train_loss:.4f}")

            # Validation
            if val_features is not None:
                val_loss = self.compute_validation_loss(val_features, val_targets)
                self.val_losses.append(val_loss)

                self.logger.info(
                    f"  Validation: val_loss={val_loss:.4f}, "
                    f"best={self.best_val_loss:.4f}"
                )

                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0

                    # Save checkpoint
                    if checkpoint_dir:
                        checkpoint_path = Path(checkpoint_dir) / "best_model.pkl"
                        self.network.save(str(checkpoint_path))
                        self.logger.info(
                            f"  → Saved best model (val_loss={val_loss:.4f})"
                        )
                else:
                    patience_counter += 1
                    self.logger.debug(
                        f"  No improvement ({patience_counter}/{early_stopping_patience})"
                    )

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch+1}"
                    )
                    break
            else:
                self.val_losses.append(train_loss)

        self.logger.info("=" * 60)
        self.logger.info("Training completed")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info("=" * 60)

        # Save final model
        if checkpoint_dir:
            final_path = Path(checkpoint_dir) / "final_model.pkl"
            self.network.save(str(final_path))
            self.logger.info(f"Saved final model to {final_path}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }


def train_model(
    csv_path: str,
    output_path: str,
    input_size: int = 1260,
    hidden_size: int = 256,
    learning_rate: float = 0.01,
    target_scale: float = 100.0,
    max_grad_norm: float = 1.0,
    epochs: int = 10,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    validation_ratio: float = 0.1,
    loss_fn: str = 'mse',
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
    augment: bool = False,
    augment_prob: float = 0.5
) -> NNUE:
    """
    Train NNUE model on CSV data.

    Args:
        csv_path: Path to training CSV
        output_path: Path to save trained model
        input_size: NNUE input size
        hidden_size: NNUE hidden size
        learning_rate: Learning rate
        target_scale: Scale factor for targets (converts ±10000 to ±scale)
        max_grad_norm: Maximum gradient norm for clipping
        epochs: Number of epochs
        batch_size: Batch size
        max_samples: Max samples to use (for quick testing)
        validation_ratio: Validation split ratio
        loss_fn: Loss function
        checkpoint_dir: Checkpoint directory
        verbose: Print progress
        augment: Apply data augmentation (vertical flip)
        augment_prob: Probability of applying augmentation per sample

    Returns:
        Trained NNUE network
    """
    # Create network
    network = NNUE(input_size=input_size, hidden_size=hidden_size)
    network.target_scale = target_scale  # Store scale in network for inference

    # Create trainer
    trainer = Trainer(
        network,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        target_scale=target_scale,
        max_grad_norm=max_grad_norm
    )

    # Create checkpoint directory
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Train
    history = trainer.train(
        train_csv_path=csv_path,
        epochs=epochs,
        batch_size=batch_size,
        max_train_samples=max_samples,
        validation_ratio=validation_ratio,
        checkpoint_dir=checkpoint_dir,
        verbose=verbose,
        augment=augment,
        augment_prob=augment_prob
    )

    # Save final model
    network.save(output_path)

    # Save final model
    network.save(output_path)
    logger.info(f"Model saved to: {output_path}")

    if verbose:
        print(f"\nTraining complete!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Model saved to: {output_path}")

    return network
