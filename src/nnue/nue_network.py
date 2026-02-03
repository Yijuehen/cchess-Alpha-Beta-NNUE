"""
NNUE (Neural Network User Evaluation) Network Architecture

Shallow neural network for fast position evaluation.
Uses SCReLU (Squared Clipped ReLU) for quantization-friendly activation.

Architecture: 1,260 features → 256 hidden → 1 output
"""

import numpy as np
from typing import Tuple, Optional
import pickle


# SCReLU activation function
def screlu(x: np.ndarray) -> np.ndarray:
    """
    Shifted Clipped ReLU activation function.

    SCReLU(x) = clip(x, 0, 127)

    This is quantization-friendly and commonly used in NNUE.

    Args:
        x: Input array

    Returns:
        Clipped array with values in [0, 127]
    """
    return np.clip(x, 0, 127)


def screlu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of SCReLU for training.

    d/dx SCReLU(x) = 1 if 0 < x < 127, else 0

    Args:
        x: Input array

    Returns:
        Derivative array
    """
    return ((x > 0) & (x < 127)).astype(np.float32)


class NNUE:
    """
    NNUE (Neural Network User Evaluation) network.

    Shallow network for fast position evaluation in chess engines.

    Architecture:
        Input (1260) → Hidden (256, SCReLU) → Output (1)
    """

    def __init__(
        self,
        input_size: int = 1260,
        hidden_size: int = 256,
        random_init: bool = True,
        target_scale: float = 100.0
    ):
        """
        Initialize NNUE network.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden neurons
            random_init: Whether to randomly initialize weights
            target_scale: Expected scale of target values (for better initialization)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_scale = target_scale

        # Initialize weights
        if random_init:
            # Xavier initialization
            scale1 = np.sqrt(2.0 / (input_size + hidden_size))
            self.W1 = np.random.randn(input_size, hidden_size).astype(np.float32) * scale1
            scale2 = np.sqrt(2.0 / (hidden_size + 1))
            self.W2 = np.random.randn(hidden_size, 1).astype(np.float32) * scale2

            # Initialize output bias to match target scale (critical for convergence!)
            # This sets initial predictions closer to the target range
            self.b1 = np.zeros(hidden_size, dtype=np.float32)
            self.b2 = np.zeros(1, dtype=np.float32)
        else:
            self.W1 = np.zeros((input_size, hidden_size), dtype=np.float32)
            self.W2 = np.zeros((hidden_size, 1), dtype=np.float32)
            self.b1 = np.zeros(hidden_size, dtype=np.float32)
            self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, features: np.ndarray) -> float:
        """
        Forward pass through the network.

        Args:
            features: Input feature vector of shape (input_size,)

        Returns:
            Evaluation score (centipawns)
        """
        # Hidden layer
        hidden = features @ self.W1 + self.b1
        hidden = screlu(hidden)

        # Output layer
        output = hidden @ self.W2 + self.b2

        return float(output[0])

    def forward_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass for a batch of features.

        Args:
            features: Input feature array of shape (batch_size, input_size)

        Returns:
            Evaluation scores of shape (batch_size,)
        """
        # Hidden layer
        hidden = features @ self.W1 + self.b1
        hidden = screlu(hidden)

        # Output layer
        output = hidden @ self.W2 + self.b2

        return output.flatten()

    def get_weights(self) -> dict:
        """Get network weights as dictionary."""
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'target_scale': getattr(self, 'target_scale', 1.0),  # For backward compatibility
        }

    def set_weights(self, weights: dict) -> None:
        """Set network weights from dictionary."""
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']
        self.input_size = weights['input_size']
        self.hidden_size = weights['hidden_size']
        self.target_scale = weights.get('target_scale', 1.0)  # Backward compatible

    def save(self, filepath: str) -> None:
        """
        Save network weights to file.

        Args:
            filepath: Path to save weights
        """
        weights = self.get_weights()
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)

    def load(self, filepath: str) -> None:
        """
        Load network weights from file.

        Args:
            filepath: Path to load weights from
        """
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        self.set_weights(weights)

    def clone(self) -> 'NNUE':
        """Create a deep copy of the network."""
        new_net = NNUE(self.input_size, self.hidden_size, random_init=False)
        new_net.W1 = self.W1.copy()
        new_net.b1 = self.b1.copy()
        new_net.W2 = self.W2.copy()
        new_net.b2 = self.b2.copy()
        return new_net

    def __call__(self, features: np.ndarray) -> float:
        """Allow calling network as function."""
        return self.forward(features)


class NNUETrainer:
    """
    Training utilities for NNUE network.

    Supports gradient descent training with backpropagation.
    """

    def __init__(self, network: NNUE, learning_rate: float = 0.01):
        """
        Initialize trainer.

        Args:
            network: NNUE network to train
            learning_rate: Learning rate for gradient descent
        """
        self.network = network
        self.learning_rate = learning_rate

    def compute_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute MSE loss.

        Args:
            predictions: Predicted scores
            targets: Target scores

        Returns:
            MSE loss value
        """
        return float(np.mean((predictions - targets) ** 2))

    def compute_loss_gradient(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of MSE loss.

        dL/dy = 2 * (predictions - targets) / n

        Args:
            predictions: Predicted scores
            targets: Target scores

        Returns:
            Gradient w.r.t. predictions
        """
        n = len(predictions)
        return 2 * (predictions - targets) / n

    def backward(
        self,
        features: np.ndarray,
        loss_grad: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass to compute gradients.

        Args:
            features: Input features of shape (batch_size, input_size)
            loss_grad: Gradient of loss w.r.t. output

        Returns:
            Tuple of (dW1, db1, dW2, db2) gradients
        """
        batch_size = features.shape[0]

        # Forward pass (cached for backward)
        hidden_pre = features @ self.network.W1 + self.network.b1  # (batch, hidden)
        hidden = screlu(hidden_pre)  # (batch, hidden)

        # Output gradients
        # dL/dW2 = hidden.T @ dL/doutput
        dW2 = hidden.T @ loss_grad.reshape(-1, 1)  # (hidden, 1)
        db2 = np.sum(loss_grad)  # scalar

        # Hidden layer gradients
        # dL/dhidden = dL/doutput @ W2.T * screlu'
        # Reshape loss_grad to (batch, 1) for proper matrix multiplication
        dhidden = (loss_grad.reshape(-1, 1) @ self.network.W2.T) * screlu_derivative(hidden_pre)  # (batch, hidden)

        # Input gradients
        # dL/dW1 = features.T @ dhidden
        dW1 = features.T @ dhidden  # (input, hidden)
        db1 = np.sum(dhidden, axis=0)  # (hidden,)

        return dW1, db1, dW2, db2

    def update_step(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        max_grad_norm: float = 1.0
    ) -> float:
        """
        Perform one gradient descent update step with gradient clipping.

        Args:
            features: Input features of shape (batch_size, input_size)
            targets: Target scores
            max_grad_norm: Maximum L2 norm for gradients (None = no clipping)

        Returns:
            Loss value
        """
        # Forward pass
        predictions = self.network.forward_batch(features)

        # Compute loss and gradients
        loss = self.compute_loss(predictions, targets)
        loss_grad = self.compute_loss_gradient(predictions, targets)

        # Backward pass
        dW1, db1, dW2, db2 = self.backward(features, loss_grad)

        # Gradient clipping by L2 norm (prevents explosion)
        if max_grad_norm is not None:
            grad_norm = np.sqrt(
                np.sum(dW1 ** 2) + np.sum(db1 ** 2) +
                np.sum(dW2 ** 2) + np.sum(db2 ** 2)
            )
            if grad_norm > max_grad_norm:
                scale = max_grad_norm / grad_norm
                dW1 *= scale
                db1 *= scale
                dW2 *= scale
                db2 *= scale

        # Update weights
        self.network.W1 -= self.learning_rate * dW1
        self.network.b1 -= self.learning_rate * db1
        self.network.W2 -= self.learning_rate * dW2
        self.network.b2 -= self.learning_rate * db2

        return loss

    def train_epoch(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 32
    ) -> float:
        """
        Train for one epoch.

        Args:
            features: Input features of shape (num_samples, input_size)
            targets: Target scores
            batch_size: Mini-batch size

        Returns:
            Average loss over epoch
        """
        num_samples = features.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        total_loss = 0.0

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            batch_features = features[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]

            loss = self.update_step(batch_features, batch_targets)
            total_loss += loss

        return total_loss / num_batches


def create_nnue_from_config(config: dict) -> NNUE:
    """
    Create NNUE network from configuration dictionary.

    Args:
        config: Configuration dict with 'input_size' and 'hidden_size'

    Returns:
        Initialized NNUE network
    """
    input_size = config.get('input_size', 1260)
    hidden_size = config.get('hidden_size', 256)
    return NNUE(input_size, hidden_size)
