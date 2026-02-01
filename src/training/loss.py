"""
Loss functions for NNUE training.

Includes MSE, WDL (Win-Draw-Loss), and custom loss variants.
"""

import numpy as np
from typing import Tuple


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean Squared Error loss.

    L = (1/n) * Σ(predictions - targets)²

    Args:
        predictions: Predicted scores
        targets: Target scores

    Returns:
        MSE loss value
    """
    return float(np.mean((predictions - targets) ** 2))


def mse_loss_gradient(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE loss.

    dL/dy = (2/n) * (predictions - targets)

    Args:
        predictions: Predicted scores
        targets: Target scores

    Returns:
        Gradient with respect to predictions
    """
    n = len(predictions)
    return (2.0 / n) * (predictions - targets)


def wdl_to_score(w: float, d: float, l: float, eps: float = 1e-3) -> float:
    """
    Convert WDL (Win-Draw-Loss) probabilities to centipawn score.

    Uses the logistic model:
    score = 1000 * ln((w + d/2) / (l + d/2))

    Args:
        w: Win probability
        d: Draw probability
        l: Loss probability
        eps: Small value to avoid log(0)

    Returns:
        Score in centipawns
    """
    expected = w + d / 2.0
    unexpected = l + d / 2.0
    return 1000.0 * np.log(expected / (unexpected + eps) + eps)


def score_to_wdl(score: float) -> Tuple[float, float, float]:
    """
    Convert centipawn score to WDL probabilities.

    Uses the logistic model:
    w = 1 / (1 + 10^(-score/400))

    Args:
        score: Score in centipawns

    Returns:
        Tuple of (win, draw, loss) probabilities
    """
    # Convert score to win probability using logistic function
    win_prob = 1.0 / (1.0 + 10.0 ** (-score / 400.0))

    # Simplified WDL model (no draw probability for now)
    win = win_prob
    draw = 0.0
    loss = 1.0 - win_prob

    return win, draw, loss


def cross_entropy_loss(
    predictions: np.ndarray,
    wdl_targets: np.ndarray
) -> float:
    """
    Cross-entropy loss for WDL predictions.

    Args:
        predictions: Predicted scores
        wdl_targets: Target WDL probabilities of shape (n, 3)

    Returns:
        Cross-entropy loss
    """
    # Convert predictions to WDL
    wdl_preds = np.array([score_to_wdl(s) for s in predictions])

    # Compute cross-entropy
    # Avoid log(0) with clipping
    wdl_preds = np.clip(wdl_preds, 1e-7, 1 - 1e-7)
    loss = -np.sum(wdl_targets * np.log(wdl_preds)) / len(predictions)

    return float(loss)


def huber_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    delta: float = 100.0
) -> float:
    """
    Huber loss, less sensitive to outliers than MSE.

    L = {
        0.5 * (y - t)²              if |y - t| <= delta
        delta * (|y - t| - 0.5 * delta)  otherwise
    }

    Args:
        predictions: Predicted scores
        targets: Target scores
        delta: Threshold for quadratic vs linear loss

    Returns:
        Huber loss value
    """
    error = predictions - targets
    abs_error = np.abs(error)

    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic

    loss = 0.5 * quadratic ** 2 + delta * linear
    return float(np.mean(loss))


def huber_loss_gradient(
    predictions: np.ndarray,
    targets: np.ndarray,
    delta: float = 100.0
) -> np.ndarray:
    """
    Gradient of Huber loss.

    Args:
        predictions: Predicted scores
        targets: Target scores
        delta: Threshold for quadratic vs linear loss

    Returns:
        Gradient with respect to predictions
    """
    error = predictions - targets
    abs_error = np.abs(error)

    # Quadratic region: gradient = error
    # Linear region: gradient = delta * sign(error)
    grad = np.where(abs_error <= delta, error, delta * np.sign(error))

    return grad / len(predictions)


def scaled_mse_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    scale: float = 100.0
) -> float:
    """
    Scaled MSE loss for better gradient flow.

    L = (1/n) * Σ((predictions - targets) / scale)²

    Args:
        predictions: Predicted scores
        targets: Target scores
        scale: Scaling factor

    Returns:
        Scaled MSE loss
    """
    scaled_error = (predictions - targets) / scale
    return float(np.mean(scaled_error ** 2))


def scaled_mse_loss_gradient(
    predictions: np.ndarray,
    targets: np.ndarray,
    scale: float = 100.0
) -> np.ndarray:
    """
    Gradient of scaled MSE loss.

    Args:
        predictions: Predicted scores
        targets: Target scores
        scale: Scaling factor

    Returns:
        Gradient with respect to predictions
    """
    error = predictions - targets
    n = len(predictions)
    return (2.0 / (scale ** 2 * n)) * error


def get_loss_function(name: str):
    """
    Get loss function by name.

    Args:
        name: Loss function name ('mse', 'huber', 'scaled_mse')

    Returns:
        Tuple of (loss_fn, gradient_fn)
    """
    losses = {
        'mse': (mse_loss, mse_loss_gradient),
        'huber': (huber_loss, huber_loss_gradient),
        'scaled_mse': (scaled_mse_loss, scaled_mse_loss_gradient),
    }

    if name not in losses:
        raise ValueError(f"Unknown loss function: {name}")

    return losses[name]
