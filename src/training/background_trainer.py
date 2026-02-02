"""
Background training support for NNUE model.

Allows training to run in background threads with progress tracking.
"""

import time
from typing import Optional, Callable

from ..utils.background import get_background_manager, TaskStatus
from ..utils.logger import get_logger
from .trainer import train_model

logger = get_logger("background_training")


def train_model_background(
    csv_path: str,
    output_path: str,
    task_id: str,
    progress_callback: Optional[Callable[[float], None]] = None,
    **training_kwargs
) -> str:
    """
    Train model in background.

    Args:
        csv_path: Path to training data
        output_path: Path to save model
        task_id: Unique task identifier
        progress_callback: Callback for progress updates (0.0 to 1.0)
        **training_kwargs: Additional training arguments

    Returns:
        Task ID
    """
    def training_wrapper():
        logger.info(f"Background training task started: {task_id}")
        logger.info(f"Data: {csv_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Epochs: {training_kwargs.get('epochs', 10)}")

        # Run training
        network = train_model(
            csv_path=csv_path,
            output_path=output_path,
            verbose=False,  # Disable print, use logging
            **training_kwargs
        )

        logger.info(f"Background training task completed: {task_id}")
        return network

    # Submit to background manager
    manager = get_background_manager()
    manager.submit_task(
        task_id=task_id,
        func=training_wrapper,
        progress_callback=progress_callback
    )

    logger.info(f"Background training submitted: {task_id}")
    return task_id


def check_training_status(task_id: str):
    """
    Check status of background training.

    Args:
        task_id: Task identifier

    Returns:
        TaskResult or None if not found
    """
    manager = get_background_manager()
    task = manager.get_task_status(task_id)

    if task:
        logger.info(
            f"Task {task_id}: status={task.status.value}, "
            f"progress={task.progress:.1%}, elapsed={task.elapsed_time():.1f}s"
        )
        return task
    else:
        logger.warning(f"Task not found: {task_id}")
        return None


def wait_for_training(task_id: str, timeout: Optional[float] = None):
    """
    Wait for background training to complete.

    Args:
        task_id: Task identifier
        timeout: Optional timeout in seconds

    Returns:
        Trained network or None
    """
    manager = get_background_manager()
    task = manager.wait_for_task(task_id, timeout=timeout)

    if task.status == TaskStatus.COMPLETED:
        logger.info(f"Training completed successfully: {task_id}")
        return task.result
    elif task.status == TaskStatus.FAILED:
        logger.error(f"Training failed: {task.error}")
        raise Exception(task.error)
    elif task.status == TaskStatus.CANCELLED:
        logger.warning(f"Training was cancelled: {task_id}")
        return None
    else:
        logger.warning(f"Training timeout or still running: {task_id}")
        return None


def list_training_tasks():
    """List all training tasks."""
    manager = get_background_manager()
    tasks = manager.list_tasks()
    return tasks
