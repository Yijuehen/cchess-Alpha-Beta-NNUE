"""
Tests for background task manager.
"""

import pytest
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.background import (
    BackgroundTaskManager,
    TaskStatus,
    get_background_manager
)


def test_manager_creation():
    """Test manager can be created."""
    manager = BackgroundTaskManager()
    assert manager is not None
    assert manager.tasks == {}


def test_submit_task():
    """Test task submission."""
    manager = BackgroundTaskManager()

    def simple_task():
        return 42

    task_id = manager.submit_task("test1", simple_task)

    assert task_id == "test1"
    assert "test1" in manager.tasks


def test_task_completion():
    """Test task completes successfully."""
    manager = BackgroundTaskManager()

    def simple_task():
        time.sleep(0.1)
        return 42

    task_id = manager.submit_task("test2", simple_task)

    # Wait a bit for task to complete
    time.sleep(0.2)

    status = manager.get_task_status("test2")
    assert status is not None
    assert status.status == TaskStatus.COMPLETED
    assert status.result == 42
    assert status.progress == 1.0


def test_task_failure():
    """Test task failure handling."""
    manager = BackgroundTaskManager()

    def failing_task():
        raise ValueError("Test error")

    task_id = manager.submit_task("test3", failing_task)

    time.sleep(0.2)

    status = manager.get_task_status("test3")
    assert status.status == TaskStatus.FAILED
    assert "Test error" in status.error


def test_wait_for_task():
    """Test waiting for task."""
    manager = BackgroundTaskManager()

    def quick_task():
        return "done"

    task_id = manager.submit_task("test4", quick_task)

    result = manager.wait_for_task("test4", timeout=1.0)

    assert result is not None
    assert result.status == TaskStatus.COMPLETED
    assert result.result == "done"


def test_cancel_task():
    """Test task cancellation."""
    manager = BackgroundTaskManager()

    def long_task():
        time.sleep(5)
        return "done"

    task_id = manager.submit_task("test5", long_task)

    # Cancel immediately
    success = manager.cancel_task(task_id)

    # Note: This sets the status to cancelled but doesn't actually stop the thread
    assert success is True


def test_global_manager():
    """Test global manager instance."""
    manager = get_background_manager()
    assert isinstance(manager, BackgroundTaskManager)

    # Should return same instance
    manager2 = get_background_manager()
    assert manager is manager2


def test_elapsed_time():
    """Test elapsed time calculation."""
    manager = BackgroundTaskManager()

    def quick_task():
        time.sleep(0.1)
        return "done"

    task_id = manager.submit_task("test6", quick_task)

    time.sleep(0.2)

    task = manager.get_task_status("test6")
    assert task.elapsed_time() >= 0.1


def test_list_tasks():
    """Test listing all tasks."""
    manager = BackgroundTaskManager()

    def task1():
        return 1

    def task2():
        return 2

    manager.submit_task("list_test1", task1)
    manager.submit_task("list_test2", task2)

    time.sleep(0.2)

    tasks = manager.list_tasks()
    assert "list_test1" in tasks
    assert "list_test2" in tasks
