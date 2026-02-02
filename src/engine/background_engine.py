"""
Background engine for async search operations.

Extends XiangqiEngine with background task support.
"""

from typing import List

from ..utils.background import get_background_manager
from ..utils.logger import get_logger
from .xiangqi_engine import XiangqiEngine

logger = get_logger("background_engine")


class BackgroundEngine(XiangqiEngine):
    """Engine with background search capabilities."""

    def __init__(self, *args, **kwargs):
        """Initialize background engine."""
        super().__init__(*args, **kwargs)
        self.manager = get_background_manager()
        logger.info("BackgroundEngine initialized")

    def search_background(
        self,
        depth: int,
        task_id: str
    ) -> str:
        """
        Search in background.

        Args:
            depth: Search depth
            task_id: Task identifier

        Returns:
            Task ID
        """
        def search_func():
            logger.info(f"Background search task {task_id}: depth={depth}")
            score, move = self.search(depth=depth)
            return {'score': score, 'move': move}

        self.manager.submit_task(
            task_id=task_id,
            func=search_func
        )

        logger.info(f"Background search started: {task_id}")
        return task_id

    def analyze_batch_background(
        self,
        positions: List[str],
        depth: int,
        task_id: str
    ) -> str:
        """
        Analyze multiple positions in background.

        Args:
            positions: List of board strings
            depth: Search depth
            task_id: Task identifier

        Returns:
            Task ID
        """
        def analyze_batch():
            logger.info(f"Batch analysis {task_id}: {len(positions)} positions")
            results = []

            for i, pos_str in enumerate(positions):
                try:
                    self.set_position(pos_str)
                    score, move = self.search(depth=depth)
                    results.append({
                        'position': pos_str,
                        'score': score,
                        'move': move,
                        'success': True
                    })
                except Exception as e:
                    logger.error(f"Position {i} failed: {e}")
                    results.append({
                        'position': pos_str,
                        'error': str(e),
                        'success': False
                    })

                # Progress update
                progress = (i + 1) / len(positions)
                if (i + 1) % 10 == 0 or progress == 1.0:
                    logger.info(f"Batch analysis: {progress*100:.1f}% complete")

            logger.info(f"Batch analysis {task_id} completed: {len(results)} positions")
            return results

        self.manager.submit_task(
            task_id=task_id,
            func=analyze_batch
        )

        logger.info(f"Background batch analysis started: {task_id}")
        return task_id

    def get_task_status(self, task_id: str):
        """Get status of a background task."""
        return self.manager.get_task_status(task_id)
