from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json
import joblib
import numpy as np
from pathlib import Path

from app.core.config import get_settings
from app.models.schemas import FeedbackRequest

settings = get_settings()
logger = logging.getLogger(__name__)


class FeedbackLoop:
    """
    Continuous learning system that collects user feedback and uses it to improve models.
    Implements active learning and fine-tuning pipelines.
    """

    def __init__(self):
        self.feedback_history: List[Dict[str, Any]] = []
        self.feedback_by_model: Dict[str, List[Dict]] = defaultdict(list)
        self.training_data: List[Dict[str, Any]] = []
        self.model_versions: Dict[str, int] = defaultdict(int)

    def record_feedback(self, feedback: FeedbackRequest):
        """
        Record user feedback on query results.

        Args:
            feedback: FeedbackRequest with user's rating and corrections
        """
        feedback_entry = {
            "query_id": feedback.query_id,
            "rating": feedback.rating,
            "is_correct": feedback.is_correct,
            "comments": feedback.comments,
            "correct_sql": feedback.correct_sql,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.feedback_history.append(feedback_entry)

        # Extract model from query_id (assuming format: model_timestamp_uuid)
        parts = feedback.query_id.split("_")
        if parts:
            model = parts[0]
            self.feedback_by_model[model].append(feedback_entry)

        logger.info(f"Recorded feedback for query {feedback.query_id}: rating={feedback.rating}")

        # Check if we should trigger retraining
        self._check_retraining_trigger()

    def _check_retraining_trigger(self):
        """Check if we have enough feedback to trigger retraining."""
        total_feedback = len(self.feedback_history)

        # Trigger retraining every 100 feedback entries
        if total_feedback > 0 and total_feedback % 100 == 0:
            logger.info(f"Accumulated {total_feedback} feedback entries, triggering retraining")
            self.prepare_training_data()

    def prepare_training_data(self):
        """
        Prepare training data from feedback for model fine-tuning.
        Focuses on incorrect predictions and low-rated results.
        """
        # Filter for negative feedback (priority for retraining)
        negative_feedback = [
            f for f in self.feedback_history
            if not f["is_correct"] or f["rating"] <= 3
        ]

        # Also include some positive feedback for balance
        positive_feedback = [
            f for f in self.feedback_history
            if f["is_correct"] and f["rating"] >= 4
        ][:len(negative_feedback)]  # Balance with negative

        # Combine and prepare training data
        training_samples = []

        for feedback in negative_feedback + positive_feedback:
            # This would need to be extended to retrieve the original question
            # For now, we'll store the feedback structure
            training_samples.append({
                "query_id": feedback["query_id"],
                "is_correct": feedback["is_correct"],
                "rating": feedback["rating"],
                "correct_sql": feedback.get("correct_sql"),
                "timestamp": feedback["timestamp"]
            })

        self.training_data = training_samples

        logger.info(f"Prepared {len(training_samples)} training samples")

        # Save training data
        self._save_training_data()

    def _save_training_data(self):
        """Save training data to disk."""
        try:
            training_dir = Path("data/training")
            training_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filepath = training_dir / f"training_data_{timestamp}.json"

            with open(filepath, 'w') as f:
                json.dump(self.training_data, f, indent=2)

            logger.info(f"Saved training data to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save training data: {e}", exc_info=True)

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about collected feedback."""
        if not self.feedback_history:
            return {
                "total_feedback": 0,
                "avg_rating": 0.0,
                "accuracy": 0.0,
                "by_model": {}
            }

        total = len(self.feedback_history)
        avg_rating = sum(f["rating"] for f in self.feedback_history) / total
        accuracy = sum(1 for f in self.feedback_history if f["is_correct"]) / total

        # Stats by model
        model_stats = {}
        for model, feedback_list in self.feedback_by_model.items():
            model_total = len(feedback_list)
            model_avg = sum(f["rating"] for f in feedback_list) / model_total
            model_acc = sum(1 for f in feedback_list if f["is_correct"]) / model_total

            model_stats[model] = {
                "total": model_total,
                "avg_rating": model_avg,
                "accuracy": model_acc
            }

        return {
            "total_feedback": total,
            "avg_rating": avg_rating,
            "accuracy": accuracy,
            "by_model": model_stats
        }

    def get_low_performing_queries(
        self,
        threshold: float = 3.0,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get queries with low ratings for analysis.

        Args:
            threshold: Rating threshold
            limit: Maximum number of queries to return

        Returns:
            List of low-performing query feedback entries
        """
        low_performing = [
            f for f in self.feedback_history
            if f["rating"] <= threshold
        ]

        # Sort by rating (lowest first)
        low_performing.sort(key=lambda x: x["rating"])

        return low_performing[:limit]

    def get_improvement_suggestions(self) -> List[str]:
        """
        Analyze feedback and provide improvement suggestions.

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        stats = self.get_feedback_stats()

        # Check overall accuracy
        if stats["accuracy"] < 0.8:
            suggestions.append(
                f"Overall accuracy ({stats['accuracy']:.1%}) is below 80%. "
                "Consider collecting more training data or adjusting model parameters."
            )

        # Check model-specific performance
        for model, model_stats in stats["by_model"].items():
            if model_stats["accuracy"] < 0.75:
                suggestions.append(
                    f"Model '{model}' has low accuracy ({model_stats['accuracy']:.1%}). "
                    "Consider fine-tuning or replacing this model."
                )

        # Check for common error patterns
        if self.training_data:
            error_types = defaultdict(int)
            for sample in self.training_data:
                if not sample["is_correct"]:
                    # Would need more sophisticated error classification
                    error_types["sql_generation"] += 1

            if error_types["sql_generation"] > 10:
                suggestions.append(
                    "High number of SQL generation errors. "
                    "Consider improving schema context or adding more examples."
                )

        if not suggestions:
            suggestions.append("Performance is good. Continue monitoring feedback.")

        return suggestions

    def export_training_data(self, filepath: str):
        """
        Export training data for external fine-tuning.

        Args:
            filepath: Path to save the exported data
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.training_data, f, indent=2)

            logger.info(f"Exported training data to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export training data: {e}", exc_info=True)

    def load_training_data(self, filepath: str):
        """
        Load training data from file.

        Args:
            filepath: Path to load training data from
        """
        try:
            with open(filepath, 'r') as f:
                self.training_data = json.load(f)

            logger.info(f"Loaded {len(self.training_data)} training samples from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load training data: {e}", exc_info=True)


class ActiveLearningSelector:
    """
    Active learning selector to identify most informative samples for labeling.
    """

    def __init__(self):
        self.uncertainty_threshold = 0.5
        self.diversity_threshold = 0.7

    def select_samples(
        self,
        unlabeled_samples: List[Dict[str, Any]],
        predictions: List[float],
        n_samples: int = 10
    ) -> List[int]:
        """
        Select most informative samples for human labeling.

        Args:
            unlabeled_samples: List of unlabeled samples
            predictions: Prediction confidence scores
            n_samples: Number of samples to select

        Returns:
            Indices of selected samples
        """
        # Strategy 1: Uncertainty sampling (select samples with low confidence)
        uncertainty_indices = [
            i for i, conf in enumerate(predictions)
            if conf < self.uncertainty_threshold
        ]

        # If we have enough uncertain samples, use them
        if len(uncertainty_indices) >= n_samples:
            return uncertainty_indices[:n_samples]

        # Strategy 2: Diversity sampling (select diverse samples)
        # This would require embeddings, simplified for now
        selected = list(range(min(n_samples, len(unlabeled_samples))))

        return selected

    def calculate_uncertainty(self, predictions: List[float]) -> List[float]:
        """
        Calculate uncertainty from prediction confidence.

        Args:
            predictions: List of confidence scores

        Returns:
            List of uncertainty scores
        """
        return [1.0 - conf for conf in predictions]
