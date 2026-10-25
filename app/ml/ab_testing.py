from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from scipy import stats

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class ABTestFramework:
    """
    A/B testing framework for model comparison.
    Implements statistical testing and Bayesian bandit for model selection.
    """

    def __init__(self):
        self.experiments: Dict[str, Dict] = {}
        self.results: Dict[str, List[Dict]] = defaultdict(list)
        self.bandit_state: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"successes": 1.0, "failures": 1.0}  # Beta prior
        )

    def create_experiment(
        self,
        experiment_id: str,
        control_model: str,
        treatment_model: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create a new A/B test experiment.

        Args:
            experiment_id: Unique identifier for the experiment
            control_model: Name of control model
            treatment_model: Name of treatment model
            description: Description of the experiment

        Returns:
            Experiment configuration
        """
        if experiment_id in self.experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")

        experiment = {
            "id": experiment_id,
            "control_model": control_model,
            "treatment_model": treatment_model,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
            "min_samples": settings.ab_test_min_samples,
            "traffic_split": settings.ab_test_ratio
        }

        self.experiments[experiment_id] = experiment
        logger.info(f"Created A/B test experiment: {experiment_id}")

        return experiment

    def assign_variant(
        self,
        experiment_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """
        Assign a user to control or treatment variant.

        Args:
            experiment_id: Experiment identifier
            user_id: User identifier for consistent assignment

        Returns:
            "control" or "treatment"
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        # Use user_id for consistent assignment (if provided)
        if user_id:
            # Hash user_id to get consistent assignment
            hash_val = hash(user_id) % 100
            variant = "treatment" if hash_val < (experiment["traffic_split"] * 100) else "control"
        else:
            # Random assignment
            variant = "treatment" if np.random.random() < experiment["traffic_split"] else "control"

        return variant

    def record_result(
        self,
        experiment_id: str,
        variant: str,
        success: bool,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Record result for a variant.

        Args:
            experiment_id: Experiment identifier
            variant: "control" or "treatment"
            success: Whether the result was successful
            metrics: Additional metrics to record
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        result = {
            "variant": variant,
            "success": success,
            "metrics": metrics or {},
            "timestamp": datetime.utcnow().isoformat()
        }

        self.results[experiment_id].append(result)

        # Update bandit state
        model = self.experiments[experiment_id][f"{variant}_model"]
        if success:
            self.bandit_state[model]["successes"] += 1
        else:
            self.bandit_state[model]["failures"] += 1

        logger.info(f"Recorded result for {experiment_id}: {variant}, success={success}")

    def get_experiment_stats(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Get statistics for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment statistics
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        results = self.results[experiment_id]
        experiment = self.experiments[experiment_id]

        if not results:
            return {
                "experiment_id": experiment_id,
                "status": experiment["status"],
                "control_samples": 0,
                "treatment_samples": 0,
                "control_success_rate": 0.0,
                "treatment_success_rate": 0.0,
                "is_significant": False,
                "p_value": None,
                "recommendation": "insufficient_data"
            }

        # Separate results by variant
        control_results = [r for r in results if r["variant"] == "control"]
        treatment_results = [r for r in results if r["variant"] == "treatment"]

        control_successes = sum(1 for r in control_results if r["success"])
        treatment_successes = sum(1 for r in treatment_results if r["success"])

        control_rate = control_successes / len(control_results) if control_results else 0
        treatment_rate = treatment_successes / len(treatment_results) if treatment_results else 0

        # Statistical significance test (Fisher's exact test for binary outcomes)
        is_significant = False
        p_value = None

        if len(control_results) >= experiment["min_samples"] and len(treatment_results) >= experiment["min_samples"]:
            try:
                # Create contingency table
                table = [
                    [control_successes, len(control_results) - control_successes],
                    [treatment_successes, len(treatment_results) - treatment_successes]
                ]
                _, p_value = stats.fisher_exact(table, alternative='two-sided')
                is_significant = p_value < 0.05
            except Exception as e:
                logger.warning(f"Statistical test failed: {e}")

        # Determine recommendation
        if is_significant:
            if treatment_rate > control_rate:
                recommendation = "treatment"
            else:
                recommendation = "control"
        else:
            recommendation = "continue_testing"

        return {
            "experiment_id": experiment_id,
            "status": experiment["status"],
            "control_samples": len(control_results),
            "treatment_samples": len(treatment_results),
            "control_success_rate": control_rate,
            "treatment_success_rate": treatment_rate,
            "lift": (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0,
            "is_significant": is_significant,
            "p_value": p_value,
            "recommendation": recommendation
        }

    def get_bandit_recommendation(self, models: List[str]) -> str:
        """
        Get model recommendation using Thompson sampling (Bayesian bandit).

        Args:
            models: List of model names to consider

        Returns:
            Recommended model name
        """
        samples = {}

        for model in models:
            state = self.bandit_state[model]
            # Sample from Beta distribution
            samples[model] = np.random.beta(
                state["successes"],
                state["failures"]
            )

        # Return model with highest sample
        return max(samples.items(), key=lambda x: x[1])[0]

    def conclude_experiment(
        self,
        experiment_id: str,
        winner: Optional[str] = None
    ):
        """
        Conclude an experiment and optionally declare a winner.

        Args:
            experiment_id: Experiment identifier
            winner: "control", "treatment", or None for auto-detection
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        if winner is None:
            stats = self.get_experiment_stats(experiment_id)
            winner = stats["recommendation"]

        self.experiments[experiment_id]["status"] = "concluded"
        self.experiments[experiment_id]["winner"] = winner
        self.experiments[experiment_id]["concluded_at"] = datetime.utcnow().isoformat()

        logger.info(f"Experiment {experiment_id} concluded with winner: {winner}")

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments with their statistics."""
        experiments_with_stats = []

        for exp_id, experiment in self.experiments.items():
            stats = self.get_experiment_stats(exp_id)
            experiments_with_stats.append({
                **experiment,
                "stats": stats
            })

        return experiments_with_stats

    def reset_experiment(self, experiment_id: str):
        """Reset an experiment (clear results)."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        self.results[experiment_id] = []
        self.experiments[experiment_id]["status"] = "active"
        self.experiments[experiment_id]["created_at"] = datetime.utcnow().isoformat()

        if "winner" in self.experiments[experiment_id]:
            del self.experiments[experiment_id]["winner"]
        if "concluded_at" in self.experiments[experiment_id]:
            del self.experiments[experiment_id]["concluded_at"]

        logger.info(f"Reset experiment: {experiment_id}")
