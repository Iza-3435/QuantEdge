"""MLflow integration for experiment tracking and model registry."""
from typing import Dict, Any, Optional
from pathlib import Path
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.core.config import settings
from src.core.logging import app_logger


class MLflowTracker:
    """
    MLflow experiment tracking and model registry.

    Tracks:
    - Training metrics
    - Model artifacts
    - Hyperparameters
    - Model versions
    """

    def __init__(self):
        # Set tracking URI
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

        # Set experiment
        experiment = mlflow.set_experiment(settings.mlflow.experiment_name)
        self.experiment_id = experiment.experiment_id

        self.client = MlflowClient()
        self.active_run = None

        app_logger.info(
            f"MLflow tracking initialized: {settings.mlflow.tracking_uri}, "
            f"experiment: {settings.mlflow.experiment_name}"
        )

    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start a new MLflow run."""
        if self.active_run:
            app_logger.warning("Run already active, ending previous run")
            self.end_run()

        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )

        run_id = self.active_run.info.run_id
        app_logger.info(f"Started MLflow run: {run_id}")
        return run_id

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if not self.active_run:
            app_logger.warning("No active run, starting new run")
            self.start_run()

        mlflow.log_params(params)
        app_logger.debug(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if not self.active_run:
            app_logger.warning("No active run, starting new run")
            self.start_run()

        mlflow.log_metrics(metrics, step=step)
        app_logger.debug(f"Logged {len(metrics)} metrics")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log single metric."""
        if not self.active_run:
            app_logger.warning("No active run, starting new run")
            self.start_run()

        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact (file or directory)."""
        if not self.active_run:
            app_logger.warning("No active run, starting new run")
            self.start_run()

        mlflow.log_artifact(local_path, artifact_path)
        app_logger.debug(f"Logged artifact: {local_path}")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Log model to MLflow.

        Automatically detects model type (PyTorch, sklearn, etc.)
        """
        if not self.active_run:
            app_logger.warning("No active run, starting new run")
            self.start_run()

        # Detect model type
        model_type = type(model).__module__.split('.')[0]

        if model_type == 'torch':
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs
            )
        elif model_type == 'sklearn':
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs
            )
        else:
            # Generic pickling
            mlflow.pyfunc.log_model(
                artifact_path,
                python_model=model,
                registered_model_name=registered_model_name,
                **kwargs
            )

        app_logger.info(f"Logged model: {artifact_path}")

        if registered_model_name:
            app_logger.info(f"Registered model: {registered_model_name}")

    def log_tags(self, tags: Dict[str, str]):
        """Log tags."""
        if not self.active_run:
            app_logger.warning("No active run, starting new run")
            self.start_run()

        mlflow.set_tags(tags)

    def end_run(self):
        """End active run."""
        if self.active_run:
            mlflow.end_run()
            app_logger.info(f"Ended MLflow run: {self.active_run.info.run_id}")
            self.active_run = None

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register model in MLflow Model Registry.

        Returns version number.
        """
        if not settings.mlflow.model_registry:
            app_logger.warning("Model registry disabled in config")
            return ""

        result = mlflow.register_model(model_uri, name)

        if tags:
            self.client.set_registered_model_tag(name, tags)

        version = result.version
        app_logger.info(f"Registered model {name} version {version}")
        return version

    def transition_model_stage(
        self,
        name: str,
        version: str,
        stage: str
    ):
        """
        Transition model to stage (Staging, Production, Archived).
        """
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage
        )
        app_logger.info(f"Transitioned {name} v{version} to {stage}")

    def load_model(self, model_uri: str) -> Any:
        """Load model from MLflow."""
        model = mlflow.pyfunc.load_model(model_uri)
        app_logger.info(f"Loaded model from {model_uri}")
        return model

    def get_best_run(self, metric: str, ascending: bool = False) -> Optional[Dict]:
        """Get best run based on metric."""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )

        if runs:
            return runs[0].to_dictionary()
        return None

    def compare_runs(self, run_ids: list, metrics: list) -> Dict:
        """Compare multiple runs."""
        comparison = {}

        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                'params': run.data.params,
                'metrics': {m: run.data.metrics.get(m) for m in metrics}
            }

        return comparison


# Context manager for automatic run management
class mlflow_run:
    """Context manager for MLflow runs."""

    def __init__(self, tracker: MLflowTracker, run_name: Optional[str] = None):
        self.tracker = tracker
        self.run_name = run_name

    def __enter__(self):
        self.tracker.start_run(self.run_name)
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.end_run()


# Global tracker instance
mlflow_tracker = MLflowTracker()
