"""
Experiment tracker for monitoring and logging machine learning experiments.
"""
from typing import Any, Dict, List, Optional, Union
import logging
import os
import json
import mlflow
import pandas as pd
from datetime import datetime
import uuid


logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Experiment tracker for monitoring and logging machine learning experiments."""
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_dir: str = "experiments",
        backend: str = "mlflow"
    ):
        """
        Initialize experiment tracker.
        
        Args:
            tracking_uri: URI for MLflow tracking server (if using MLflow)
            experiment_dir: Directory for storing experiment logs
            backend: Backend to use for tracking ('mlflow' or 'local')
        """
        self.experiment_dir = experiment_dir
        self.backend = backend.lower()
        self.current_run_id = None
        self.current_experiment_id = None
        
        # Create experiment directory if it doesn't exist
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Configure backend
        if self.backend == "mlflow":
            # Set MLflow tracking URI
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                # Use local tracking URI
                local_tracking_uri = f"file:{os.path.abspath(os.path.join(experiment_dir, 'mlflow'))}"
                mlflow.set_tracking_uri(local_tracking_uri)
                
            logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        elif self.backend == "local":
            # Initialize local tracking
            self.experiments = {}
            self.runs = {}
            
            # Load existing experiments if any
            self._load_local_experiments()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _load_local_experiments(self) -> None:
        """Load existing local experiments."""
        experiments_file = os.path.join(self.experiment_dir, "experiments.json")
        runs_file = os.path.join(self.experiment_dir, "runs.json")
        
        if os.path.exists(experiments_file):
            with open(experiments_file, 'r') as f:
                self.experiments = json.load(f)
        
        if os.path.exists(runs_file):
            with open(runs_file, 'r') as f:
                self.runs = json.load(f)
    
    def _save_local_experiments(self) -> None:
        """Save local experiments to disk."""
        experiments_file = os.path.join(self.experiment_dir, "experiments.json")
        runs_file = os.path.join(self.experiment_dir, "runs.json")
        
        with open(experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        
        with open(runs_file, 'w') as f:
            json.dump(self.runs, f, indent=2)
    
    def create_experiment(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Create a new experiment.
        
        Args:
            name: Name of the experiment
            tags: Optional tags for the experiment
            
        Returns:
            Experiment ID
        """
        if self.backend == "mlflow":
            # Check if experiment already exists
            experiment = mlflow.get_experiment_by_name(name)
            if experiment:
                experiment_id = experiment.experiment_id
            else:
                # Create new experiment
                experiment_id = mlflow.create_experiment(name=name, tags=tags)
            
            return experiment_id
        elif self.backend == "local":
            # Generate experiment ID
            experiment_id = str(uuid.uuid4())
            
            # Create experiment
            self.experiments[experiment_id] = {
                "name": name,
                "tags": tags or {},
                "creation_time": datetime.utcnow().isoformat(),
                "runs": []
            }
            
            # Save experiments
            self._save_local_experiments()
            
            return experiment_id
    
    def start_experiment(
        self,
        name: str,
        tags: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new experiment run.
        
        Args:
            name: Name of the experiment
            tags: Optional tags for the experiment
            params: Optional parameters for the run
            
        Returns:
            Run ID
        """
        # Format tags
        experiment_tags = {}
        if tags:
            experiment_tags = {"tags": ",".join(tags)}
        
        if self.backend == "mlflow":
            # Get or create experiment
            experiment_id = self.create_experiment(name, experiment_tags)
            
            # Set active experiment
            mlflow.set_experiment(experiment_id=experiment_id)
            
            # Start run
            run = mlflow.start_run()
            self.current_run_id = run.info.run_id
            self.current_experiment_id = experiment_id
            
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            return self.current_run_id
        elif self.backend == "local":
            # Get or create experiment
            experiment_id = None
            for exp_id, exp in self.experiments.items():
                if exp["name"] == name:
                    experiment_id = exp_id
                    break
            
            if experiment_id is None:
                experiment_id = self.create_experiment(name, experiment_tags)
            
            # Generate run ID
            run_id = str(uuid.uuid4())
            
            # Create run
            self.runs[run_id] = {
                "experiment_id": experiment_id,
                "status": "RUNNING",
                "start_time": datetime.utcnow().isoformat(),
                "end_time": None,
                "params": params or {},
                "metrics": {},
                "tags": tags or [],
                "artifacts": {}
            }
            
            # Add run to experiment
            self.experiments[experiment_id]["runs"].append(run_id)
            
            # Save
            self._save_local_experiments()
            
            # Set current run
            self.current_run_id = run_id
            self.current_experiment_id = experiment_id
            
            return run_id
    
    def end_experiment(self, status: str = "success") -> None:
        """
        End the current experiment run.
        
        Args:
            status: Status of the run ('success' or 'failed')
        """
        if self.current_run_id is None:
            logger.warning("No active run to end")
            return
        
        if self.backend == "mlflow":
            # End run
            mlflow.end_run(status=status)
        elif self.backend == "local":
            # Update run status
            self.runs[self.current_run_id]["status"] = status.upper()
            self.runs[self.current_run_id]["end_time"] = datetime.utcnow().isoformat()
            
            # Save
            self._save_local_experiments()
        
        # Clear current run
        self.current_run_id = None
    
    def log_param(self, key: str, value: Any) -> None:
        """
        Log a parameter value.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        if self.current_run_id is None:
            logger.warning("No active run to log parameter")
            return
        
        if self.backend == "mlflow":
            mlflow.log_param(key, value)
        elif self.backend == "local":
            self.runs[self.current_run_id]["params"][key] = value
            self._save_local_experiments()
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log multiple parameter values.
        
        Args:
            params: Dictionary of parameter names and values
        """
        if self.current_run_id is None:
            logger.warning("No active run to log parameters")
            return
        
        if self.backend == "mlflow":
            mlflow.log_params(params)
        elif self.backend == "local":
            self.runs[self.current_run_id]["params"].update(params)
            self._save_local_experiments()
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if self.current_run_id is None:
            logger.warning("No active run to log metric")
            return
        
        if self.backend == "mlflow":
            mlflow.log_metric(key, value, step=step)
        elif self.backend == "local":
            if key not in self.runs[self.current_run_id]["metrics"]:
                self.runs[self.current_run_id]["metrics"][key] = []
            
            metric_entry = {"value": value, "timestamp": datetime.utcnow().isoformat()}
            if step is not None:
                metric_entry["step"] = step
            
            self.runs[self.current_run_id]["metrics"][key].append(metric_entry)
            self._save_local_experiments()
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metric values.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        if self.current_run_id is None:
            logger.warning("No active run to log metrics")
            return
        
        if self.backend == "mlflow":
            mlflow.log_metrics(metrics, step=step)
        elif self.backend == "local":
            for key, value in metrics.items():
                self.log_metric(key, value, step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact file.
        
        Args:
            local_path: Path to the local file
            artifact_path: Optional path within artifact directory
        """
        if self.current_run_id is None:
            logger.warning("No active run to log artifact")
            return
        
        if self.backend == "mlflow":
            mlflow.log_artifact(local_path, artifact_path)
        elif self.backend == "local":
            # Create artifact directory if it doesn't exist
            run_artifacts_dir = os.path.join(self.experiment_dir, "artifacts", self.current_run_id)
            if artifact_path:
                run_artifacts_dir = os.path.join(run_artifacts_dir, artifact_path)
            
            os.makedirs(run_artifacts_dir, exist_ok=True)
            
            # Copy artifact file
            artifact_filename = os.path.basename(local_path)
            artifact_dest = os.path.join(run_artifacts_dir, artifact_filename)
            
            import shutil
            shutil.copy2(local_path, artifact_dest)
            
            # Record artifact
            if artifact_path:
                artifact_key = os.path.join(artifact_path, artifact_filename)
            else:
                artifact_key = artifact_filename
            
            self.runs[self.current_run_id]["artifacts"][artifact_key] = artifact_dest
            self._save_local_experiments()
    
    def log_figure(self, figure, artifact_path: Optional[str] = None) -> None:
        """
        Log a matplotlib or plotly figure.
        
        Args:
            figure: Matplotlib or plotly figure
            artifact_path: Optional path within artifact directory
        """
        if self.current_run_id is None:
            logger.warning("No active run to log figure")
            return
        
        # Create a temporary file for the figure
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        
        # Save figure to temporary file
        try:
            # Check if it's a matplotlib figure
            import matplotlib.pyplot as plt
            if isinstance(figure, plt.Figure):
                figure.savefig(tmp_path, bbox_inches="tight")
            else:
                # Assume it's a plotly figure
                try:
                    import plotly
                    if isinstance(figure, plotly.graph_objs.Figure):
                        figure.write_image(tmp_path)
                    else:
                        logger.warning("Unsupported figure type")
                        return
                except ImportError:
                    logger.warning("Plotly not installed, cannot save figure")
                    return
            
            # Log artifact
            self.log_artifact(tmp_path, artifact_path)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def log_table(self, data: pd.DataFrame, name: str) -> None:
        """
        Log a pandas DataFrame as a table.
        
        Args:
            data: Pandas DataFrame
            name: Name of the table
        """
        if self.current_run_id is None:
            logger.warning("No active run to log table")
            return
        
        # Create a temporary file for the table
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        
        # Save table to temporary file
        try:
            data.to_csv(tmp_path, index=False)
            
            # Log artifact
            self.log_artifact(tmp_path, name)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment details.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Experiment details if found, None otherwise
        """
        if self.backend == "mlflow":
            experiment = mlflow.get_experiment(experiment_id)
            if experiment:
                return {
                    "id": experiment.experiment_id,
                    "name": experiment.name,
                    "tags": experiment.tags or {},
                    "creation_time": datetime.fromtimestamp(experiment.creation_time / 1000).isoformat(),
                    "last_update_time": datetime.fromtimestamp(experiment.last_update_time / 1000).isoformat() if experiment.last_update_time else None
                }
            return None
        elif self.backend == "local":
            return self.experiments.get(experiment_id)
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get run details.
        
        Args:
            run_id: ID of the run
            
        Returns:
            Run details if found, None otherwise
        """
        if self.backend == "mlflow":
            try:
                run = mlflow.get_run(run_id)
                if run:
                    return {
                        "id": run.info.run_id,
                        "experiment_id": run.info.experiment_id,
                        "status": run.info.status,
                        "start_time": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                        "end_time": datetime.fromtimestamp(run.info.end_time / 1000).isoformat() if run.info.end_time else None,
                        "params": run.data.params,
                        "metrics": run.data.metrics,
                        "tags": run.data.tags,
                        "artifacts": mlflow.list_artifacts(run_id)
                    }
            except Exception as e:
                logger.error(f"Error getting run: {str(e)}")
                return None
        elif self.backend == "local":
            return self.runs.get(run_id)
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments.
        
        Returns:
            List of experiment summaries
        """
        if self.backend == "mlflow":
            experiments = mlflow.search_experiments()
            return [
                {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "tags": exp.tags or {},
                    "creation_time": datetime.fromtimestamp(exp.creation_time / 1000).isoformat()
                }
                for exp in experiments
            ]
        elif self.backend == "local":
            return [
                {
                    "id": exp_id,
                    "name": exp["name"],
                    "tags": exp["tags"],
                    "creation_time": exp["creation_time"]
                }
                for exp_id, exp in self.experiments.items()
            ]
    
    def list_runs(self, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List runs for an experiment.
        
        Args:
            experiment_id: Optional experiment ID (if None, list all runs)
            
        Returns:
            List of run summaries
        """
        if self.backend == "mlflow":
            filters = []
            if experiment_id:
                filters.append(f"experiment_id = '{experiment_id}'")
            
            runs = mlflow.search_runs(experiment_ids=[experiment_id] if experiment_id else None)
            return [
                {
                    "id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                    "end_time": datetime.fromtimestamp(run.info.end_time / 1000).isoformat() if run.info.end_time else None
                }
                for run in runs
            ]
        elif self.backend == "local":
            if experiment_id:
                if experiment_id not in self.experiments:
                    return []
                
                run_ids = self.experiments[experiment_id]["runs"]
                return [
                    {
                        "id": run_id,
                        "experiment_id": experiment_id,
                        "status": self.runs[run_id]["status"],
                        "start_time": self.runs[run_id]["start_time"],
                        "end_time": self.runs[run_id]["end_time"]
                    }
                    for run_id in run_ids if run_id in self.runs
                ]
            else:
                return [
                    {
                        "id": run_id,
                        "experiment_id": run["experiment_id"],
                        "status": run["status"],
                        "start_time": run["start_time"],
                        "end_time": run["end_time"]
                    }
                    for run_id, run in self.runs.items()
                ]
    
    def get_run_metrics(self, run_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get metrics for a run.
        
        Args:
            run_id: ID of the run
            
        Returns:
            Dictionary of metric history
        """
        if self.backend == "mlflow":
            try:
                run = mlflow.get_run(run_id)
                if run:
                    # For MLflow, we need to get metric history for each metric
                    metrics = {}
                    for metric_name in run.data.metrics.keys():
                        metric_history = mlflow.tracking.MlflowClient().get_metric_history(run_id, metric_name)
                        metrics[metric_name] = [
                            {
                                "value": metric.value,
                                "timestamp": datetime.fromtimestamp(metric.timestamp / 1000).isoformat(),
                                "step": metric.step
                            }
                            for metric in metric_history
                        ]
                    return metrics
            except Exception as e:
                logger.error(f"Error getting run metrics: {str(e)}")
                return {}
        elif self.backend == "local":
            run = self.runs.get(run_id)
            if run:
                return run.get("metrics", {})
            return {}
    
    def download_artifacts(self, run_id: str, output_dir: str) -> None:
        """
        Download artifacts for a run.
        
        Args:
            run_id: ID of the run
            output_dir: Directory to download artifacts to
        """
        if self.backend == "mlflow":
            try:
                mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=output_dir)
            except Exception as e:
                logger.error(f"Error downloading artifacts: {str(e)}")
        elif self.backend == "local":
            run = self.runs.get(run_id)
            if run:
                os.makedirs(output_dir, exist_ok=True)
                
                import shutil
                for artifact_key, artifact_path in run.get("artifacts", {}).items():
                    if os.path.exists(artifact_path):
                        artifact_dest = os.path.join(output_dir, artifact_key)
                        os.makedirs(os.path.dirname(artifact_dest), exist_ok=True)
                        shutil.copy2(artifact_path, artifact_dest)