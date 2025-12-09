"""
MLflow Configuration for Real Estate Estimator

This module centralizes all MLflow configuration settings.
Think of this as the "settings file" for experiment tracking.

MLflow Concepts Explained:
--------------------------
1. TRACKING URI: Where MLflow stores experiment data
   - "sqlite:///mlflow.db" = Local SQLite database
   - "http://server:5000" = Remote MLflow server
   - "./mlruns" = File-based storage (simplest)

2. EXPERIMENT: A logical grouping of runs
   - Like a "project" or "initiative"
   - Example: "real-estate-model-development"

3. RUN: A single training execution
   - Logs: parameters, metrics, artifacts, tags
   - Has unique run_id for tracking

4. ARTIFACT: Files produced by a run
   - Models, plots, data samples, configs

5. REGISTRY: Production model versioning
   - Stages: None -> Staging -> Production -> Archived
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Project root (where this file lives)
PROJECT_ROOT = Path(__file__).parent

# MLflow tracking directory
MLFLOW_DIR = PROJECT_ROOT / "mlflow"
MLFLOW_DIR.mkdir(exist_ok=True)

# =============================================================================
# MLFLOW SETTINGS
# =============================================================================

# Tracking URI: Where experiment data is stored
# Using SQLite for simplicity - works locally, no server needed
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"

# Artifact location: Where model files are stored
# Use file:// URI format for cross-platform compatibility
MLFLOW_ARTIFACT_LOCATION = (MLFLOW_DIR / "artifacts").as_uri()

# Default experiment name
MLFLOW_EXPERIMENT_NAME = "real-estate-predictor"

# Model registry name (for production deployments)
MLFLOW_MODEL_NAME = "real-estate-xgboost"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================


def setup_mlflow():
    """
    Configure MLflow for this project.

    Call this at the start of any script that uses MLflow.

    Returns:
        experiment_id: The ID of the configured experiment
    """
    import mlflow

    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create/get experiment
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        # Create new experiment
        experiment_id = mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_LOCATION
        )
        print(f"Created new experiment: {MLFLOW_EXPERIMENT_NAME} (id: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {MLFLOW_EXPERIMENT_NAME} (id: {experiment_id})")

    # Set as active experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    return experiment_id


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_latest_production_model():
    """
    Get the latest Production model from the registry.

    Returns:
        model: The loaded model, or None if no production model exists
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}/Production")
        return model
    except Exception as e:
        print(f"No production model found: {e}")
        return None


def list_experiments():
    """List all MLflow experiments."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiments = mlflow.search_experiments()

    print("\nMLflow Experiments:")
    print("-" * 60)
    for exp in experiments:
        print(f"  {exp.experiment_id}: {exp.name}")
        print(f"     Artifact Location: {exp.artifact_location}")
        print(f"     Lifecycle Stage: {exp.lifecycle_stage}")

    return experiments


def list_runs(experiment_name: str = MLFLOW_EXPERIMENT_NAME, max_results: int = 10):
    """List recent runs for an experiment."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    runs = mlflow.search_runs(
        experiment_names=[experiment_name], max_results=max_results, order_by=["start_time DESC"]
    )

    if runs.empty:
        print(f"No runs found for experiment: {experiment_name}")
        return runs

    print(f"\nRecent runs for '{experiment_name}':")
    print("-" * 80)

    # Display key columns
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]

    for _, run in runs.iterrows():
        print(f"\nRun: {run['run_id'][:8]}...")
        print(f"  Status: {run['status']}")
        print(f"  Started: {run['start_time']}")

        # Print metrics
        if metric_cols:
            print("  Metrics:")
            for col in metric_cols[:5]:  # First 5 metrics
                metric_name = col.replace("metrics.", "")
                if pd.notna(run[col]):
                    print(f"    {metric_name}: {run[col]:.4f}")

    return runs


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print(" MLflow Configuration for Real Estate Estimator")
    print("=" * 60)
    print(f"\nTracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Artifact Location: {MLFLOW_ARTIFACT_LOCATION}")
    print(f"Experiment Name: {MLFLOW_EXPERIMENT_NAME}")
    print(f"Model Name: {MLFLOW_MODEL_NAME}")

    # Setup and list experiments
    setup_mlflow()

    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_experiments()
        elif sys.argv[1] == "runs":
            import pandas as pd  # Import here to avoid dependency in config

            list_runs()
    else:
        print("\nUsage:")
        print("  python mlflow_config.py         # Show config")
        print("  python mlflow_config.py list    # List experiments")
        print("  python mlflow_config.py runs    # List recent runs")
