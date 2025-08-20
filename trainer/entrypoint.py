import os
import sys
import time
import urllib.request
import urllib.error
import pytest
from src.train import main as train_main


def wait_for_mlflow_api(uri: str, seconds: int = 180) -> None:
    """Attend que l'API MLflow réponde 200 sur /api/2.0/..."""
    url = uri.rstrip("/") + "/api/2.0/mlflow/experiments/list"
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"MLflow API ready at {uri}")
                    return
                else:
                    print(f"Waiting MLflow API (status={resp.status})...")
        except Exception as e:
            print(f"Waiting MLflow API ({e.__class__.__name__})...")
        time.sleep(2)
    raise SystemExit(f"MLflow API not ready after {seconds}s at {uri}")


if __name__ == "__main__":
    # 1) Attendre l'API MLflow
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    wait_for_mlflow_api(uri, 180)

    # 2) Tests
    code = pytest.main(["-q"])
    if code != 0:
        sys.exit(code)

    # 3) Entraînement
    train_main([])
