import sys
import pytest
from src.train import main as train_main

if __name__ == "__main__":
    code = pytest.main(["-q"])
    if code != 0:
        sys.exit(code)
    # Entraînement avec paramètres par défaut (min-accuracy=0.0)
    train_main([])

