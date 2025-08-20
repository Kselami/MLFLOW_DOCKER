import argparse
import os
import tempfile

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split


def train_and_log(
    experiment_name: str,
    registered_model_name: str,
    n_estimators: int,
    max_depth: int | None,
    seed: int,
    min_accuracy: float,
) -> float:
    # Pointeur MLflow + expérience
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(experiment_name)

    # Données (30 features) + noms de colonnes pour un schéma propre
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=list(data.feature_names))
    y = pd.Series(data.target, name="target")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    with mlflow.start_run() as run:
        # Params
        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth, "seed": seed})

        # Modèle
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1
        )
        clf.fit(X_tr, y_tr)

        # Prédictions
        y_pred = clf.predict(X_te)
        y_proba = None
        if hasattr(clf, "predict_proba"):
            try:
                y_proba = clf.predict_proba(X_te)[:, 1]
            except Exception:
                y_proba = None

        # Métriques
        metrics = {
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "precision": float(precision_score(y_te, y_pred, zero_division=0)),
            "recall": float(recall_score(y_te, y_pred, zero_division=0)),
            "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        }
        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_te, y_proba))
            except Exception:
                pass
            try:
                metrics["log_loss"] = float(log_loss(y_te, y_proba, labels=[0, 1]))
            except Exception:
                pass
            try:
                metrics["avg_precision"] = float(average_precision_score(y_te, y_proba))
            except Exception:
                pass

        mlflow.log_metrics(metrics)

        # Artefacts texte (pas besoin de libs en plus)
        try:
            cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["true_malignant(0)", "true_benign(1)"],
                                 columns=["pred_malignant(0)", "pred_benign(1)"])
            with tempfile.TemporaryDirectory() as td:
                # Classification report
                rep_path = os.path.join(td, "classification_report.txt")
                with open(rep_path, "w", encoding="utf-8") as f:
                    f.write(classification_report(y_te, y_pred, digits=4))
                mlflow.log_artifact(rep_path, artifact_path="reports")

                # Confusion matrix CSV
                cm_path = os.path.join(td, "confusion_matrix.csv")
                cm_df.to_csv(cm_path, index=True)
                mlflow.log_artifact(cm_path, artifact_path="reports")

                # Feature importances
                if hasattr(clf, "feature_importances_"):
                    fi = (
                        pd.Series(clf.feature_importances_, index=X.columns)
                        .sort_values(ascending=False)
                        .to_frame("importance")
                    )
                    fi_path = os.path.join(td, "feature_importances.csv")
                    fi.to_csv(fi_path, index=True)
                    mlflow.log_artifact(fi_path, artifact_path="reports")
        except Exception:
            pass

        # Signature + input_example => MLflow montre Schema / Input / Output
        signature = infer_signature(X_te, y_pred)
        input_example = X_te.head(2)

        # Enregistrement du modèle (Model Registry)
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
        )

        acc = metrics["accuracy"]
        print(f"Run: {run.info.run_id}  accuracy={acc:.4f}")
        if acc < min_accuracy:
            print(f"[WARN] accuracy {acc:.4f} < min_accuracy {min_accuracy:.4f} (le run reste FINISHED).")
        return acc


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_name", default="ci-experiment")
    ap.add_argument("--registered_model_name", default="CancerClassifier")
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_accuracy", type=float, default=0.0)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    train_and_log(
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.seed,
        min_accuracy=args.min_accuracy,
    )


if __name__ == "__main__":
    main()

