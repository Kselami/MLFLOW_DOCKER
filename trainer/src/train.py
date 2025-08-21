
import argparse, os
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train_and_log(experiment_name: str,
                  registered_model_name: str,
                  n_estimators: int, max_depth: int | None, seed: int,
                  min_accuracy: float) -> float:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(experiment_name)

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    with mlflow.start_run() as run:
        params = {"n_estimators": n_estimators, "max_depth": max_depth, "seed": seed}
        for k, v in params.items():
            mlflow.log_param(k, v)

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
        clf.fit(X_tr, y_tr)
        y_pr = clf.predict(X_te)
        acc = float(accuracy_score(y_te, y_pr))
        mlflow.log_metric("accuracy", acc)

        # Transition to Production if accuracy is good
        if acc >= min_accuracy:
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                registered_model_name=registered_model_name
            )
            # Transition the new model version to Production
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions(registered_model_name, stages=["None"])
            for version in latest_versions:
                if version.run_id == run.info.run_id:
                    client.transition_model_version_stage(
                        name=registered_model_name,
                        version=version.version,
                        stage="Production",
                        archive_existing_versions=True
                    )
        else:
            print(f"[WARN] accuracy {acc:.4f} < min_accuracy {min_accuracy:.4f}. Model not registered.")

        print(f"Run: {run.info.run_id}  accuracy={acc:.4f}")
        return acc

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_name", default="ci-experiment")
    ap.add_argument("--registered_model_name", default="CancerClassifier")
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_accuracy", type=float, default=0.9)
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



