import os
import shutil

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json

# global CV object (you can also create it inside train())
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def train(data_path="data/breast_cancer.csv", n_estimators=100, max_depth=5):
    mlflow.set_experiment("mlflow_lifecycle")

    # NOTE: capture the run object so we can use run.info.run_id
    with mlflow.start_run(run_name="model_training") as run:
        # --- Data ---
        df = pd.read_csv(data_path)
        X = df.drop(columns=["target"])
        y = df["target"]

        # Compute historical stats for each feature (based on training data)
        feature_stats = {}
        for col in X.columns:
            series = X[col]
            feature_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }

        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        stats_path = os.path.join("data", "feature_stats.json")

        with open(stats_path, "w") as f:
            json.dump(feature_stats, f, indent=2)

        # Log as artifact for provenance
        mlflow.log_artifact(stats_path)

        print(f"✅ Feature stats saved to: {stats_path}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- Pipeline + grid ---
        LR_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("LR", LogisticRegression(max_iter=500, n_jobs=None))
        ])

        LR_param_grid = [
            {
                "LR": [LogisticRegression(max_iter=2000, solver="lbfgs")],
                "LR__C": np.logspace(-3, 3, 7),
                "LR__penalty": ["l2"],
            }
        ]

        model = GridSearchCV(
            estimator=LR_pipe,
            param_grid=LR_param_grid,
            cv=skf,
            scoring="roc_auc",   # more appropriate for probabilities
            n_jobs=-1,
            verbose=1
        )

        # --- Fit ---
        model.fit(X_train, y_train)

        # --- Predictions + metrics ---
        y_pred = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds_proba)

        # --- Log params/metrics ---
        # log all best params
        mlflow.log_params(model.best_params_)
        # if you also want a specific one:
        mlflow.log_param("LR_C", model.best_params_["LR__C"])
        mlflow.log_metric("auc", auc)

        # --- Log model (to artifacts) ---
        mlflow.sklearn.log_model(model, "model")

        # --- Confusion matrix artifact ---
        plt.figure(figsize=(10, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.tight_layout()
        plt.savefig("ConfusionMatrix.png")
        plt.close()
        mlflow.log_artifact("ConfusionMatrix.png")

        print(f"Model trained, AUC={auc:.4f}")

        # --- Save chosen model locally ---
        target_path = os.path.join(os.getcwd(), "chosen_model")
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        mlflow.sklearn.save_model(model, target_path)

        print(f"✅ Model saved to: {target_path}")

            # log the model to artifacts
        mlflow.sklearn.log_model(model, "model")

        # Register it
        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name="LR_Cancer_Model",
        )

        # Set alias "champion" to THIS version
        client = MlflowClient()
        client.set_registered_model_alias(
            name="LR_Cancer_Model",
            alias="champion",
            version=result.version,
        )

        print(f"📦 Registered version: {result.version} (alias: champion)")

if __name__ == "__main__":
    train()
