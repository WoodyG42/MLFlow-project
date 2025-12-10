import os

import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

# 👇 This should match your registered model name and alias
MODEL_URI = "models:/LR_Cancer_Model@champion"


def explain():
    # Separate experiment just for interpretability runs (optional but nice)
    mlflow.set_experiment("mlflow_interpretability")

    # --- 1. Load the same data used in training ---
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=["target"])
    y = df["target"]

    # Use the SAME split as in train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --- 2. Load the trained model from MLflow ---
    # This returns your GridSearchCV + Pipeline model
    model = mlflow.sklearn.load_model(MODEL_URI)

    # Best pipeline chosen by GridSearchCV
    pipe = model.best_estimator_
    scaler = pipe.named_steps["scaler"]
    lr = pipe.named_steps["LR"]

    # --- 3. Transform data the same way as during training ---
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 4. Build SHAP explainer for Logistic Regression ---
    # LinearExplainer is appropriate for linear models like LR
    explainer = shap.LinearExplainer(lr, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)  # shape: (n_samples, n_features)

    # --- 5. Make directory for artifacts ---
    os.makedirs("explain/artifacts", exist_ok=True)
    shap_plot_path = "explain/artifacts/shap_summary.png"

    # --- 6. Create SHAP summary plot ---
    plt.figure()
    # Use scaled features but keep original feature names for readability
    shap.summary_plot(
        shap_values,
        X_test_scaled,
        feature_names=X.columns,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(shap_plot_path, bbox_inches="tight")
    plt.close()

    # --- 7. Log the SHAP plot to MLflow ---
    with mlflow.start_run(run_name="shap_explanations"):
        mlflow.log_artifact(shap_plot_path)

    print(f"✅ SHAP summary plot saved and logged: {shap_plot_path}")


if __name__ == "__main__":
    explain()
