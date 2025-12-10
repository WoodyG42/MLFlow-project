import mlflow
from sklearn.datasets import load_breast_cancer
import pandas as pd
import os

def ingest(output_path="data/breast_cancer.csv"):
    mlflow.set_experiment("mlflow_lifecycle")
    with mlflow.start_run(run_name="data_ingestion"):
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        os.makedirs("data", exist_ok=True)
        df.to_csv(output_path, index=False)
        mlflow.log_param("rows", len(df))
        mlflow.log_artifact(output_path)
        print("Data saved to:", output_path)
if __name__ == "__main__":
    ingest()