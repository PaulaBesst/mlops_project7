import mlflow
import pandas as pd
import numpy as np
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import os

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def load_model(run_id):
    logged_model = f'runs:/{run_id}/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

@task
def load_data():
    # In a real scenario, you would load new data from a database or file storage
    # For this example, we'll generate random data
    n_samples = 100
    n_features = 30  # Breast cancer dataset has 30 features
    X = np.random.rand(n_samples, n_features)
    return X

@task
def make_predictions(model, X):
    predictions = model.predict(X)
    return predictions

@task
def save_results(predictions):
    # Save results to a local CSV file
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame({"prediction": predictions})
    df.to_csv("data/predictions.csv", index=False)
    print("Predictions saved to data/predictions.csv")

@flow
def batch_prediction_pipeline(run_id):
    model = load_model(run_id)
    data = load_data()
    predictions = make_predictions(model, data)
    save_results(predictions)

if __name__ == "__main__":
    # Replace with your actual run_id from MLflow
    run_id = "your-mlflow-run-id"
    batch_prediction_pipeline(run_id)