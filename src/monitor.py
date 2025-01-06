import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *
import os

def generate_evidently_report(reference_data, current_data, target_column):
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        ColumnDriftMetric(column_name=target_column),
        DatasetMissingValuesMetric(),
        DatasetCorrelationsMetric(),
    ])

    report.run(reference_data=reference_data, current_data=current_data)
    
    os.makedirs('reports', exist_ok=True)
    report.save_html("reports/model_monitoring_report.html")

def main():
    # Load reference data (training data)
    reference_data = pd.read_csv("data/reference_data.csv")

    # Load current data (new predictions)
    current_data = pd.read_csv("data/predictions.csv")

    # Generate and save the monitoring report
    generate_evidently_report(reference_data, current_data, target_column="prediction")

if __name__ == "__main__":
    main()