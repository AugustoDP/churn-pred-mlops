import pandas as pd
import numpy as np
import requests
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, RegressionPreset, ClassificationPreset
from sklearn.preprocessing import LabelEncoder
import os
import sys
import json
from auxiliary_functions import get_predictions, preprocess_data


def check_for_drift(drift_score, drift_by_columns, dataset_name):
    num_columns_drift = sum(1 for col, values in drift_by_columns.items() if values.get("drift_detected", False))
    if drift_score > 0.5:
        print("Drift detectado no Dataset")
        os.system(f"start /wait cmd /k py train.py {dataset_name}")
    else:
        if num_columns_drift > 2:
            print(f"Drift detectado em {num_columns_drift} colunas! Treinando novo modelo...")
            os.system(f"start /wait cmd /k py train.py {dataset_name}")
        else:
            print("Modelo ainda está bom, sem necessidade de re-treinamento.")
            print("Nenhum drift detectado nas colunas e no dataset")

def load_new_data(dataset_name):
    df = pd.read_csv(dataset_name)
    df = df.sample(1000)  # Pegamos exemplos aleatórios para testar
    X, y, _ = preprocess_data(df)
    return X, y

# Avaliar degradação do modelo
def evaluate_model(df, y, new_data):
    if new_data is None:
        print("Avaliando modelo com dados originais")
        df["prediction"] = get_predictions(df)
        df["prediction"] = df["prediction"].astype(int)
        print(df["prediction"].unique())
        df["target"] = y
        print(df["target"].unique())
        report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
        report.run(reference_data=df, current_data=df)
        report.save_html("monitoring_report_df.html")
        report_dict = report.as_dict()
        drift_score = report_dict["metrics"][0]["result"]["dataset_drift"]
        print(f"Score de drift: {drift_score}")
        drift_by_columns = report_dict["metrics"][1]["result"].get("drift_by_columns", {})
        print(f"Coluns drift: {drift_by_columns}")
        return drift_score, drift_by_columns
    else:
        print("Avaliando modelo com dados artificiais")
        new_data["prediction"] = get_predictions(new_data)
        new_data["prediction"] = new_data["prediction"].astype(int)
        print(new_data["prediction"].unique())
        new_data["target"] = y
        print(new_data["target"].unique())
        report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
        report.run(reference_data=df, current_data=new_data)
        report.save_html("monitoring_report_df_new_data.html")
        report_dict = report.as_dict()
        drift_score = report_dict["metrics"][0]["result"]["dataset_drift"]
        print(f"Score de drift: {drift_score}")
        drift_by_columns = report_dict["metrics"][1]["result"].get("drift_by_columns", {})
        print(f"Coluns drift: {drift_by_columns}")
        return drift_score, drift_by_columns


def monitor_data_drift(dataset_name, new_dataset_name):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Primeiro testa o modelo com si mesmo
    data_path = os.path.join(project_root, "data", "raw", dataset_name)
    df_examples, y = load_new_data(data_path)
    drift_score, drift_by_columns = evaluate_model(df_examples, y, None)
    # Em seguida testa o modelo com novos dados
    new_data_path = os.path.join(project_root, "data", "raw", new_dataset_name)
    new_data, _ = load_new_data(new_data_path)
    drift_score, drift_by_columns = evaluate_model(df_examples, y, new_data)
    check_for_drift(drift_score, drift_by_columns, new_dataset_name)

def main():
    if len(sys.argv) > 2:
        dataset_name = sys.argv[1]
        new_dataset_name = sys.argv[2]
        monitor_data_drift(dataset_name, new_dataset_name)
        
    else:
        print("Incorrect calling method. \n")
        print("Usage: py monitor.py \"old_dataset_used_to_train.csv\" \"new_dataset_to_check.csv\"")

if __name__ == "__main__":
    main()



