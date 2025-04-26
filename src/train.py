import os
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from auxiliary_functions import preprocess_data
from churn_promote import promote_best_model
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("customerchurn_experiment")


# def train_and_log(model, model_name, X_train, X_test, y_train, y_test):
#     with mlflow.start_run():
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         y_proba = model.predict_proba(X_test)[:, 1]
#         accuracy = accuracy_score(y_test, y_pred)
#         auc = roc_auc_score(y_test, y_proba)
#         report = classification_report(y_test, y_pred)
#         mlflow.log_param("model", model_name)
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.log_metric("auc", auc)
#         mlflow.log_text(report, "classification_report.txt")
#         input_example = X_train.iloc[[0]]
#         signature = infer_signature(X_train, y_pred)
#         mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=input_example, registered_model_name=model_name)
#         print(f"{model_name} - Accuracy: {accuracy} | AUC: {auc}")
#         print(report)


def train_xgboost(X, y, dataset_name, mlflow_dataset):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 10, None],
        "learning_rate": [0.05, 0.1, 0.2],
    }

    rf = xgb.XGBClassifier(random_state=42)

    for params in (dict(zip(param_grid.keys(), values)) for values in 
                   [(n, d, s) for n in param_grid["n_estimators"] 
                               for d in param_grid["max_depth"] 
                               for s in param_grid["learning_rate"]]):
        
        rf.set_params(**params)
        rf.fit(X_train, y_train)
        y_test_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)

        with mlflow.start_run(run_name=f"XGB_{params['n_estimators']}_{params['max_depth']}_{params['learning_rate']}"):
            mlflow.log_input(mlflow_dataset, context="training")
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.set_tag("dataset_used", dataset_name)

            signature = infer_signature(X_train, y_test_pred)
            model_info = mlflow.sklearn.log_model(rf, "XGBoost_model", 
                                     signature=signature, 
                                     input_example=X_train, 
                                     registered_model_name="XGBoost")

            loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
            predictions = loaded_model.predict(X_test)
            result = pd.DataFrame(X_test, columns=X.columns.values)
            result["label"] = y_test.values
            result["predictions"] = predictions

            mlflow.evaluate(
                data=result,
                targets="label",
                predictions="predictions",
                model_type="classifier",
            )

            print(result[:5])

    return rf

def train_random_forest(X, y, dataset_name, mlflow_dataset):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    rf = RandomForestClassifier(random_state=42)

    for params in (dict(zip(param_grid.keys(), values)) for values in 
                   [(n, d, s) for n in param_grid["n_estimators"] 
                               for d in param_grid["max_depth"] 
                               for s in param_grid["min_samples_split"]]):
        
        rf.set_params(**params)
        rf.fit(X_train, y_train)
        y_test_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)

        with mlflow.start_run(run_name=f"RF_{params['n_estimators']}_{params['max_depth']}_{params['min_samples_split']}"):
            mlflow.log_input(mlflow_dataset, context="training")
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.set_tag("dataset_used", dataset_name)

            signature = infer_signature(X_train, y_test_pred)
            model_info = mlflow.sklearn.log_model(rf, "random_forest_model", 
                                     signature=signature, 
                                     input_example=X_train, 
                                     registered_model_name="RandomForestGridSearch")

            loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
            predictions = loaded_model.predict(X_test)
            result = pd.DataFrame(X_test, columns=X.columns.values)
            result["label"] = y_test.values
            result["predictions"] = predictions

            mlflow.evaluate(
                data=result,
                targets="label",
                predictions="predictions",
                model_type="classifier",
            )

            print(result[:5])

    return rf

def main():

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = "customer_churn_dataset.csv"
    data_path = os.path.join(project_root, "data", "raw", dataset_name)
    # Read and process data
    df = pd.read_csv(data_path)
    X, y, mlflow_dataset = preprocess_data(df)


    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    X.to_csv(processed_dir / "X.csv", index=False)
    y.to_csv(processed_dir / "y.csv", index=False)

    print(f"Concluído. Arquivos salvos em: {processed_dir}")

    _ = train_random_forest(X, y, dataset_name, mlflow_dataset)
    _ = train_xgboost(X, y, dataset_name, mlflow_dataset)

    promote_best_model()


if __name__ == "__main__":
    main()
