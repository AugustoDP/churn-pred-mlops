import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import requests

def preprocess_data(df):
    df.drop(columns=["Customer_ID"], inplace=True, errors="ignore")
    df = df.infer_objects(copy=False)

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("float64")
    
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

    df.fillna(0, inplace=True)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    mlflow_dataset = mlflow.data.from_pandas(df, targets="Churn")

    return X, y.astype(int), mlflow_dataset


def get_predictions(data):
    print(data.head())

    # Defina as colunas esperadas pelo modelo
    columns = [
        "Age", "Gender", "Monthly_Spending", "Subscription_Length", "Support_Interactions", 
    ]
    
    # Crie uma lista de dicionários, onde cada dicionário representa uma instância
    instances = []
    for _, row in data.iterrows():
        instance = {col: row[col] for col in columns}
        instances.append(instance)


    url = "http://127.0.0.1:8000/invocations"
    headers = {"Content-Type": "application/json"}
    payload = {"instances": instances}
    
    response = requests.post(url, headers=headers, json=payload)
    predictions = response.json()
    predictions = predictions.get("predictions")
    print(predictions)
    return predictions