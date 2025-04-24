import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException# Import the PyCaret Regression module
from pydantic import BaseModel
import pandas as pd
import os# Load the environment variables from the .env file into the application
import mlflow
import requests
from mlflow.tracking import MlflowClient
from auxiliary_functions import get_predictions

class churnSample(BaseModel):
    Age: float
    Gender: float
    Monthly_Spending: float
    Subscription_Length: float
    Support_Interactions: float

app = FastAPI()# Create a class to store the deployed model & use it for prediction


@app.post("/predict_csv")
async def create_upload_file(file: UploadFile = File(...)):
    # Handle the file only if it is a CSV
    if file.filename.endswith(".csv"):
        # Create a temporary file with the same name as the uploaded 
        # CSV file to load the data into a pandas Dataframe
        with open(file.filename, "wb")as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)        # Return a JSON object containing the model predictions
        return get_predictions(data)
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request 
        # (you can learn more about HTTP response status codes here)
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")# Check if the environment variables for AWS access are available. 


@app.post("/predict")
# Fazer previsões com o modelo
async def call_predict(data: churnSample):
    data = data.model_dump()
    data = pd.DataFrame([data])
    columns = [
        "Age", "Gender", "Monthly_Spending", "Subscription_Length", "Support_Interactions", 
    ]
    
    # Crie uma lista de dicionários, onde cada dicionário representa uma instância
    instances = []
    for _, row in data.iterrows():
        instance = {col: row[col].item() for col in columns}
        instances.append(instance)
    url = "http://127.0.0.1:8000/invocations"
    headers = {"Content-Type": "application/json"}
    payload = {"instances": instances}
    print(payload)
    response = requests.post(url, headers=headers, json=payload)
    
    predictions = response.json()
    predictions = predictions.get("predictions")
    print(predictions)
    return predictions