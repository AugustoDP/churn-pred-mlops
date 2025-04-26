@echo off
if exist "%~dp0mlops_venv\" (
echo Folder already exists
call mlops_venv\Scripts\activate
set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
) else (
py -m venv mlops_venv
pause
call mlops_venv\Scripts\activate
pip install -r requirements.txt
set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
)


