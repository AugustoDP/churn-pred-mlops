import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random


# Shuffle column values for "artificial new data"
# Randomly increase Monthly_Spending and Subscription_Length by a set range each
# giving the illusion of an artificial data drift, (inflation, time passed, etc.)
def generate_data(df):
    df = df.infer_objects(copy=False)

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("float64")
    
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

    df.fillna(0, inplace=True)


    # Modifica os dados completamente, gerando novo dataset, não queremos isso, apenas drift
    # for col in df.columns:
    #     if col != "Customer_ID":
    #         df[col] = df[col].sample(frac=1).reset_index(drop=True)

    
    df["Monthly_Spending"] = df["Monthly_Spending"].apply(lambda x: random.randrange(1, 20)* x)
    df["Subscription_Length"] = df["Subscription_Length"].apply(lambda x: random.randrange(3, 8)* x)


    return df


def main():

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_name = "customer_churn_dataset.csv"
    dataset_directory = os.path.join(project_root, "data", "raw")
    data_path = os.path.join(dataset_directory, dataset_name)
    # Read and process data
    df = pd.read_csv(data_path)
    new_df = generate_data(df)

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    processed_dir = project_root / "data" / "raw"

    new_df.to_csv(processed_dir / "new_customer_churn_dataset.csv", index=False)

    print(f"Concluído. Arquivos salvos em: {data_path}")


if __name__ == "__main__":
    main()