import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
list_of_models = ["RandomForestGridSearch", "XGBoost"]
for model_name in list_of_models:

    # Definir os limites de accuracy-score para Staging e Production
    staging_threshold = 0.56  # Apenas modelos acima deste accuracy-score vão para Staging

    # Buscar todas as versões do modelo
    versions = client.search_model_versions(f"name='{model_name}'")

    best_model = None  # Para armazenar o modelo Champion
    best_accuracy_score = 0  # Para rastrear o melhor accuracy

    for version in versions:
        run_id = version.run_id
        metrics = client.get_run(run_id).data.metrics

        if "accuracy" in metrics:
            accuracy = metrics["accuracy"]

            # Adicionar modelos qualificados para Staging
            if accuracy > staging_threshold:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Staging"
                )
                print(f"Modelo versão {version.version} com accuracy-score {accuracy} movido para Staging.")

            # Encontrar o melhor modelo para Produção
            if accuracy > best_accuracy_score:
                best_accuracy_score = accuracy
                best_model = version.version

    # Atualizar o Champion (Produção)
    if best_model:
        client.transition_model_version_stage(
            name=model_name,
            version=best_model,
            stage="Production"
        )
        print(f"Modelo versão {best_model} agora é o Champion com accuracy-score {best_accuracy_score}.")
    else:
        print("Nenhum modelo atende ao critério para ser Champion.")