import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import warnings

warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "demo_mlflow_educativo"
MODEL_NAME = "iris_classifier_educativo"

# mlflow ui --backend-store-uri sqlite:///mlflow_educativo.db
mlflow.set_tracking_uri("sqlite:///mlflow_educativo.db")

# experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

feature_names = iris.feature_names
target_names = iris.target_names

parent_run = mlflow.start_run(run_name="parent_run_model_comparison")

mlflow.log_param("experiment_type", "model_comparison")
mlflow.log_param("models_tested", 2)
mlflow.log_param("dataset", "iris")
mlflow.log_param("num_classes", len(target_names))
mlflow.log_param("feature_names", feature_names)

best_model = None
best_accuracy = 0
best_run_id = None

print(parent_run.info.run_id)

rf_params = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 7}
]

for params in rf_params:
    with mlflow.start_run(run_name=f"rf_test_{params['n_estimators']}_trees", nested=True, experiment_id=experiment_id):
        # Treina o modelo Random Forest com os parâmetros atuais
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Faz predições no conjunto de teste
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Registra os hiperparâmetros testados
        mlflow.log_params(params)

        # Registra métricas de avaliação
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("n_features", X_train.shape[1]) # Número de features usadas

        # Salva o modelo treinado como artefato da run
        mlflow.sklearn.log_model(model, "random_forest")

        # Mostra resultado no console
        print(f"RF com {params['n_estimators']} árvores: accuracy = {accuracy:.4f}")

        # Verifica se é o melhor modelo até agora
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_run_id = mlflow.active_run().info.run_id


lr_params = [
    {"C": 0.1, "max_iter": 1000}, # Regularização forte (penaliza mais os coeficientes)
    {"C": 1.0, "max_iter": 1000}, # Configuração padrão
    {"C": 10.0, "max_iter": 1000}, # Regularização fraca (coeficientes podem ser maiores)
]

for params in lr_params:
    # Inicia run aninhado - run filho dentro da run pai
    with mlflow.start_run(run_name=f"lr_test_C_{params['C']}", nested=True, experiment_id=experiment_id):
        # Treina o modelo de Regressão Logística com os parâmetros definidos
        model = LogisticRegression(**params, random_state=42)
        model.fit(X_train, y_train)

        # Faz predições no conjunto de teste
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Registra os hiperparâmetros usados
        mlflow.log_params(params)
        
        # Registra métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("n_features", X_train.shape[1]) # Número de features do dataset
        
        # Salva o modelo treinado como artefato
        mlflow.sklearn.log_model(model, "logistic_regression")
        
        # Exibe resultado no console
        print(f"LR com C={params['C']}: accuracy = {accuracy:.4f}")
        
        # Atualiza informações do melhor modelo se necessário
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_run_id = mlflow.active_run().info.run_id

# Registra informações do melhor modelo na run pai
mlflow.log_metric("best_accuracy", best_accuracy)
mlflow.set_tag("best_child_run_id", best_run_id)
mlflow.set_tag("best_model_type", type(best_model).__name__)

# Exibe no console os resultados finais
print(f"\nMelhor modelo obteve accuracy de {best_accuracy:.4f}")
print(f"Run ID do melhor modelo: {best_run_id}")

# Finaliza a run pai
mlflow.end_run()

print("\nNested runs concluídos!")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="autolog_gridsearch", experiment_id=experiment_id):
    print(f"Executando GridSearchCV com autologging...\n")

    # Define grid de hiperparâmetros
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }

    # Configura GridSearchCV
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    # Treina - o autologging captura tudo automaticamente!
    grid_search.fit(X_train, y_train)

    # Predições no conjunto de teste
    predictions = grid_search.predict(X_test)
    
    # Calcula métricas adicionais
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    
    # Log de métricas customizadas (além das capturadas automaticamente)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)

    # Salva o run_id para uso posterior
    grid_run_id = mlflow.active_run().info.run_id

    print("\n# Resultados do GridSearchCV:")
    print(f"# Melhor configuração: {grid_search.best_params_}")
    print(f"# Melhor score CV: {grid_search.best_score_:.4f}")
    print(f"# Acuracy no teste: {accuracy:.4f}")
    print(f"# Precision: {precision:.4f}")
    print(f"# Recall: {recall:.4f}")
    print(f"# F1-Score: {f1:.4f}")

    # Guarda o melhor modelo
    best_grid_model = grid_search.best_estimator_


mlflow.sklearn.autolog(disable=True)

client = mlflow.MlflowClient()
model_uri = f"runs:/{grid_run_id}/model"
model_version = mlflow.register_model(model_uri, MODEL_NAME)
version = model_version.version

print(model_version)
print(version)

print("Adicionando metadados ao modelo...\n")

# Adiciona descrição
client.update_model_version(
    name=MODEL_NAME,
    version=version,
    description="Modelo de classificação Iris usando Random Forest otimizado com GridSearchCV"
)

# Adiciona tags
tags_to_add = {
    "algorithm": "RandomForest",
    "dataset": "iris",
    "framework": "scikit-learn",
    "optimization": "GridSearchCV",
    "author": "MLflow Demo"
}

for key, value in tags_to_add.items():
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key=key,
        value=value
    )
    print(f"Tag adicionada: {key} = {value}")
    
print("\nMetadados adicionados com sucesso!")

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version,
    stage="Staging",
    archive_existing_versions=False
)

print("Validando modelo em Staging...\n")

# Carrega modelo de Staging
staging_model = mlflow.sklearn.load_model(
    model_uri=f"models:/{MODEL_NAME}/Staging"
)

# Faz predições de teste
test_predictions = staging_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"Resultados da validação")
print(f"Acuracy no teste: {test_accuracy:.4f}")

# Simula validações de negócio
validation_checks = {
    "Validação de performance": test_accuracy > 0.9,
    "Validação de latência": True, # Simulado
    "Validação de negócio": True, # Simulado
    "Aprovação do time": True, # Simulado
}

all_passed = all(validation_checks.values())
print(f"\nChecklist de validação:")
for check, passed in validation_checks.items():
    status = "OK" if passed else "Falhou"
    print(f"- {check}: {status}")

if all_passed:
    print("\nModelo aprovado para produção!")
else:
    print("\nModelo precisa de ajustes!")


# Transiciona o modelo para o estágio de Produção
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version,
    stage="Production",
    archive_existing_versions=True
)

# Obtém detalhes do modelo
model_version_details = client.get_model_version(MODEL_NAME, version)

print("Informações do modelo registrado:")
print(f"Nome: {model_version_details.name}")
print(f"Versão: {model_version_details.version}")
print(f"Stage atual: {model_version_details.current_stage}")
print(f"Status: {model_version_details.status}")
print(f"Run ID: {model_version_details.run_id}")
print(f"Criado em: {model_version_details.creation_timestamp}")
print(f"Última atualização: {model_version_details.last_updated_timestamp}")

print("Carregando modelo de produção...\n")

# Carrega modelo de produção
production_model = mlflow.sklearn.load_model(
    model_uri=f"models:/{MODEL_NAME}/Production"
)

print("Modelo carregado com sucesso!")
print(f"Tipo: {type(production_model)}")
print(f"Parâmetros: {production_model.get_params()}")

# Exemplos de novas amostras para classificação
new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2], # Provavelmente Setosa
    [6.2, 3.4, 5.4, 2.3], # Provavelmente Virginica
    [5.9, 3.0, 4.2, 1.5] # Provavelmente Versicolor
])

# Faz predições
predictions = production_model.predict(new_samples)
probabilities = production_model.predict_proba(new_samples)

print("Predições do modelo de produção:\n")
for i, (sample, pred, probs) in enumerate(zip(new_samples, predictions, probabilities)):
    print(f"Amostra [{i+1}]: {sample}")
    print(f"Classe predita: {target_names[pred]}")
    print(f"Probabilidades: {dict(zip(target_names, probs.round(3)))}\n")
    print()
