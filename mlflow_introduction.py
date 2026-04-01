import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Generating synthetic data for linear regression example
sample_size = 100
X = np.linspace(0, 10, sample_size).reshape(-1, 1)
noise = 10
y_true = 2 * X + 5
y = y_true + noise * np.random.randn(sample_size, 1)

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'MSE: {mse:.4f}')
print(f'R2: {r2:.4f}')
print(f"Coeficiente: {str(map('{:.4f}%'.format,model.coef_[0]))}")
print(f"Intercepto: {str(map('{:.4f}%'.format,model.intercept_))}")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Dados')
plt.plot(X, y_pred, color='red', linewidth=2, label='Previsão')
plt.plot(X, y_true, color='green', linestyle='--', linewidth=2, label='Verdadeiro')
plt.title(f'Regressão Linear (Ruído: {noise})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nR2: {r2:.4f}',
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.show()

# MLFlow for tracking the experiment with different noise levels

EXPERIMENT_NAME = "Hello-World-MLFlow"
mlflow.set_experiment(EXPERIMENT_NAME)

noise_levels = [5, 10, 20]

for alpha in noise_levels:
    # Gerando dados sintéticos
    sample_size = 100
    X = np.linspace(0, 10, sample_size).reshape(-1, 1)
    y_true = 2 * X + 5
    y = y_true + alpha * np.random.randn(sample_size, 1)
    
    # Iniciando um run do MLflow
    with mlflow.start_run(run_name=f"ruido_{alpha}"):
        print(f"Executando experimento com ruído = {alpha}")

        # Registrando parâmetros
        mlflow.log_param("nivel_ruido", alpha)
        mlflow.log_param("n_amostras", sample_size)
        
        # Treinando o modelo
        modelo = LinearRegression()
        modelo.fit(X, y)
        
        # Fazendo previsões
        y_pred = modelo.predict(X)
        
        # Calculando métricas
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Registrando métricas
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("coeficiente", modelo.coef_[0][0])
        mlflow.log_metric("intercepto", modelo.intercept_[0])
        
        # Visualizando os resultados
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.5, label='Dados')
        plt.plot(X, y_pred, color='red', linewidth=2, label='Previsão')
        plt.plot(X, y_true, color='green', linestyle='--', linewidth=2, label='Verdadeiro')
        plt.title(f'Regressão Linear (Ruído: {alpha})')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        
        # Adicionando texto com métricas
        plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nR2: {r2:.4f}',
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Salvando a figura
        plt.savefig("regressao_linear.png")
        plt.close()
        
        # Registrando o modelo
        mlflow.sklearn.log_model(modelo, "modelo")
        
        # Registrando a figura como artefato
        mlflow.log_artifact("regressao_linear.png")
        
        print(f"MSE: {mse:.4f}, R2: {r2:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("-" * 50)
