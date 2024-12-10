import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold

# Carregar os dados
data = pd.read_csv("treino_sinais_vitais.txt", header=None)
data.columns = [
    "id", "pressao_sistolica", "pressao_diastolica", "qPA", 
    "pulso", "respiracao", "gravidade", "rotulo"
]

# Separar recursos (features) e rótulos
X = data[["pressao_sistolica", "pressao_diastolica", "qPA", "pulso", "respiracao"]].values
y_gravity = data["gravidade"].values  # Regressão
y_label = data["rotulo"].values  # Classificação

# Normalizar os recursos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em treinamento e validação
X_train, X_val, y_grav_train, y_grav_val, y_label_train, y_label_val = train_test_split(
    X_scaled, y_gravity, y_label, test_size=0.2, random_state=42
)

# Defina a rede neural para regressão (gravidade)
def create_model():
    model = keras.Sequential([
    Dense(1024, input_dim=X_train.shape[1], activation="relu"),
    Dropout(0.2),
    Dense(24, activation="relu"),
    Dense(1, activation="linear")  # Saída para gravidade (regressão)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

fold_accuracies = []

for train_index, val_index in kf.split(X_train):
    # Divida os dados
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Crie e treine o modelo
    model = create_model()
    model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=16)
    
    # Avalie no fold de validação
    _, accuracy = model.evaluate(X_val_fold, y_val_fold)
    fold_accuracies.append(accuracy)

# Calcule a média da acurácia dos folds
mean_accuracy = np.mean(fold_accuracies)
print(f'Média de acurácia da validação cruzada: {mean_accuracy}')
