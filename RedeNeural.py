import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

# Carregar os dados
data = pd.read_csv("treino_sinais_vitais_com_label.txt", header=None)
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

# Definir a rede neural para regressão (gravidade)
model = keras.Sequential([
    Dense(1024, input_dim=X_train.shape[1], activation="relu"),
    Dropout(0.2),
    Dense(24, activation="relu"),
    Dense(1, activation="linear")  # Saída para gravidade (regressão)
])

# Compilar o modelo de regressão
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Treinar a rede para prever gravidade
model.fit(X_train, y_grav_train, validation_data=(X_val, y_grav_val), epochs=2000, batch_size=512)

# Avaliar o modelo de regressão
loss, mae = model.evaluate(X_val, y_grav_val)
print(f"Mean Absolute Error: {mae}")

'''
# Converter rótulos para one-hot encoding
# Subtrair 1 dos rótulos para ajustar para o formato 0, 1, 2, 3
y_label_train_one_hot = to_categorical(y_label_train - 1, num_classes=4)
y_label_val_one_hot = to_categorical(y_label_val - 1, num_classes=4)

# Modelo para prever rótulos (classificação multiclasse)
model_classification = keras.Sequential([
    Dense(64, input_dim=X_train.shape[1], activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(4, activation="softmax")  # Saída para 4 classes
])


model_classification.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Treinar o modelo de classificação
model_classification.fit(X_train, y_label_train_one_hot, validation_data=(X_val, y_label_val_one_hot), epochs=50, batch_size=16)

# Avaliar o modelo de classificação
loss, accuracy = model_classification.evaluate(X_val, y_label_val_one_hot)
print(f"Acurácia: {accuracy:.2f}")
'''