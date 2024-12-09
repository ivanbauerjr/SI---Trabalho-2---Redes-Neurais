import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout

#from tensorflow.python.keras.layers import Dense, Dropout
#from tensorflow.python.keras.engine.sequential import Sequential

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

# Definir a rede neural
model = keras.Sequential([
    Dense(64, input_dim=X_train.shape[1], activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")  # Saída para gravidade (regressão)
])

# Compilar o modelo
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Treinar a rede para prever gravidade
model.fit(X_train, y_grav_train, validation_data=(X_val, y_grav_val), epochs=50, batch_size=16)

# Avaliar o modelo de regressão
loss, mae = model.evaluate(X_val, y_grav_val)
print(f"Mean Absolute Error: {mae}")

# Modelo para prever rótulos
model_classification = keras.Sequential([
    Dense(64, input_dim=X_train.shape[1], activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")  # Saída binária (0 ou 1)
])

model_classification.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Treinar o modelo de classificação
model_classification.fit(X_train, y_label_train, validation_data=(X_val, y_label_val), epochs=50, batch_size=16)

# Prever no dataset de teste cego
test_data = pd.read_csv("teste_cego.txt", header=None)
test_data.columns = ["id", "pressao_sistolica", "pressao_diastolica", "qPA", "pulso", "respiracao", "gravidade"]

X_test = scaler.transform(test_data[["pressao_sistolica", "pressao_diastolica", "qPA", "pulso", "respiracao"]])


# Previsões
gravity_predictions = model.predict(X_test)
label_predictions = model_classification.predict(X_test).round().astype(int)

# Salvar as previsões
output = pd.DataFrame({
    "Gravidade": gravity_predictions.flatten(),
    "Classe": label_predictions.flatten()
})
output.to_csv("predictions.csv", index=False, header=False)

print("Previsões salvas em 'predictions.csv'.")
