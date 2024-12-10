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

# Configurar K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []  # Para armazenar os resultados de MAE em cada fold

for train_index, val_index in kf.split(X):
    # Dividir os dados em treino e validação
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y_gravity[train_index], y_gravity[val_index]
    
    # Redefinir o modelo para cada fold
    model = keras.Sequential([
        Dense(1024, input_dim=X_train.shape[1], activation="relu"),
        Dropout(0.2),
        Dense(24, activation="relu"),
        Dense(1, activation="linear")
    ])
    
    # Compilar o modelo
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    # Treinar o modelo
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    
    # Avaliar o modelo
    loss, mae = model.evaluate(X_val, y_val, verbose=0)
    mae_scores.append(mae)  # Armazenar a métrica MAE

# Exibir os resultados médios de MAE
print(f"Mean Absolute Error (MAE) médio nos 5 folds: {np.mean(mae_scores):.4f}")

accuracy_scores = []  # Para armazenar os resultados de acurácia em cada fold

for train_index, val_index in kf.split(X_scaled):
    # Dividir os dados em treino e validação
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y_label[train_index], y_label[val_index]
    
    # Converter rótulos para one-hot encoding
    y_train_one_hot = to_categorical(y_train - 1, num_classes=4)
    y_val_one_hot = to_categorical(y_val - 1, num_classes=4)
    
    # Redefinir o modelo para cada fold
    model_classification = keras.Sequential([
        Dense(64, input_dim=X_train.shape[1], activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(4, activation="softmax")
    ])
    
    # Compilar o modelo
    model_classification.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Treinar o modelo
    model_classification.fit(X_train, y_train_one_hot, epochs=50, batch_size=16, verbose=0)
    
    # Avaliar o modelo
    loss, accuracy = model_classification.evaluate(X_val, y_val_one_hot, verbose=0)
    accuracy_scores.append(accuracy)  # Armazenar a métrica de acurácia

# Exibir os resultados médios de acurácia
print(f"Acurácia média nos 5 folds: {np.mean(accuracy_scores):.4f}")
