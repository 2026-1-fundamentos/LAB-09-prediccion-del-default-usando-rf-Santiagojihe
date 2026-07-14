# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import gzip
import os
import pickle
import json
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def clean_data(df):
    df = df.copy()

    # Paso 1. Renombrar la columna objetivo
    df = df.rename(columns={"default payment next month": "default"})

    # Remueva la columna ID
    df = df.drop(columns=["ID"])

    # Elimine los registros con informacion no disponible (0 representa N/A)
    #df = df[df["EDUCATION"] != 0]
    #df = df[df["MARRIAGE"] != 0]

    # Agrupar los valores > 4 de EDUCATION en la categoría 4 ("others")
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4

    return df


def pregunta_01():

    train = pd.read_csv(
        "files/input/train_default_of_credit_card_clients.csv"
    )

    test = pd.read_csv(
        "files/input/test_default_of_credit_card_clients.csv"
    )

    train = clean_data(train)
    print(train["EDUCATION"].value_counts().sort_index())
    print(train["MARRIAGE"].value_counts().sort_index())

    test = clean_data(test)

    # -----------------------------------------
    # Separar variables
    # -----------------------------------------
    x_train = train.drop(columns=["default"])
    y_train = train["default"]

    x_test = test.drop(columns=["default"])
    y_test = test["default"]

    print(x_train.shape)
    print(y_train.shape)

    print(x_test.shape)
    print(y_test.shape)

    # -----------------------------------------
    # Variables categóricas
    # -----------------------------------------
    categorical_features = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]

    # -----------------------------------------
    # Preprocesador
    # -----------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )
    print(preprocessor)

    # -----------------------------------------
    # Pipeline
    # -----------------------------------------
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=12345,
                    
                ),
            ),
        ]
    )
    print(pipeline)

    # -----------------------------------------
    # Grid Search
    # -----------------------------------------
    param_grid = {
        "classifier__n_estimators": [200],
        "classifier__max_depth": [None],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
    }

    model = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    print(model)
    # -----------------------------------------
    # Entrenar el modelo
    # -----------------------------------------
    model.fit(x_train, y_train)
    print("Entrenamiento finalizado")
    
     # -----------------------------------------
    # Crear carpeta de salida
    # -----------------------------------------
    os.makedirs("files/models", exist_ok=True)
    

    # -----------------------------------------
    # Guardar modelo
    # -----------------------------------------
    with gzip.open(
        "files/models/model.pkl.gz",
        "wb",
    ) as file:

        pickle.dump(model, file)
        print("Modelo guardado")

    
    # -----------------------------------------
    # Predicciones
    # -----------------------------------------
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)


    os.makedirs("files/output", exist_ok=True)

    metrics = []

    metrics.append({
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, train_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_train, train_pred)
        ),
        "recall": float(recall_score(y_train, train_pred)),
        "f1_score": float(f1_score(y_train, train_pred)),
    })

    metrics.append({
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, test_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, test_pred)
        ),
        "recall": float(recall_score(y_test, test_pred)),
        "f1_score": float(f1_score(y_test, test_pred)),
    })

    cm_train = confusion_matrix(y_train, train_pred)
    cm_test = confusion_matrix(y_test, test_pred)

    metrics.append({
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(cm_train[0, 0]),
            "predicte_1": int(cm_train[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_train[1, 0]),
            "predicted_1": int(cm_train[1, 1]),
        },
    })

    metrics.append({
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(cm_test[0, 0]),
            "predicte_1": int(cm_test[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_test[1, 0]),
            "predicted_1": int(cm_test[1, 1]),
        },
    })

    with open(
        "files/output/metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        for item in metrics:
            file.write(json.dumps(item) + "\n")

    print("Métricas guardadas")

if __name__ == "__main__":
    pregunta_01()





    
    