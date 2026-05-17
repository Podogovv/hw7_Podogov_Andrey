from flask import Flask, request, jsonify

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Загрузка данных
iris = load_iris()

X = iris.data
y = iris.target

# Гиперпараметры
hyperparameters = {
    "n_estimators": 100,
    "random_state": 42
}

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Обучение модели
model = RandomForestClassifier(**hyperparameters)

model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Метрика
accuracy = accuracy_score(y_test, y_pred)

print(f"Точность accuracy: {accuracy:.2f}")

# Health endpoint
@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "accuracy": round(float(accuracy), 2)
    })

# Predict endpoint
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = data["x"]

    # Проверка количества признаков
    if len(features) != 4:

        return jsonify({
            "error": "Expected 4 features"
        }), 400

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]

    return jsonify({
        "prediction": int(prediction)
    })

# Запуск сервиса
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )