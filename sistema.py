import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Cargar dataset
df = pd.read_csv(r"C:\Users\LUISA\Downloads\MachineLearning\dataset_citas_random_forest.csv")

# 2. Variable objetivo
df["inasistencia"] = df["estado_cita"].apply(lambda x: 1 if x == "NO_ASISTIO" else 0)

# 3. Variables predictoras
X = df[
    [
        "edad",
        "sexo",
        "tipo_cita",
        "especialidad",
        "dia_semana",
        "hora",
        "antecedentes_inasistencias",
        "cantidad_citas_previas"
    ]
]

y = df["inasistencia"]

# 4. Columnas categóricas y numéricas
categorical_features = [
    "sexo",
    "tipo_cita",
    "especialidad",
    "dia_semana"
]

numeric_features = [
    "edad",
    "hora",
    "antecedentes_inasistencias",
    "cantidad_citas_previas"
]

# 5. Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# 6. Modelo Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# 7. Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# 8. Entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

# 9. Evaluación
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# 10. Guardar modelo
joblib.dump(
    pipeline,
    r"C:\Users\LUISA\Downloads\MachineLearning\modelo_random_forest_inasistencias.pkl"
)

print("\nModelo guardado como modelo_random_forest_inasistencias.pkl")

import os
print(os.getcwd())

# 11. Prueba con una cita nueva
nueva_cita = pd.DataFrame([{
    "edad": 35,
    "sexo": "FEMENINO",
    "tipo_cita": "CONSULTA",
    "especialidad": "Medicina General",
    "dia_semana": "LUNES",
    "hora": 10,
    "antecedentes_inasistencias": 1,
    "cantidad_citas_previas": 4
}])

prediccion = pipeline.predict(nueva_cita)[0]
probabilidad = pipeline.predict_proba(nueva_cita)[0][1]

print("\nPredicción nueva cita:")
print("Resultado:", "NO ASISTIRÁ" if prediccion == 1 else "ASISTIRÁ")
print("Probabilidad de inasistencia:", round(probabilidad * 100, 2), "%")