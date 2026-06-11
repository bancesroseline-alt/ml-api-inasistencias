import joblib
import pandas as pd

# cargar modelo entrenado
modelo = joblib.load("modelo_random_forest_inasistencias.pkl")

# datos del paciente
nueva_cita = pd.DataFrame([{
    "edad": 35,
    "sexo": "FEMENINO",
    "tipo_cita": "CONSULTA",
    "especialidad": "Medicina General",
    "dia_semana": "LUNES",
    "hora": 10,
    "antecedentes_inasistencias": 2,
    "cantidad_citas_previas": 8
}])

# predicción
prediccion = modelo.predict(nueva_cita)[0]
probabilidad = modelo.predict_proba(nueva_cita)[0][1]

resultado = "NO_ASISTIRA" if prediccion == 1 else "ASISTIRA"

print("Resultado:", resultado)
print("Probabilidad:", round(probabilidad * 100, 2), "%")
