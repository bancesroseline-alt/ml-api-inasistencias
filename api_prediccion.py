from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

modelo = joblib.load(
    "modelo_random_forest_inasistencias.pkl"
)

class PrediccionRequest(BaseModel):
    edad: int
    sexo: str
    tipo_cita: str
    especialidad: str
    dia_semana: str
    hora: int
    antecedentes_inasistencias: int
    cantidad_citas_previas: int


@app.post("/predict")
def predict(data: PrediccionRequest):

    df = pd.DataFrame([{
        "edad": data.edad,
        "sexo": data.sexo,
        "tipo_cita": data.tipo_cita,
        "especialidad": data.especialidad,
        "dia_semana": data.dia_semana,
        "hora": data.hora,
        "antecedentes_inasistencias": data.antecedentes_inasistencias,
        "cantidad_citas_previas": data.cantidad_citas_previas
    }])

    prediccion = modelo.predict(df)[0]
    probabilidad = float(modelo.predict_proba(df)[0][1])

    return {
        "prediccion": int(prediccion),
        "probabilidad": probabilidad
    }