"""API de riesgo de mercado — Diplomatura ML para Riesgo (Universidad Austral).

Sirve la probabilidad de un MOVIMIENTO grande del S&P 500: |retorno a 5 ruedas| > 2,5%.
Uso educativo: no es una recomendación de inversión.
"""
import json
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

from .datos import CacheDatos
from .features import FEATURES, HORIZONTE, UMBRAL, construir_features

RAIZ = Path(__file__).resolve().parent.parent
META = json.loads((RAIZ / "modelo" / "meta.json").read_text(encoding="utf-8"))
MODELO = XGBClassifier()
MODELO.load_model(RAIZ / "modelo" / "modelo_movimiento.json")
assert META["features"] == FEATURES, "El modelo se entrenó con otra lista de features"

PERCENTILES = np.array(META["percentiles_probabilidad"]["valores"])   # p0..p100 en el entrenamiento
PREVALENCIA = META["prevalencia_entrenamiento"]
AVISO = "Uso educativo. No es una recomendación de inversión."

app = FastAPI(
    title="API de riesgo de mercado · ML para Riesgo (Austral)",
    description=(
        "Probabilidad de que el S&P 500 se mueva más de 2,5% (en cualquier dirección) en las próximas "
        "5 ruedas. Modelo XGBoost entrenado en la Clase 7. " + AVISO
    ),
    version=META["version"],
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

cache = CacheDatos()
_predicciones = {"clave": None, "tabla": None}


def tabla_predicciones():
    """Calcula features y probabilidades para toda la historia (se recalcula al refrescar los datos)."""
    datos, fuente, error = cache.obtener()
    clave = (id(datos), len(datos), datos.index[-1])
    if _predicciones["clave"] != clave:
        X = construir_features(datos)
        validas = X.notna().all(axis=1)
        tabla = pd.DataFrame(index=X.index)
        tabla["probabilidad"] = np.nan
        tabla.loc[validas, "probabilidad"] = MODELO.predict_proba(X.loc[validas, FEATURES])[:, 1]
        ret_futuro = datos["sp500"].shift(-HORIZONTE) / datos["sp500"] - 1
        tabla["retorno_5d_realizado"] = ret_futuro
        tabla = tabla.join(X[["vix", "vol_20d", "umbral_sigmas_vix"]]).join(datos["sp500"])
        _predicciones.update(clave=clave, tabla=tabla.dropna(subset=["probabilidad"]))
    return _predicciones["tabla"], fuente, error


def nivel(percentil: float) -> str:
    if percentil >= 95:
        return "muy alto"
    if percentil >= 80:
        return "alto"
    if percentil >= 50:
        return "normal"
    return "bajo"


def armar_respuesta(fecha_pedida, fila_fecha, fila, fuente, error):
    p = float(fila["probabilidad"])
    pct = float(np.interp(p, PERCENTILES, np.arange(101)))
    ret = fila["retorno_5d_realizado"]
    realizado = None
    if pd.notna(ret):
        realizado = {"retorno_5d": round(float(ret), 4), "hubo_movimiento": bool(abs(ret) > UMBRAL)}
    return {
        "fecha_pedida": fecha_pedida,
        "fecha_datos": fila_fecha.date().isoformat(),
        "evento": f"|retorno del S&P 500 a {HORIZONTE} ruedas| > {UMBRAL:.1%}",
        "probabilidad_movimiento": round(p, 4),
        "prevalencia_historica": round(PREVALENCIA, 4),
        "veces_la_prevalencia": round(p / PREVALENCIA, 2),
        "percentil_historico": round(pct, 1),
        "nivel": nivel(pct),
        "contexto": {
            "sp500": round(float(fila["sp500"]), 2),
            "vix": round(float(fila["vix"]), 2),
            "vol_realizada_20d": round(float(fila["vol_20d"]), 4),
            "umbral_en_sigmas_vix": round(float(fila["umbral_sigmas_vix"]), 2),
        },
        "realizado": realizado,
        "fuente_datos": fuente,
        "aviso_datos": error,
        "modelo_version": META["version"],
        "aviso": AVISO,
    }


@app.get("/", summary="Información general")
def inicio():
    return {
        "api": app.title,
        "evento": f"|retorno del S&P 500 a {HORIZONTE} ruedas| > {UMBRAL:.1%}",
        "endpoints": {
            "/salud": "estado del servicio",
            "/modelo": "ficha del modelo (hiperparámetros, métricas, features)",
            "/prediccion/hoy": "probabilidad con el último cierre disponible",
            "/prediccion/{fecha}": "probabilidad para una fecha histórica (AAAA-MM-DD)",
            "/historial?desde=AAAA-MM-DD&hasta=AAAA-MM-DD": "serie de probabilidades (máx. 3 años)",
            "/docs": "documentación interactiva (Swagger)",
        },
        "aviso": AVISO,
    }


@app.get("/salud", summary="Estado del servicio")
def salud():
    tabla, fuente, error = tabla_predicciones()
    return {"estado": "ok", "modelo_version": META["version"], "ultimo_dato": tabla.index[-1].date().isoformat(),
            "fuente_datos": fuente, "aviso_datos": error}


@app.get("/modelo", summary="Ficha del modelo")
def modelo():
    return {k: v for k, v in META.items() if k != "percentiles_probabilidad"}


@app.get("/prediccion/hoy", summary="Probabilidad con el último cierre disponible")
def prediccion_hoy():
    tabla, fuente, error = tabla_predicciones()
    return armar_respuesta("hoy", tabla.index[-1], tabla.iloc[-1], fuente, error)


@app.get("/prediccion/{fecha}", summary="Probabilidad para una fecha histórica")
def prediccion_fecha(fecha: date):
    tabla, fuente, error = tabla_predicciones()
    ts = pd.Timestamp(fecha)
    if ts < tabla.index[0] or ts > tabla.index[-1]:
        raise HTTPException(404, f"Fecha fuera de rango: hay datos entre {tabla.index[0].date()} y {tabla.index[-1].date()}")
    pos = tabla.index.searchsorted(ts, side="right") - 1      # última rueda <= fecha (sin mirar al futuro)
    return armar_respuesta(fecha.isoformat(), tabla.index[pos], tabla.iloc[pos], fuente, error)


@app.get("/historial", summary="Serie de probabilidades entre dos fechas (máx. 3 años)")
def historial(desde: date = Query(..., description="AAAA-MM-DD"), hasta: Optional[date] = Query(None, description="AAAA-MM-DD")):
    tabla, fuente, _ = tabla_predicciones()
    hasta = hasta or tabla.index[-1].date()
    if desde > hasta:
        raise HTTPException(422, "'desde' debe ser anterior a 'hasta'")
    if (hasta - desde).days > 3 * 366:
        raise HTTPException(422, "El rango máximo es de 3 años")
    t = tabla.loc[pd.Timestamp(desde):pd.Timestamp(hasta)]
    return {
        "desde": desde.isoformat(), "hasta": hasta.isoformat(), "n": len(t), "fuente_datos": fuente,
        "prevalencia_historica": round(PREVALENCIA, 4),
        "serie": [{"fecha": i.date().isoformat(), "probabilidad": round(float(r.probabilidad), 4),
                   "hubo_movimiento": None if pd.isna(r.retorno_5d_realizado) else bool(abs(r.retorno_5d_realizado) > UMBRAL)}
                  for i, r in t.iterrows()],
    }


class Features(BaseModel):
    """Las 33 features ya calculadas (para quien quiera probar escenarios hipotéticos)."""
    valores: dict[str, float] = Field(..., description="Diccionario feature -> valor, con las 33 features del modelo")


@app.post("/prediccion/escenario", summary="Probabilidad para un vector de features hipotético")
def prediccion_escenario(entrada: Features):
    faltan = [f for f in FEATURES if f not in entrada.valores]
    sobran = [f for f in entrada.valores if f not in FEATURES]
    if faltan or sobran:
        raise HTTPException(422, {"faltan": faltan, "desconocidas": sobran})
    X = pd.DataFrame([[entrada.valores[f] for f in FEATURES]], columns=FEATURES)
    p = float(MODELO.predict_proba(X)[:, 1][0])
    pct = float(np.interp(p, PERCENTILES, np.arange(101)))
    return {"probabilidad_movimiento": round(p, 4), "percentil_historico": round(pct, 1), "nivel": nivel(pct), "aviso": AVISO}
