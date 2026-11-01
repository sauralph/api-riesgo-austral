"""Tests de la API sin red: se fuerza el uso del CSV congelado.

Correr con:  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import datos, main
from app.features import FEATURES, construir_features


@pytest.fixture(autouse=True)
def sin_red(monkeypatch):
    def falla():
        raise ConnectionError("sin red en los tests")
    monkeypatch.setattr(datos, "cargar_en_vivo", falla)
    main.cache._datos = None
    main._predicciones.update(clave=None, tabla=None)


cliente = TestClient(main.app)


def test_salud_usa_respaldo():
    r = cliente.get("/salud").json()
    assert r["estado"] == "ok"
    assert "congelado" in r["fuente_datos"]


def test_prediccion_historica_conocida():
    r = cliente.get("/prediccion/2020-03-09").json()
    assert 0 <= r["probabilidad_movimiento"] <= 1
    assert r["realizado"]["hubo_movimiento"] is True
    assert r["nivel"] in {"bajo", "normal", "alto", "muy alto"}


def test_fecha_no_habil_usa_rueda_anterior():
    r = cliente.get("/prediccion/2025-04-05").json()    # sábado
    assert r["fecha_datos"] == "2025-04-04"


def test_errores_de_entrada():
    assert cliente.get("/prediccion/2035-01-01").status_code == 404
    assert cliente.get("/prediccion/hola").status_code == 422
    assert cliente.get("/historial", params={"desde": "2000-01-01", "hasta": "2010-01-01"}).status_code == 422


def test_escenario_coincide_con_historico():
    X = construir_features(datos.cargar_congelado())
    fila = X.loc["2020-03-09", FEATURES]
    r = cliente.post("/prediccion/escenario", json={"valores": fila.to_dict()}).json()
    esperado = cliente.get("/prediccion/2020-03-09").json()["probabilidad_movimiento"]
    assert r["probabilidad_movimiento"] == pytest.approx(esperado, abs=1e-4)


def test_features_sin_futuro():
    """Recalcular con datos cortados en t no cambia el valor de t (auditoría de la Clase 6)."""
    d = datos.cargar_congelado()
    completo = construir_features(d)
    for fecha in ["2008-10-10", "2020-03-16", "2024-08-05"]:
        cortado = construir_features(d.loc[:fecha])
        np.testing.assert_allclose(cortado.iloc[-1].to_numpy(), completo.loc[fecha].to_numpy(), rtol=1e-10)
