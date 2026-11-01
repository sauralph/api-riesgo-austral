"""Descarga de datos de mercado para la API (misma lógica que el notebook).

- S&P 500 (cierre, máximo, mínimo): Yahoo Finance (^GSPC), desde 1990.
- VIX y pendiente 10Y-2Y: FRED (CSV público, sin API key).
Si la descarga en vivo falla, se usa el CSV congelado del repositorio y se avisa en la respuesta.
"""
import io
import logging
import threading
import time
from pathlib import Path

import pandas as pd
import requests

from .features import HISTORIA_DESDE

log = logging.getLogger("datos")
RAIZ = Path(__file__).resolve().parent.parent
CSV_SP500 = RAIZ / "datos" / "clase6_sp500_ohlc_yahoo.csv"
CSV_FRED = RAIZ / "datos" / "clase6_fred_series.csv"
SERIES_FRED = {"VIXCLS": "vix", "T10Y2Y": "spread_10y2y"}
TIMEOUT = 30


def _descargar_sp500() -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download("^GSPC", start=HISTORIA_DESDE, auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance devolvió una tabla vacía")
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return raw[["high", "low", "close"]]


def _descargar_fred() -> pd.DataFrame:
    partes = []
    for fred_id, nombre in SERIES_FRED.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}&cosd={HISTORIA_DESDE}"
        resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "api-riesgo-austral/1.0"})
        resp.raise_for_status()
        serie = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True, na_values=["."])
        serie.columns = [nombre]
        partes.append(serie)
    return pd.concat(partes, axis=1, sort=True)


def _armar(spx: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    """Mismo armado que el notebook: calendario del S&P 500 y ffill (sólo hacia adelante) de FRED."""
    data = spx[["high", "low", "close"]].rename(columns={"close": "sp500"}).dropna(subset=["sp500"]).sort_index()
    data = data.loc[~data.index.duplicated(keep="last")].loc[HISTORIA_DESDE:]
    data = data.join(fred[list(SERIES_FRED.values())], how="left").apply(pd.to_numeric, errors="coerce")
    data[list(SERIES_FRED.values())] = data[list(SERIES_FRED.values())].ffill()
    # El modelo se entrenó con el CSV congelado (4 decimales). Redondeamos igual: sin esto,
    # diferencias de 1e-5 cruzan umbrales de los árboles y cambian la probabilidad (training-serving skew).
    return data.round(4)


def cargar_congelado() -> pd.DataFrame:
    spx = pd.read_csv(CSV_SP500, index_col="date", parse_dates=True)
    fred = pd.read_csv(CSV_FRED, index_col="date", parse_dates=True)
    return _armar(spx, fred)


def cargar_en_vivo() -> pd.DataFrame:
    return _armar(_descargar_sp500(), _descargar_fred())


class CacheDatos:
    """Guarda los datos en memoria y los refresca cada `ttl` segundos."""

    def __init__(self, ttl: int = 6 * 3600):
        self.ttl = ttl
        self._lock = threading.Lock()
        self._datos = None
        self._fuente = None
        self._hora = 0.0
        self._error = None

    def obtener(self):
        with self._lock:
            if self._datos is None or time.time() - self._hora > self.ttl:
                try:
                    self._datos, self._fuente, self._error = cargar_en_vivo(), "en vivo (Yahoo Finance + FRED)", None
                except Exception as exc:                      # red caída, Yahoo bloqueado, etc.
                    log.warning("Descarga en vivo falló: %s", exc)
                    self._error = f"{type(exc).__name__}: {exc}"
                    if self._datos is None:
                        self._datos, self._fuente = cargar_congelado(), "CSV congelado (sin datos recientes)"
                self._hora = time.time()
            return self._datos, self._fuente, self._error
