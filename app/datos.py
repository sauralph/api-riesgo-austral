"""Descarga de datos de mercado para la API (misma lógica que el notebook).

- S&P 500 (cierre, máximo, mínimo): Yahoo Finance (^GSPC), desde 1990.
- VIX y pendiente 10Y-2Y: FRED (CSV público, sin API key).
Plan B, si una fuente falla:
- VIX: Yahoo Finance (^VIX, mismo cierre de CBOE).
- Pendiente 10Y-2Y: último valor del CSV congelado, arrastrado hacia adelante (se avisa).
- Todo: CSV congelado del repositorio (se avisa).
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
TIMEOUT = 45
INTENTOS = 3


def _yahoo(ticker: str, columnas) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(ticker, start=HISTORIA_DESDE, auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo Finance devolvió una tabla vacía para {ticker}")
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return raw[columnas]


def _fred(fred_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}&cosd={HISTORIA_DESDE}"
    ultimo_error = None
    for intento in range(INTENTOS):
        try:
            resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0 (api-riesgo-austral)"})
            resp.raise_for_status()
            serie = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True, na_values=["."]).iloc[:, 0]
            return pd.to_numeric(serie, errors="coerce")
        except Exception as exc:
            ultimo_error = exc
            time.sleep(2 * (intento + 1))
    raise RuntimeError(f"FRED {fred_id}: {type(ultimo_error).__name__}")


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


def cargar_en_vivo():
    """Devuelve (datos, avisos). Lanza excepción sólo si no se pudo obtener el S&P 500."""
    avisos = []
    spx = _yahoo("^GSPC", ["high", "low", "close"])
    congelado = pd.read_csv(CSV_FRED, index_col="date", parse_dates=True)
    columnas = {}
    try:
        columnas["vix"] = _fred("VIXCLS")
    except Exception as exc:
        # Historia: la de FRED congelada (con la que se entrenó). Sólo las fechas nuevas vienen de Yahoo:
        # el ^VIX histórico de Yahoo difiere del de FRED en algunos días (otro caso de skew).
        historia = congelado["vix"].dropna()
        nuevo = _yahoo("^VIX", ["close"])["close"].loc[lambda s: s.index > historia.index[-1]]
        columnas["vix"] = pd.concat([historia, nuevo])
        avisos.append(f"VIX posterior al {historia.index[-1].date()} desde Yahoo (^VIX) porque falló {exc}")
    try:
        columnas["spread_10y2y"] = _fred("T10Y2Y")
    except Exception as exc:
        columnas["spread_10y2y"] = congelado["spread_10y2y"]
        avisos.append(f"Pendiente 10Y-2Y congelada al {congelado['spread_10y2y'].last_valid_index().date()} porque falló {exc}")
    return _armar(spx, pd.DataFrame(columnas)), avisos


class CacheDatos:
    """Guarda los datos en memoria. Se refresca cada `ttl` segundos, o cada `ttl_error` si hubo problemas."""

    def __init__(self, ttl: int = 6 * 3600, ttl_error: int = 15 * 60):
        self.ttl, self.ttl_error = ttl, ttl_error
        self._lock = threading.Lock()
        self._datos, self._fuente, self._error = None, None, None
        self._hora = 0.0

    def obtener(self):
        with self._lock:
            vigencia = self.ttl if self._error is None else self.ttl_error
            if self._datos is None or time.time() - self._hora > vigencia:
                try:
                    self._datos, avisos = cargar_en_vivo()
                    self._fuente = "en vivo (Yahoo Finance + FRED)"
                    self._error = "; ".join(avisos) or None
                except Exception as exc:                      # sin red, Yahoo bloqueado, etc.
                    log.warning("Descarga en vivo falló: %s", exc)
                    self._error = f"{type(exc).__name__}: {exc}"
                    if self._datos is None:
                        self._datos, self._fuente = cargar_congelado(), "CSV congelado (sin datos recientes)"
                self._hora = time.time()
            return self._datos, self._fuente, self._error
