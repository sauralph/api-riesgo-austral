"""Features del modelo de riesgo: UNA sola implementación para entrenar (Colab) y servir (API).

Es el mismo código de la Clase 6 (familias BASE, VOLATILIDAD, IMPLÍCITA, RÉGIMEN, INTRADÍA).

Entrada: DataFrame diario (ruedas del S&P 500) indexado por fecha, con columnas
    sp500         cierre del S&P 500
    high, low     máximo y mínimo del día del S&P 500
    vix           VIX al cierre (FRED VIXCLS)
    spread_10y2y  pendiente 10Y-2Y (FRED T10Y2Y)
Todas las features miran SÓLO hacia atrás. El GARCH usa una varianza de largo plazo
expansiva: para reproducir exactamente el entrenamiento, la historia debe empezar en HISTORIA_DESDE.
"""
import numpy as np
import pandas as pd

HISTORIA_DESDE = "1990-01-01"
HORIZONTE = 5            # ruedas
UMBRAL = 0.025           # 2,5%
RUEDAS_ANUALES = 252
EWMA_ALPHA = 0.06
EWMA_ALPHA_RAPIDA = 0.20
VENTANA_REGIMEN = 252
MIN_PERIODOS_REGIMEN = 126
GARCH_ALPHA, GARCH_BETA = 0.08, 0.90

FAMILIAS = {
    "BASE": ["ret_5d", "ret_20d", "vol_20d", "ewma_vol", "drawdown_1y", "vix", "vix_change_5d", "spread_10y2y"],
    "VOLATILIDAD": ["vol_ratio_5_60", "vol_ratio_20_252", "ewma_vol_rapida", "garch_vol",
                    "semivol_20d", "peor_dia_20d", "skew_60d"],
    "IMPLÍCITA": ["vrp", "vix_rv_ratio", "umbral_sigmas_vix", "umbral_sigmas_ewma"],
    "RÉGIMEN": ["vix_z_252", "vix_rank_252", "vix_vs_ma20", "dist_ma50", "dist_ma200", "drawdown_60d",
                "spread_z_252", "spread_chg_60d", "ret_5d_en_sigmas", "ret_20d_en_sigmas"],
    "INTRADÍA": ["parkinson_vol_10d", "parkinson_vs_cierre", "rango_hoy_sigmas", "rango_max_5d_sigmas"],
}
FEATURES = [f for fam in FAMILIAS.values() for f in fam]


def zscore_rodante(s, ventana=VENTANA_REGIMEN, min_periodos=MIN_PERIODOS_REGIMEN):
    media = s.rolling(ventana, min_periods=min_periodos).mean()
    desvio = s.rolling(ventana, min_periods=min_periodos).std()
    return (s - media) / desvio


def rank_rodante(s, ventana=VENTANA_REGIMEN, min_periodos=MIN_PERIODOS_REGIMEN):
    return s.rolling(ventana, min_periods=min_periodos).rank(pct=True)


def garch_vol_fija(r, alpha=GARCH_ALPHA, beta=GARCH_BETA, anual=RUEDAS_ANUALES):
    """GARCH(1,1) con parámetros fijados a priori y varianza de largo plazo expansiva (sin leakage)."""
    r = r.fillna(0.0).to_numpy()
    var_larga = pd.Series(r ** 2).expanding(min_periods=20).mean().shift(1).bfill().to_numpy()
    s2 = np.empty_like(r)
    s2[0] = var_larga[0]
    for t in range(1, len(r)):
        s2[t] = (1 - alpha - beta) * var_larga[t] + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
    s2_hoy = np.append(s2[1:], np.nan)          # varianza que ya incorpora el retorno de t
    # Última rueda (la de "hoy" en producción): mismo paso, con la varianza de largo plazo hasta t.
    s2_hoy[-1] = (1 - alpha - beta) * np.mean(r ** 2) + alpha * r[-1] ** 2 + beta * s2[-1]
    return np.sqrt(s2_hoy * anual)


def construir_features(datos: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame con las columnas FEATURES (NaN en las primeras ruedas)."""
    d = datos[["sp500", "high", "low", "vix", "spread_10y2y"]].astype(float)
    precio, vix, spread = d["sp500"], d["vix"], d["spread_10y2y"]
    r = precio.pct_change(fill_method=None)
    raiz = np.sqrt(RUEDAS_ANUALES)
    f = pd.DataFrame(index=d.index)

    # BASE
    f["ret_5d"] = precio.pct_change(5, fill_method=None)
    f["ret_20d"] = precio.pct_change(20, fill_method=None)
    f["vol_20d"] = r.rolling(20).std() * raiz
    f["ewma_vol"] = r.ewm(alpha=EWMA_ALPHA, adjust=False).std() * raiz
    f["drawdown_1y"] = precio / precio.rolling(252, min_periods=20).max() - 1
    f["vix"] = vix
    f["vix_change_5d"] = vix.pct_change(5, fill_method=None)
    f["spread_10y2y"] = spread

    # VOLATILIDAD
    f["vol_ratio_5_60"] = (r.rolling(5).std() * raiz) / (r.rolling(60).std() * raiz)
    f["vol_ratio_20_252"] = f["vol_20d"] / (r.rolling(252, min_periods=126).std() * raiz)
    f["ewma_vol_rapida"] = r.ewm(alpha=EWMA_ALPHA_RAPIDA, adjust=False).std() * raiz
    f["garch_vol"] = garch_vol_fija(r)
    f["semivol_20d"] = np.sqrt((r.clip(upper=0) ** 2).rolling(20).mean()) * raiz
    f["peor_dia_20d"] = r.rolling(20).min()
    f["skew_60d"] = r.rolling(60).skew()

    # IMPLÍCITA
    f["vrp"] = vix / 100 - f["vol_20d"]
    f["vix_rv_ratio"] = (vix / 100) / f["vol_20d"]
    f["umbral_sigmas_vix"] = UMBRAL / (vix / 100 * np.sqrt(HORIZONTE / RUEDAS_ANUALES))
    f["umbral_sigmas_ewma"] = UMBRAL / (f["ewma_vol"] * np.sqrt(HORIZONTE / RUEDAS_ANUALES))

    # RÉGIMEN
    f["vix_z_252"] = zscore_rodante(vix)
    f["vix_rank_252"] = rank_rodante(vix)
    f["vix_vs_ma20"] = vix / vix.rolling(20).mean() - 1
    f["dist_ma50"] = precio / precio.rolling(50).mean() - 1
    f["dist_ma200"] = precio / precio.rolling(200).mean() - 1
    f["drawdown_60d"] = precio / precio.rolling(60).max() - 1
    f["spread_z_252"] = zscore_rodante(spread)
    f["spread_chg_60d"] = spread.diff(60)
    f["ret_5d_en_sigmas"] = f["ret_5d"] / (f["vol_20d"] / raiz * np.sqrt(5))
    f["ret_20d_en_sigmas"] = f["ret_20d"] / (f["vol_20d"] / raiz * np.sqrt(20))

    # INTRADÍA (sólo máximo y mínimo: la apertura histórica de Yahoo no es confiable)
    rango = np.log(d["high"] / d["low"])
    f["parkinson_vol_10d"] = np.sqrt((rango ** 2).rolling(10).mean() / (4 * np.log(2))) * raiz
    f["parkinson_vs_cierre"] = (np.sqrt((rango ** 2).rolling(20).mean() / (4 * np.log(2))) * raiz) / f["vol_20d"]
    f["rango_hoy_sigmas"] = rango / (f["vol_20d"] / raiz)
    f["rango_max_5d_sigmas"] = rango.rolling(5).max() / (f["vol_20d"] / raiz)

    return f[FEATURES].replace([np.inf, -np.inf], np.nan)


def construir_target_movimiento(precio: pd.Series) -> pd.Series:
    """1 si |P(t+5)/P(t) - 1| > 2,5%. Usa el futuro: SÓLO para entrenar, nunca como feature."""
    ret_futuro = precio.shift(-HORIZONTE) / precio - 1
    return (ret_futuro.abs() > UMBRAL).astype(float).where(ret_futuro.notna())
