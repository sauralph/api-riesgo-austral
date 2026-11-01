# API de riesgo de mercado · ML para Riesgo (Universidad Austral)

Servicio web que devuelve la probabilidad de que el **S&P 500 se mueva más de 2,5%** (en cualquier dirección) en las próximas **5 ruedas**. Es el modelo de la Clase 7 del Módulo 4 de la Diplomatura en Análisis Cuantitativo y Machine Learning aplicado al Riesgo Financiero.

> Uso educativo. No es una recomendación de inversión.

## Endpoints

| Método | Ruta | Devuelve |
|---|---|---|
| GET | `/prediccion/hoy` | Probabilidad con el último cierre disponible |
| GET | `/prediccion/{AAAA-MM-DD}` | Probabilidad para una fecha histórica y lo que efectivamente pasó |
| GET | `/historial?desde=AAAA-MM-DD&hasta=AAAA-MM-DD` | Serie de probabilidades (máx. 3 años) |
| POST | `/prediccion/escenario` | Probabilidad para un vector de 33 features hipotético |
| GET | `/modelo` | Ficha del modelo: hiperparámetros, métricas, features, versiones |
| GET | `/salud` | Estado del servicio y fecha del último dato |
| GET | `/docs` | Documentación interactiva |

Ejemplo:

```bash
curl https://api-riesgo-austral-kzmr7.ondigitalocean.app/prediccion/2020-03-09
```

```python
import requests
requests.get("https://api-riesgo-austral-kzmr7.ondigitalocean.app/prediccion/hoy").json()
```

## Cómo funciona

1. **Datos:** S&P 500 (cierre, máximo y mínimo) desde 1990 de Yahoo Finance; VIX y pendiente 10Y–2Y de FRED. Se guardan en memoria 6 horas. Si FRED no responde, el VIX nuevo se toma de Yahoo (`^VIX`) y la pendiente queda en su último valor congelado; si falla todo, se usa el CSV congelado de `datos/`. En cualquier plan B, la respuesta lo avisa en `aviso_datos` y se reintenta a los 15 minutos.
2. **Features:** `app/features.py`, el **mismo archivo** que usa el notebook de entrenamiento (33 features de la Clase 6).
3. **Modelo:** `modelo/modelo_movimiento.json` (XGBoost, formato nativo) y `modelo/meta.json` (ficha del modelo). Los genera la sección 8 del notebook `07_hiperparametros_y_produccion_colab.ipynb`.

## Correr localmente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# abrir http://127.0.0.1:8000/docs
```

Tests (no necesitan red):

```bash
pip install pytest httpx
python -m pytest -q
```

## Desplegar en DigitalOcean App Platform

Con [`doctl`](https://docs.digitalocean.com/reference/doctl/):

```bash
doctl apps create --spec .do/app.yaml
```

O desde el panel: **Create → App Platform → GitHub**, elegir este repositorio y la rama `main`. App Platform detecta Python, instala `requirements.txt` y corre el `run_command` de `.do/app.yaml`. La instancia más chica cuesta unos USD 5 por mes.

## Actualizar el modelo

1. Reentrenar con el notebook (sección 8) → reemplazar los archivos de `modelo/`.
2. Subir `VERSION_MODELO`, hacer commit y push.
3. Volver a desplegar (si la app está conectada con GitHub, el push la redespliega sola).

## Una lección de training–serving skew

El modelo se entrenó con precios redondeados a 4 decimales (los del CSV congelado). Sin redondear igual en producción, diferencias de 0,00005 cruzaban umbrales de los árboles y cambiaban la probabilidad hasta 11 puntos en 48 días de la historia. Por eso `app/datos.py` redondea los datos en vivo, y `tests/test_api.py` verifica que las features no usen información futura.
