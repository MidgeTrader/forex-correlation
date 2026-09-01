"""Motor de detección de régimen de mercado (GMM) para el dashboard.

Detecta el régimen estadístico diario de un activo mediante Gaussian
Mixture Model sobre features de retorno, volatilidad, drawdown y rango.
Basado en el script de referencia ``regimen_sp500.py``, adaptado a consumo
programático (sin CLI): recibe series OHLCV por símbolo y devuelve un dict
listo para JSON.

Regla de honestidad del volumen: los pares FX de Yahoo traen Volume=0 en
todas las filas; ``volume_relative_20`` solo se computa si la serie de
volumen no es constante (``nunique()>1``). Sin volumen válido la feature se
omite y el clustering funciona igual con el resto de features.

El ajuste es in-sample (el GMM se entrena con toda la historia disponible):
sirve para describir el régimen actual y el histórico, no para operar con
datos que el modelo no habría visto (para eso haría falta walk-forward).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

# K mínimo/máximo evaluados por BIC. El script de referencia usa 3-7, pero
# con ~460 observaciones (2 años diarios) un GMM full-covariance con K>5
# sobreajusta y el BIC en datos financieros tiende a decrecer de forma
# monótona con K (acabaría eligiendo siempre el tope). K_MAX=5 deja cards
# legibles con 3-5 regímenes.
K_MIN = 3
K_MAX = 5

# Semilla fija: el clustering es reproducible entre regeneraciones.
RANDOM_STATE = 42

# Reinicios del EM. El script de referencia usa 10; con 5 se converge igual
# en ~460 obs y se reduce el coste total del dashboard (~40 activos).
N_INIT = 5

# Observaciones mínimas para poder detectar régimen (ventanas de 60 días +
# margen). Por debajo devolvemos None (el selector muestra "Sin datos").
MIN_OBS = 120

# Colores semánticos por régimen (paleta del dashboard). El matching se hace
# sobre palabras clave del nombre: stress/bearish=rojo, bullish=verde,
# volatile=naranja, neutral=gris; el resto cae en _PALETA por orden.
_COLORES = {
    "stress": "#f85149",
    "bearish": "#f85149",
    "bullish": "#3fb950",
    "volatile": "#ff9f43",
    "neutral": "#8b949e",
}
_PALETA = ["#58a6ff", "#a371f7", "#ffd33d", "#39ff14", "#d2a8ff"]


def _color_regimen(nombre: str, index: int) -> str:
    """Color semántico de un régimen por palabra clave (fallback por índice)."""
    clave = nombre.lower()
    for palabra, color in _COLORES.items():
        if palabra in clave:
            return color
    return _PALETA[index % len(_PALETA)]


def construir_features(
    close: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    volume: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Construye las features del clustering a partir de series OHLCV.

    Args:
        close: Serie de precios de cierre (indexada por fecha).
        high: Serie de máximos (opcional; habilita ``range_20``).
        low: Serie de mínimos (opcional; habilita ``range_20``).
        volume: Serie de volumen (opcional; solo si no es constante).

    Returns:
        DataFrame con features (returns, vol, drawdown, range, volumen) y la
        fila ``Close`` como auxiliar. Sin NaNs: se descartan las filas con
        valores no finitos tras las ventanas de 60 periodos.
    """
    out = pd.DataFrame(index=close.index)

    returns = close.pct_change()

    out["return_5"] = close.pct_change(5)
    out["return_20"] = close.pct_change(20)
    out["return_60"] = close.pct_change(60)

    out["vol_20"] = returns.rolling(20).std() * np.sqrt(252)
    out["vol_60"] = returns.rolling(60).std() * np.sqrt(252)

    # Máximo hasta t-1 para evitar look-ahead en esta feature.
    max_60 = close.shift(1).rolling(60).max()
    out["drawdown_60"] = (close / max_60) - 1

    if high is not None and low is not None:
        daily_range = (high - low) / close
        out["range_20"] = daily_range.rolling(20).mean()

    # Volumen solo si la serie no es constante/cero (los FX de Yahoo vienen 0).
    if volume is not None and volume.nunique() > 1:
        out["volume_relative_20"] = volume / volume.rolling(20).mean()

    out["Close"] = close

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna()

    return out


def _seleccionar_k_bic(X: np.ndarray) -> int:
    """Elige K minimizando el BIC entre K_MIN y K_MAX (menor es mejor)."""
    mejor_k, mejor_bic = K_MIN, float("inf")
    for k in range(K_MIN, K_MAX + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=N_INIT,
            random_state=RANDOM_STATE,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < mejor_bic:
            mejor_bic, mejor_k = bic, k
    return mejor_k


def _caracterizar_clusters(
    features: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Medias por cluster (retorno, vol, drawdown) para nombrarlos."""
    data = features.copy()
    data["cluster"] = labels
    resumen = data.groupby("cluster").mean(numeric_only=True)
    resumen["n_dias"] = data["cluster"].value_counts().sort_index()
    return resumen


def _nombrar_regimenes(resumen: pd.DataFrame) -> dict[int, str]:
    """Asigna nombres económicos a los clusters por cuantiles (no influye el fit)."""
    nombres: dict[int, str] = {}

    ret_q = resumen["return_20"].quantile([0.25, 0.50, 0.75])
    vol_q = resumen["vol_20"].quantile([0.50, 0.75])

    for cluster, row in resumen.iterrows():
        ret = row["return_20"]
        vol = row["vol_20"]
        dd = row["drawdown_60"]

        if ret < ret_q.loc[0.25] and vol > vol_q.loc[0.75] and dd < 0:
            nombre = "Stress"
        elif ret < ret_q.loc[0.50] and dd < 0:
            nombre = "Bearish Transition"
        elif ret > ret_q.loc[0.50] and vol > vol_q.loc[0.50]:
            nombre = "Bullish Volatile"
        elif ret > ret_q.loc[0.50] and vol <= vol_q.loc[0.50]:
            nombre = "Bullish Expansion"
        else:
            nombre = "Neutral / Consolidation"

        nombres[int(cluster)] = nombre

    return nombres


def _suavizar_regimenes(
    regimes: list[str],
    smooth_days: int,
) -> list[str]:
    """Filtro de confirmación: el nuevo régimen debe durar N días consecutivos.

    No toca las probabilidades; solo estabiliza la secuencia de regímenes
    para que cambios de un día suelto no pinten el gráfico de manchas.
    """
    if smooth_days <= 1:
        return regimes

    suave = [regimes[0]]
    activo = regimes[0]
    candidato: Optional[str] = None
    cuenta = 0

    for r in regimes[1:]:
        if r == activo:
            candidato, cuenta = None, 0
        else:
            if candidato == r:
                cuenta += 1
            else:
                candidato, cuenta = r, 1
            if cuenta >= smooth_days:
                activo = r
                candidato, cuenta = None, 0
        suave.append(activo)

    return suave


def _transiciones_semanales(
    regimes: list[str],
    fechas: pd.Index,
) -> dict[str, dict[str, float]]:
    """Matriz de transición entre semanas (cadena de Markov semanal).

    Cada semana se representa por el régimen de su último día negociado (el
    mismo criterio que el "régimen actual" al cierre) y se cuentan los cambios
    de régimen entre semanas consecutivas, normalizando por fila. Al agregar
    por semana, el ruido de un día suelto no cuenta como transición: las
    probabilidades reflejan la persistencia real del régimen de cierre a cierre
    de semana (usualmente alta: el régimen tarda semanas en cambiar).

    Args:
        regimes: Secuencia diaria de regímenes (ya suavizados).
        fechas: Índice de fechas alineado con `regimes` (para agrupar por
            semana ISO).

    Returns:
        Dict ``{origen: {destino: probabilidad}}``, destinos ordenados de mayor
        a menor probabilidad. Orígenes sin historia previa no aparecen.
    """
    serie = pd.Series(regimes, index=fechas)
    semanal = serie.groupby(serie.index.to_period("W")).agg("last").tolist()

    conteos: dict[str, dict[str, int]] = {}
    for prev, cur in zip(semanal[:-1], semanal[1:]):
        conteos.setdefault(prev, {})
        conteos[prev][cur] = conteos[prev].get(cur, 0) + 1

    resultado: dict[str, dict[str, float]] = {}
    for origen, destinos in conteos.items():
        total = sum(destinos.values())
        resultado[origen] = {
            destino: cuenta / total
            for destino, cuenta in sorted(destinos.items(), key=lambda x: -x[1])
        }
    return resultado


def detectar_regimen(
    close: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    volume: Optional[pd.Series] = None,
    smooth_days: int = 3,
) -> Optional[dict]:
    """Detecta el régimen diario de un activo y devuelve datos para el chart.

    Args:
        close: Serie de cierres (indexada por fecha).
        high: Serie de máximos (opcional).
        low: Serie de mínimos (opcional).
        volume: Serie de volumen (opcional; se omite si constante).
        smooth_days: Días consecutivos para confirmar un cambio de régimen.

    Returns:
        Dict listo para ``json.dumps`` con ``current`` (régimen actual y
        métricas: fiabilidad fuera de muestra (acuerdo OOS), K, duración de la
        racha, régimen anterior, retorno/volatilidad/drawdown a 20/20/60 días,
        color, transiciones de Markov semana a semana y duración mediana
        histórica). ``None`` si no hay historia suficiente o falla el fit.
    """
    if close is None or len(close.dropna()) < MIN_OBS:
        return None

    features = construir_features(
        close,
        high=high,
        low=low,
        volume=volume,
    )

    if len(features) < MIN_OBS:
        return None

    feature_columns = [c for c in features.columns if c != "Close"]

    try:
        scaler = RobustScaler()
        X = scaler.fit_transform(features[feature_columns])

        k = _seleccionar_k_bic(X)

        modelo = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=N_INIT,
            random_state=RANDOM_STATE,
        )
        modelo.fit(X)

        labels = modelo.predict(X)

        resumen = _caracterizar_clusters(features, labels)
        nombres = _nombrar_regimenes(resumen)

        regimes = [nombres[int(c)] for c in labels]
        regimes = _suavizar_regimenes(regimes, smooth_days)

        # Fiabilidad fuera de muestra: se entrena un GMM SOLO con el 80% inicial
        # y se predice el 20% final (que nunca vio). Se empareja cada cluster OOS
        # con el del modelo completo por centroide más cercano (evita la
        # permutación de etiquetas típica del GMM) y se mide el acuerdo entre
        # ambos para esos días de validación. Ese acuerdo es la confianza
        # honesta: cuánto se reproduce el régimen sin haber visto el presente,
        # en vez del máximo posterior in-sample (que siempre sale optimista).
        fiabilidad: Optional[float] = None
        if len(features) >= 250:  # ~200 de entrenamiento + ~50 de validación
            corte = int(len(features) * 0.8)
            train, val = features.iloc[:corte], features.iloc[corte:]
            scaler_oos = RobustScaler()
            X_train = scaler_oos.fit_transform(train[feature_columns])
            X_val = scaler_oos.transform(val[feature_columns])

            modelo_oos = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=N_INIT,
                random_state=RANDOM_STATE,
            )
            modelo_oos.fit(X_train)

            # Cada modelo etiqueta los mismos días con SU propio escalado: el
            # completo con el scaler completo, el OOS con el suyo.
            etiquetas_A_train = modelo.predict(scaler.transform(train[feature_columns]))
            etiquetas_B_train = modelo_oos.predict(X_train)

            # Permutación óptima de etiquetas B→A maximizando el acuerdo en el
            # entrenamiento (algoritmo húngaro): así el acuerdo en validación no
            # mezcla inestabilidad con la mera permutación de etiquetas del GMM.
            confusion = np.zeros((k, k), dtype=int)
            for a, b in zip(etiquetas_A_train, etiquetas_B_train):
                confusion[a, b] += 1
            fila, col = linear_sum_assignment(-confusion)
            mapeo: dict[int, int] = dict(zip(col, fila))

            etiquetas_A_val = modelo.predict(scaler.transform(val[feature_columns]))
            etiquetas_B_val = modelo_oos.predict(X_val)
            acuerdo = [mapeo[b] == int(a) for a, b in zip(etiquetas_A_val, etiquetas_B_val)]
            fiabilidad = float(np.mean(acuerdo)) if acuerdo else None

        ultimo = len(regimes) - 1

        # Duración de la racha actual (días consecutivos del mismo régimen) y
        # régimen anterior (el último distinto antes de que empezara la racha).
        actual = regimes[ultimo]
        duracion = 1
        i = ultimo - 1
        while i >= 0 and regimes[i] == actual:
            duracion += 1
            i -= 1
        prev_regimen = regimes[i] if i >= 0 else actual

        # Transiciones de Markov semana a semana desde el régimen actual
        # (incluye quedarse: el régimen suele persistir de una semana a otra).
        transiciones = _transiciones_semanales(regimes, features.index).get(actual, {})
        trans_list = [
            {"regime": destino, "prob": prob}
            for destino, prob in transiciones.items()
        ]

        # Duración MEDIANA de las rachas COMPLETAS de este régimen (excluye la
        # actual, que sigue en curso): la mediana es más robusta que la media a
        # una sola racha larga anómala.
        rachas: list[tuple[str, int]] = []
        for r in regimes:
            if rachas and rachas[-1][0] == r:
                rachas[-1] = (r, rachas[-1][1] + 1)
            else:
                rachas.append((r, 1))
        duraciones_completas = [
            length for nombre, length in rachas[:-1] if nombre == actual
        ]
        duracion_mediana = (
            float(np.median(duraciones_completas))
            if duraciones_completas
            else None
        )

        return {
            "current": {
                "regime": actual,
                "fiabilidad": fiabilidad,
                "k": k,
                "duration": duracion,
                "prevRegime": prev_regimen,
                "ret20": float(features["return_20"].iloc[-1]) * 100,
                "vol20": float(features["vol_20"].iloc[-1]) * 100,
                "dd60": float(features["drawdown_60"].iloc[-1]) * 100,
                "color": _color_regimen(actual, 0),
                "transitions": trans_list,
                "medianDuration": duracion_mediana,
            },
        }
    except Exception:
        # Nunca romper el dashboard por un activo problemático: se muestra
        # "Sin datos" en su card y el resto del análisis sigue intacto.
        return None
