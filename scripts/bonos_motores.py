"""Motor de la curva del Tesoro de EE.UU. desde FRED.

Las series ``DGS*`` de FRED son los *Daily Treasury Par Yield Curve Rates* que
publica el Departamento del Tesoro de EE.UU. (FRED solo los sirve): rendimiento
del bono del Tesoro de cada plazo, diario, en % (4.34 = 4.34 %).

Se sirve de la caché en disco (``data/bonos_curva.csv``) con auto-refresh, el
mismo patrón que ``MacroDataYF``: regenerar el dashboard no re-descarga salvo
que la caché esté desactualizada. Sin key o sin red → ``None`` (fallo seguro):
la card muestra N/A, nunca un valor inventado.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from fredapi import Fred

from data_fred import _load_api_key
from macro_catalogo import BONOS_TRAMOS

# Caché en disco: `data/` en la raíz del proyecto (un nivel por encima de scripts/).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Años de historia que se conservan (la card compara hoy vs hace 1 año y usa
# cambios diario/mensual; 2 años sobran y sobra). El usuario pidió 1-2 años.
ANIOS_HISTORIA = 2

# Umbral de antigüedad del último dato para considerar la caché al día:
# si la última sesión es anterior al penúltimo día hábil, se re-descarga.
_ULTIMO_OK = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)


def _cache_ok() -> Optional[pd.DataFrame]:
    """Carga ``data/bonos_curva.csv`` si existe y está al día; si no, None."""
    ruta = DATA_DIR / "bonos_curva.csv"
    if not ruta.exists():
        return None
    try:
        df = pd.read_csv(ruta, index_col=0, parse_dates=True)
        if df.empty:
            return None
        if df.index.max().normalize() < _ULTIMO_OK:
            return None  # caché desactualizada → re-descargar
        return df
    except Exception:
        return None  # caché corrupta → re-descargar


def _descargar_curva() -> Optional[pd.DataFrame]:
    """Descarga las 5 series DGS* desde FRED (solo últimos 2 años)."""
    key = _load_api_key()
    if key is None:
        return None

    try:
        fred = Fred(api_key=key)
        inicio = date.today() - timedelta(days=365 * ANIOS_HISTORIA)
        series = {}
        for tramo, spec in BONOS_TRAMOS.items():
            s = fred.get_series(spec["series"], observation_start=inicio).dropna()
            if s.empty:
                return None  # falta un tramo → toda la curva se considera N/A
            series[tramo] = s
        return pd.DataFrame(series).sort_index()
    except Exception:
        return None  # sin red / API caída / key inválida → fallo seguro


def analizar_curva(refresh: bool = False) -> Optional[pd.DataFrame]:
    """Curva del Tesoro (5 tramos) indexada por fecha, o None si no hay datos.

    Args:
        refresh: True fuerza la re-descarga (ignora la caché en disco).

    Returns:
        DataFrame con una columna por tramo (``TB3M``, ``TB2Y``, ``TB5Y``,
        ``TB10Y``, ``TB30Y``) en %, indexado por fecha. None si no hay key,
        falta algún tramo o FRED no responde (fallo seguro → card N/A).
    """
    if not refresh:
        cached = _cache_ok()
        if cached is not None:
            return cached

    df = _descargar_curva()
    if df is None:
        return None

    try:
        ruta = DATA_DIR / "bonos_curva.csv"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta)
    except OSError:
        pass  # sin permiso de escritura: se sigue con datos en memoria

    return df
