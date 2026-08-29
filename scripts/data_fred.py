"""Tipos de interés por divisa desde FRED, para el carry de los pares FX.

Series oficiales de FRED por divisa, API key en ``FRED_API_KEY`` (.env, nunca
hardcodeada) y fallo seguro — si no hay key o red, se degrada a
``TIPOS_RESERVA`` (últimos valores verificados) para que el dashboard siga
funcionando. Nada se inventa.

Para cada divisa se usa la serie al día que representa su tipo de mercado:
    USD → DFF              Federal Funds effective rate (diaria)
    EUR → ECBDFR           ECB Deposit Facility Rate (diaria)
    GBP → IUDSOIA          SONIA, Sterling Overnight Index Average (diaria)
    JPY → IRSTCI01JPM156N  OECD interbancario overnight (mensual)
    CHF → IR3TIB01CHM156N  OECD interbancario 3M (mensual)
    AUD → IRSTCI01AUM156N  OECD overnight = cash rate RBA (mensual)
    CAD → IRSTCI01CAM156N  OECD overnight ≈ BOC target (mensual)
    NZD → IR3TIB01NZM156N  OECD interbancario 3M (mensual)

Las series se verificaron el 2026-08-27 con la key real (valores y fecha del
último dato). Si una serie individual falla al consultarla, se usa su valor de
reserva en vez de dejar el carry en N/A.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from fredapi import Fred

from macro_catalogo import TIPOS_RESERVA

# Series de política / interbancarias de mercado por divisa — el "rate engine".
# Cada serie está al día (verificado 2026-08-27) y representa el tipo que
# define el carry de un par (diferencial base − cotizada).
POLICY_RATES: dict[str, str] = {
    "USD": "DFF",
    "EUR": "ECBDFR",
    "GBP": "IUDSOIA",
    "JPY": "IRSTCI01JPM156N",
    "CHF": "IR3TIB01CHM156N",
    "AUD": "IRSTCI01AUM156N",
    "CAD": "IRSTCI01CAM156N",
    "NZD": "IR3TIB01NZM156N",
}

# Fecha de los valores de reserva (verificados ese día).
FECHA_RESERVA = "2026-08-27"

_PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_api_key() -> Optional[str]:
    """Carga ``FRED_API_KEY`` del entorno o de ``.env`` (sin exponerla).

    Returns:
        La API key, o ``None`` si no está configurada.
    """
    key = os.environ.get("FRED_API_KEY")

    if not key:
        env_file = _PROJECT_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.split("=", 1)[0].strip() == "FRED_API_KEY":
                    key = line.split("=", 1)[1].split("#", 1)[0].strip()

    return key or None


def _ultimo_valor(fred: Fred, series_id: str) -> Optional[float]:
    """Último valor de una serie FRED, en % (3.63 = 3.63%).

    Args:
        fred: Cliente FRED ya inicializado.
        series_id: Identificador de la serie.

    Returns:
        Último valor en puntos porcentuales, o None si la serie está vacía.
    """
    try:
        serie = fred.get_series(series_id).dropna()
        if serie.empty:
            return None
        return float(serie.iloc[-1])
    except Exception:
        return None


@lru_cache(maxsize=1)
def datos_tipos() -> tuple[dict[str, float], str]:
    """Tipos actuales por divisa y su fuente, calculados una vez por generación.

    Consulta FRED para cada divisa de ``POLICY_RATES``. Si una serie falla,
    usa su valor de reserva. Si no hay key o FRED no responde en absoluto,
    devuelve todos los valores de reserva con fuente "reserva".

    Returns:
        Tupla ``(tipos, fuente)`` donde ``tipos`` es ``{divisa: puntos_%}`` y
        ``fuente`` es ``"FRED"`` o ``"reserva"``.
    """
    key = _load_api_key()

    if key is None:
        return dict(TIPOS_RESERVA), "reserva"

    try:
        fred = Fred(api_key=key)
        tipos: dict[str, float] = {}
        for divisa, series_id in POLICY_RATES.items():
            valor = _ultimo_valor(fred, series_id)
            # Fallo seguro por divisa: reserva verificada en vez de N/A.
            tipos[divisa] = valor if valor is not None else TIPOS_RESERVA[divisa]
        return tipos, "FRED"
    except Exception as e:
        # Sin red / API caída / key inválida: el dashboard sigue con los últimos
        # conocidos, pero se avisa para que no parezca que FRED respondió bien.
        print(f"  [aviso] FRED no respondió; usando valores de reserva ({type(e).__name__})")
        return dict(TIPOS_RESERVA), "reserva"


def fecha_tipos() -> str:
    """Fecha a mostrar en la etiqueta del carry de las tarjetas FX.

    Si FRED respondió, es la fecha de generación (los datos pueden ser del día
    o de la última publicación mensual). Si se usó reserva, la fecha de reserva.

    Returns:
        Fecha ISO ``YYYY-MM-DD``.
    """
    _, fuente = datos_tipos()
    if fuente == "FRED":
        return date.today().isoformat()
    return FECHA_RESERVA


# Series de nivel de IPC (índice) por divisa, para el real yield differential.
# Igual patrón que POLICY_RATES: series oficiales de FRED (inflación YoY).
CPI_SERIES: dict[str, str] = {
    "USD": "CPIAUCSL",            # US CPI All Urban (mensual)
    "EUR": "CP0000EZ19M086NEST",  # HICP Euro área (mensual)
    "GBP": "GBRCPIALLMINMEI",     # UK CPI (mensual)
    "JPY": "JPNCPIALLMINMEI",     # Japón CPI (mensual)
    "CHF": "CHECPIALLMINMEI",     # Suiza CPI (mensual)
    "CAD": "CANCPIALLMINMEI",     # Canadá CPI (mensual)
    # AUD y NZD NO están: FRED no publica IPC mensual de OECD para ellos
    # (AUSCPIALLMINMEI/NZLCPIALLMINMEI → HTTP 400, la serie no existe).
    # Los índices australiano/nz son trimestrales y sin serie OECD al día en
    # FRED, así que el Real Yield Differential de esos pares sale N/A (honesto).
}


@lru_cache(maxsize=1)
def datos_inflacion() -> dict[str, float]:
    """Inflación YoY (%) por divisa desde FRED (índices de IPC).

    Calcula la variación interanual del índice de nivel (último dato vs. el
    más reciente ≤ 1 año antes), robusto a frecuencias mensual/trimestral.

    Sin key FRED o si una serie falla, esa divisa se omite — la tarjeta la
    muestra como N/A (honesto: no hay reserva verificada de inflación).

    Returns:
        Dict ``{divisa: inflación_yoy_%}`` (puede estar vacío).
    """
    key = _load_api_key()
    if key is None:
        return {}

    try:
        fred = Fred(api_key=key)
        inflacion: dict[str, float] = {}
        for divisa, series_id in CPI_SERIES.items():
            try:
                serie = fred.get_series(series_id).dropna()
                if serie.empty:
                    continue
                ref = serie.index[-1] - pd.DateOffset(years=1)
                vals_prev = serie[serie.index <= ref]
                if vals_prev.empty:
                    continue
                actual = float(serie.iloc[-1])
                anterior = float(vals_prev.iloc[-1])
                if anterior <= 0:
                    continue
                inflacion[divisa] = (actual / anterior - 1) * 100
            except Exception:
                continue  # serie concreta falla → se omite esa divisa
        return inflacion
    except Exception:
        # Sin red / API caída: no hay inflación → las tarjetas muestran N/A.
        return {}
