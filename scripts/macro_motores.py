"""Orquesta las métricas de panel autocontenidas sobre los activos macro.

Calcula las métricas de las tarjetas con `macro_metrics` sobre los datos de
`MacroDataYF` (yfinance, con caché en disco y auto-refresh). Sin dependencias
privadas: el repo se clona y regenera en cualquier máquina.
"""

from __future__ import annotations

from typing import Optional

from macro_data import MacroDataYF
from macro_metrics import metricas


def analizar_activo(
    key: str,
    period: str = "5y",
    refresh: bool = False,
) -> dict:
    """Analiza un activo macro con las métricas de panel autocontenidas.

    Args:
        key: Clave del activo del catálogo (p.ej. ``GOLD``).
        period: Periodo de histórico (``5y`` por defecto).
        refresh: True fuerza la re-descarga (ignora la caché en disco).

    Returns:
        Dict con ``data`` (el ``MacroDataYF``, con ``error`` si falló) y
        ``results`` (dict de métricas) cuando hay datos.
    """
    data = MacroDataYF(key, period=period, refresh=refresh)

    if not data.fetch():
        return {"data": data, "results": None}

    closes = data.price_data["Close"].dropna()
    benchmark = data.spy_data["Close"].dropna() if data.spy_data is not None else None

    results = metricas(closes, benchmark)

    return {"data": data, "results": results}
