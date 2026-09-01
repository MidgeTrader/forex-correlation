"""Catálogo de activos macro (y pares FX mayores) con tickers de Yahoo Finance.

Los activos son contratos macro estándar (futuros de metales, energía y agro,
e índices), expresados con el ticker de Yahoo Finance.
Cada entrada indica su categoría (para la barra del dashboard), su etiqueta
legible y el benchmark de referencia: índice → SPY, agro → DBA (índice de
futuros agrícolas), preciosos → DBP, cobre → DBB (metales base), energía → DBE,
pares FX → DXY (índice del dólar).
"""

from __future__ import annotations

from typing import Final, Optional

# Etiquetas legibles de cada categoría para la barra del dashboard.
# "graficas" NO es una categoría de activos: es la sección del dashboard
# interactivo FX (renombrada), la etiqueta la usa el generador.
CATEGORIAS: Final[dict[str, str]] = {
    "metales": "Metales",
    "energia": "Energía",
    "indices": "Índices",
    "agro": "Agro",
    "crypto": "Crypto",
    "fx": "FX",
    "bonos": "Bonos",
}

# Orden de las categorías en la barra (el generador añade "graficas" al final).
CATEGORIAS_ORDER: Final[list[str]] = ["metales", "energia", "indices", "agro", "crypto", "fx", "bonos"]

# Benchmark por defecto según la categoría: índice → SPY, agro → DBA
# (índice de futuros agrícolas), metales → DBP (preciosos), energía → DBE,
# pares FX → DX-Y.NYB (índice del dólar DXY; los 7 mayores llevan USD).
# Los valores son TICKERS de Yahoo (p.ej. DXY no existe como ticker, es
# "DX-Y.NYB"). Un activo puede anularlo con "benchmark_yf" (p.ej. cobre → DBB).
DEFAULT_BENCHMARK: Final[dict[str, str]] = {
    "metales": "DBP",
    "energia": "DBE",
    "indices": "SPY",
    "agro": "DBA",
    "crypto": "NCI",
    "fx": "DX-Y.NYB",
}

# Días de negociación por año según la categoría. Los pares FX y los activos
# macro cotizan ~252 sesiones hábiles; las criptomonedas negocian TODOS los
# días (365). Afecta a la anualización de la volatilidad (√365 en vez de
# √252), a las ventanas 1M/1Y (30/365 días) y a la duración del "mes" en el
# canal log del dashboard. Cualquier categoría ausente se asume 252.
DIAS_ANIO: Final[dict[str, int]] = {
    "crypto": 365,
}


def dias_anio_de(categoria: str) -> int:
    """Días de negociación por año de una categoría (252 por defecto).

    Args:
        categoria: Clave de categoría (``crypto``, ``metales``, ...).

    Returns:
        Días de trading al año (365 para crypto, 252 para el resto).
    """
    return DIAS_ANIO.get(categoria, 252)

# Etiqueta legible de un ticker de benchmark (para las tarjetas).
BENCHMARK_LABELS: Final[dict[str, str]] = {
    "DX-Y.NYB": "DXY",
}

# Tipos de interés por divisa de RESERVA — último respaldo si FRED no responde
# (sin API key o sin red). Valores de política/interbancarios verificados el
# 2026-08-27: FED efectiva 3.63% · ECB depo 2.25% · SONIA 3.75% · BOJ 1.00% ·
# SNB 0.00% · RBA 4.35% · BOC 2.25% · RBNZ 2.50%. La fuente primaria es FRED
# (scripts/data_fred.py); este dict solo actúa de fallo seguro, no se inventa.
TIPOS_RESERVA: Final[dict[str, float]] = {
    "USD": 3.63,
    "EUR": 2.25,
    "GBP": 3.75,
    "JPY": 1.00,
    "CHF": 0.00,
    "AUD": 4.35,
    "CAD": 2.25,
    "NZD": 2.50,
}

MACRO_ASSETS: Final[dict[str, dict[str, str]]] = {
    # Metales (futuros COMEX/NYMEX)
    "GOLD": {"categoria": "metales", "label": "Oro", "ticker_yf": "GC=F"},
    "SILVER": {"categoria": "metales", "label": "Plata", "ticker_yf": "SI=F"},
    "COPPER": {"categoria": "metales", "label": "Cobre", "ticker_yf": "HG=F", "benchmark_yf": "DBB"},
    "PLATINUM": {"categoria": "metales", "label": "Platino", "ticker_yf": "PL=F"},
    "PALLADIUM": {"categoria": "metales", "label": "Paladio", "ticker_yf": "PA=F"},
    # Energía (futuros NYMEX/ICE)
    "CRUDE": {"categoria": "energia", "label": "Crudo WTI", "ticker_yf": "CL=F"},
    "NATGAS": {"categoria": "energia", "label": "Gas natural", "ticker_yf": "NG=F"},
    "BRENT": {"categoria": "energia", "label": "Brent", "ticker_yf": "BZ=F"},
    "GASOLINE": {"categoria": "energia", "label": "Gasolina RBOB", "ticker_yf": "RB=F"},
    "HEATINGOIL": {"categoria": "energia", "label": "Gasóleo ULSD", "ticker_yf": "HO=F"},
    # Índices (cash o proxy)
    "SPX": {"categoria": "indices", "label": "S&P 500", "ticker_yf": "^GSPC"},
    "NDX": {"categoria": "indices", "label": "Nasdaq 100", "ticker_yf": "^NDX"},
    "DAX": {"categoria": "indices", "label": "DAX", "ticker_yf": "^GDAXI"},
    "NIKKEI": {"categoria": "indices", "label": "Nikkei 225", "ticker_yf": "^N225"},
    "DXY": {"categoria": "indices", "label": "Dólar índice (DXY)", "ticker_yf": "DX-Y.NYB"},
    # Agro (futuros CBOT / ICE)
    "WHEAT": {"categoria": "agro", "label": "Trigo", "ticker_yf": "ZW=F"},
    "CORN": {"categoria": "agro", "label": "Maíz", "ticker_yf": "ZC=F"},
    "SOYBEAN": {"categoria": "agro", "label": "Soja", "ticker_yf": "ZS=F"},
    "COFFEE": {"categoria": "agro", "label": "Café", "ticker_yf": "KC=F"},
    "COCOA": {"categoria": "agro", "label": "Cacao", "ticker_yf": "CC=F"},
    # Pares mayores FX (tarjetas de la pestaña FX; benchmark DXY).
    # No se inyectan como columnas del dashboard de gráficas: el generador
    # los filtra (los 15 pares FX ya son columnas nativas de esa sección).
    "EURUSD": {"categoria": "fx", "label": "EUR/USD", "ticker_yf": "EURUSD=X"},
    "GBPUSD": {"categoria": "fx", "label": "GBP/USD", "ticker_yf": "GBPUSD=X"},
    "USDJPY": {"categoria": "fx", "label": "USD/JPY", "ticker_yf": "USDJPY=X"},
    "USDCHF": {"categoria": "fx", "label": "USD/CHF", "ticker_yf": "USDCHF=X"},
    "AUDUSD": {"categoria": "fx", "label": "AUD/USD", "ticker_yf": "AUDUSD=X"},
    "USDCAD": {"categoria": "fx", "label": "USD/CAD", "ticker_yf": "USDCAD=X"},
    "NZDUSD": {"categoria": "fx", "label": "NZD/USD", "ticker_yf": "NZDUSD=X"},
    # Cruces mayores (añadidos 2026-08-27): con los 7 de arriba suman 10 tarjetas.
    "EURGBP": {"categoria": "fx", "label": "EUR/GBP", "ticker_yf": "EURGBP=X"},
    "EURJPY": {"categoria": "fx", "label": "EUR/JPY", "ticker_yf": "EURJPY=X"},
    "GBPJPY": {"categoria": "fx", "label": "GBP/JPY", "ticker_yf": "GBPJPY=X"},
    # Criptomonedas (categoría crypto, añadidas 2026-08-30): cotizan los 365
    # días del año y su benchmark es el Nasdaq Crypto Index (NCI), descargado
    # de la API pública de nasdaq.com (ver macro_data._nci_historico).
    "BTC": {"categoria": "crypto", "label": "Bitcoin", "ticker_yf": "BTC-USD"},
    "ETH": {"categoria": "crypto", "label": "Ethereum", "ticker_yf": "ETH-USD"},
    "BNB": {"categoria": "crypto", "label": "BNB", "ticker_yf": "BNB-USD"},
    "XRP": {"categoria": "crypto", "label": "XRP", "ticker_yf": "XRP-USD"},
    "SOL": {"categoria": "crypto", "label": "Solana", "ticker_yf": "SOL-USD"},
}

# Curva del Tesoro de EE.UU. (categoría bonos): los 5 tramos estándar con su
# serie de rendimiento "constant maturity" en FRED. Estas series DGS* SON los
# "Daily Treasury Par Yield Curve Rates" que publica el propio Tesoro (FRED los
# sirve sin necesidad de saltar su anti-bot). No son activos con retorno/vol:
# se muestran como curva (nivel + cambios en pb) en una sola card.
BONOS_TRAMOS: Final[dict[str, dict[str, object]]] = {
    "TB3M": {"label": "3 meses", "series": "DGS3MO", "venc_anos": 0.25},
    "TB2Y": {"label": "2 años", "series": "DGS2", "venc_anos": 2},
    "TB5Y": {"label": "5 años", "series": "DGS5", "venc_anos": 5},
    "TB10Y": {"label": "10 años", "series": "DGS10", "venc_anos": 10},
    "TB30Y": {"label": "30 años", "series": "DGS30", "venc_anos": 30},
}


def activos_de_categoria(categoria: str) -> list[str]:
    """Devuelve las claves de los activos de una categoría, en el orden del catálogo.

    Args:
        categoria: Clave de categoría (``metales``, ``energia``, ``indices``, ``agro``).

    Returns:
        Lista de claves de activos de esa categoría.
    """
    return [k for k, v in MACRO_ASSETS.items() if v["categoria"] == categoria]


def benchmark_de(activo: str) -> str:
    """Benchmark yfinance de un activo, según categoría o override propio.

    Args:
        activo: Clave del activo (p.ej. ``GOLD`` o ``COPPER``).

    Returns:
        Ticker de Yahoo del benchmark (``SPY``, ``DBC``, ``DBA``, ``DBP``, ``DBB``).
    """
    spec = MACRO_ASSETS[activo]
    return spec.get("benchmark_yf") or DEFAULT_BENCHMARK[spec["categoria"]]


def carry_de_par(key: str, tipos: Optional[dict[str, float]] = None) -> Optional[float]:
    """Carry anual de un par FX en % = tipo(base) − tipo(cotizada) (long del par).

    Args:
        key: Clave del par (p.ej. ``EURUSD``).
        tipos: Dict ``{divisa: puntos_%}`` con los tipos vigentes (p.ej. los de
            ``data_fred.datos_tipos()``). Si es None, usa ``TIPOS_RESERVA``.

    Returns:
        Diferencial de tipos en puntos porcentuales (negativo si la base paga
        menos que la cotizada), o ``None`` si no es un par FX del catálogo o
        falta el tipo de alguna divisa.
    """
    spec = MACRO_ASSETS.get(key)
    if not spec or spec["categoria"] != "fx":
        return None
    tipos = tipos or TIPOS_RESERVA
    base, cotizada = key[:3], key[3:6]
    tipo_base = tipos.get(base)
    tipo_cotizada = tipos.get(cotizada)
    if tipo_base is None or tipo_cotizada is None:
        return None
    return tipo_base - tipo_cotizada
