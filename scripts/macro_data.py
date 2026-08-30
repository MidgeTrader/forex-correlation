"""Adaptador de datos yfinance autocontenido.

`MacroDataYF` carga precios de un activo macro desde Yahoo Finance y los
expone como `price_data`, `spy_data`, `info={}` y estados a None, de modo
que las métricas de `macro_metrics` (y cualquier consumidor futuro) funcionan
sin cambios.

La descarga usa memo por proceso + reintentos con backoff + `repair` y guarda
cada serie como CSV en `data/` (caché en disco): regenerar el dashboard no
vuelve a descargar salvo `refresh=True` o si la caché está desactualizada
(auto-refresh, ver `MacroDataYF._cache_desactualizada`).

Regla de honestidad: cualquier dato no disponible → `None` (→ N/A), nunca 0 ni
un valor inventado.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pandas.tseries.offsets import BDay

from macro_catalogo import MACRO_ASSETS, benchmark_de, dias_anio_de

# Caché en disco: `data/` en la raíz del proyecto (un nivel por encima de scripts/).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Reconstrucción del DXY (ver `MacroDataYF._dxy_reconstruido`): la serie
# ``DX-Y.NYB`` que entrega Yahoo tiene retornos diarios casi descorrelacionados
# del mercado (validado 2026-08-28: corr EURUSD −0.09 vs −0.97 reconstruido),
# así que para los pares FX se usa la fórmula geométrica oficial de ICE.
# Pesos y tickers de los pares componentes; ``inverso=True`` → el ticker cotiza
# EUR o GBP y se usa ``1/precio`` (USD/EUR, USD/GBP).
_DXY_COMPONENTES: dict[str, tuple[str, float, bool]] = {
    "USD_EUR": ("EURUSD=X", 0.576, True),
    "USD_JPY": ("JPY=X", 0.136, False),
    "USD_GBP": ("GBPUSD=X", 0.119, True),
    "USD_CAD": ("CAD=X", 0.091, False),
    "USD_SEK": ("SEK=X", 0.042, False),
    "USD_CHF": ("CHF=X", 0.036, False),
}
_DXY_NIVEL_BASE = 50.14348112  # constante de normalización oficial del DXY


# ---------------------------------------------------------------------------
# Descarga con memo por proceso + reintentos con backoff
# ---------------------------------------------------------------------------

_PRICE_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
_PRICE_CACHE_LOCK = Lock()


def _retry_call(fn, attempts: int = 3, base_delay: float = 0.5):
    """Ejecuta `fn` reintentando ante fallos transitorios (backoff exponencial).

    yfinance falla a veces de forma temporal (rate limit, timeout, "Failed
    download"). Reintentar con espera creciente absorbe esos fallos. Si todos
    los intentos fallan, se re-lanza la excepción (el llamador decide el fallo
    seguro → N/A).
    """
    delay = base_delay
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt + 1 >= attempts:
                raise
            time.sleep(delay)
            delay *= 2
    return None  # no alcanzable; por claridad para el linter


def _normalize_columns(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Reduce el MultiIndex de yfinance a un solo nivel."""
    if df is None or df.empty:
        return pd.DataFrame()
    result = df.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)
    return result


def _download_with_retry(ticker: str, period: str, repair: bool) -> pd.DataFrame:
    """Un único `yf.download` con reintentos con backoff exponencial.

    Devuelve el DataFrame normalizado, o uno vacío si todos los intentos
    fallan (fallo seguro, nunca una excepción hacia arriba).
    """

    def _call() -> pd.DataFrame:
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            repair=repair,
        )
        return _normalize_columns(df)

    try:
        return _retry_call(_call)
    except Exception:
        return pd.DataFrame()


def _download_series(ticker: str, period: str) -> pd.DataFrame:
    """Descarga una serie de precios, con memo por-proceso y fallback.

    - Memo: si la serie ya se descargó en esta ejecución, se devuelve la copia
      en memoria sin repetir la petición de red. Vive solo durante el proceso,
      nunca en disco, así que no hay riesgo de servir datos obsoletos.
    - Fallback: si `repair=True` falla (p.ej. sin sklearn en algún ticker), se
      reintenta con `repair=False` en vez de descartar.
    - Retry: cada intento reintenta ante fallos transitorios con backoff.
    """
    key = (ticker, period)

    with _PRICE_CACHE_LOCK:
        cached = _PRICE_CACHE.get(key)
    if cached is not None:
        return cached

    for repair in (True, False):
        df = _download_with_retry(ticker, period, repair)

        if not df.empty:
            with _PRICE_CACHE_LOCK:
                _PRICE_CACHE[key] = df
            return df

    return pd.DataFrame()


def _nombre_seguro(ticker: str) -> str:
    """Sanitiza un ticker para usarlo como nombre de archivo (p.ej. ``^GSPC`` → ``_GSPC``)."""
    return ticker.replace("^", "_")


class MacroDataYF:
    """Carga de precios de un activo macro desde Yahoo Finance.

    Expone los datos normalizados (`price_data`, `spy_data`, `info`, estados)
    para que `macro_metrics` funcione sin cambios. Sin dependencias privadas.
    """

    def __init__(
        self,
        key: str,
        period: str = "5y",
        benchmark: Optional[str] = None,
        refresh: bool = False,
    ):
        spec = MACRO_ASSETS[key]
        self.ticker = key
        self.period = period
        self.ticker_yf = spec["ticker_yf"]
        # Benchmark por defecto según la categoría; `benchmark` explícito lo anula.
        self.benchmark_yf = benchmark or benchmark_de(key)
        self.label = spec["label"]
        self.categoria = spec["categoria"]
        # Días de negociación al año (365 en crypto, 252 en el resto): anualiza
        # la volatilidad y define las ventanas 1M/1Y y el "mes" del canal log.
        self.dias_anio = dias_anio_de(self.categoria)
        self.refresh = refresh

        self.price_data: Optional[pd.DataFrame] = None
        self.spy_data: Optional[pd.DataFrame] = None

        self.info: dict = {}

        self.reporting_currency = "USD"
        self.fx_to_usd: Optional[float] = 1.0

        self.error: Optional[str] = None

    @staticmethod
    def _cache_desactualizada(df: pd.DataFrame) -> bool:
        """True si la caché se queda por detrás del penúltimo día hábil.

        Permite auto-refrescar las cards: si la última sesión del CSV es
        anterior al penúltimo día hábil, la serie se re-descarga al regenerar
        el dashboard (sin depender de ``--refresh``). En fin de semana no
        dispara: el último dato (viernes) sigue siendo el penúltimo día hábil
        visto desde sábado/lunes.
        """
        ultimo_dato = df.index.max().normalize()
        penultimo_habil = pd.Timestamp.today().normalize() - BDay(1)
        return bool(ultimo_dato < penultimo_habil)

    def _serie_con_cache(self, ticker: str, period: str) -> pd.DataFrame:
        """Serie de precios del ticker, sirviéndose de la caché en disco.

        Si el CSV ya existe en `data/` (y no se pidió `refresh`), se carga de él
        siempre que esté al día; una caché desactualizada (último dato anterior
        al penúltimo día hábil) se re-descarga sola, de modo que al regenerar el
        dashboard las cards siempre coinciden con la pestaña de gráficas (que
        descarga en vivo). Si no existe o se pidió `refresh`, se descarga y se
        guarda. Una caché corrupta se descarta y se re-descarga (fallo seguro).

        Args:
            ticker: Ticker de Yahoo (p.ej. ``GC=F``).
            period: Periodo de histórico (``5y``, ``1y``, ...).

        Returns:
            DataFrame con columnas Open/High/Low/Close/Volume indexado por fecha,
            o vacío si no hay datos.
        """
        ruta = DATA_DIR / f"{_nombre_seguro(ticker)}.csv"

        if ruta.exists() and not self.refresh:
            try:
                df = pd.read_csv(ruta, index_col=0, parse_dates=True)
                if not df.empty and "Close" in df.columns and not self._cache_desactualizada(df):
                    return df
            except Exception:
                pass  # caché corrupta o vieja → re-descargar

        df = _download_series(ticker, period)

        if not df.empty:
            try:
                ruta.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(ruta)
            except OSError:
                pass  # sin permiso de escritura: se sigue con datos en memoria

        return df

    def _dato_benchmark(self) -> pd.DataFrame:
        """Serie del benchmark para las métricas.

        Dos casos especiales que Yahoo no entrega bien:
          - El DXY (``DX-Y.NYB``) se RECONSTRUYE con la fórmula de ICE desde sus
            6 pares componentes en lugar de usar la serie que entrega Yahoo: la
            de Yahoo tiene retornos diarios casi descorrelacionados del mercado
            (validado el 2026-08-28: corr con EURUSD −0.09 vs −0.97
            reconstruido), aunque sus niveles sean correctos.
          - El NCI (Nasdaq Crypto Index) no tiene histórico en Yahoo (solo el
            último cierre) y se descarga de la API pública de nasdaq.com.
        El resto de benchmarks (SPY, DBA, DBP, DBB, DBE) se descarga tal cual.

        Returns:
            DataFrame con columna ``Close``, o vacío si el benchmark no pudo
            obtenerse (fallo seguro → la métrica devuelve N/A).
        """
        if self.benchmark_yf == "NCI":
            return self._nci_historico()
        if self.benchmark_yf != "DX-Y.NYB":
            return self._serie_con_cache(self.benchmark_yf, self.period)
        return self._dxy_reconstruido()

    def _dxy_reconstruido(self) -> pd.DataFrame:
        """DXY reconstruido con la fórmula geométrica oficial de ICE.

        ``DXY = NIVEL_BASE · Π (USD/divisa)^peso``, donde las divisas que se
        cotizan en dólares (EUR, GBP) entran como ``1/precio``. Se sirve de la
        caché en disco por componente y guarda el resultado para no recalcular.

        Returns:
            DataFrame con columna ``Close`` indexado por fecha, o vacío si
            falta algún componente (la métrica lo trata como N/A).
        """
        ruta = DATA_DIR / "_DXY_RECONSTRUIDO.csv"

        if ruta.exists() and not self.refresh:
            try:
                df = pd.read_csv(ruta, index_col=0, parse_dates=True)
                if not df.empty and "Close" in df.columns and not self._cache_desactualizada(df):
                    return df
            except Exception:
                pass  # caché corrupta → recalcular

        try:
            componentes: dict[str, pd.Series] = {}
            for nombre, (ticker, _peso, inverso) in _DXY_COMPONENTES.items():
                serie = self._serie_con_cache(ticker, self.period)["Close"].dropna()
                if serie.empty:
                    return pd.DataFrame()
                componentes[nombre] = 1.0 / serie if inverso else serie

            frame = pd.concat(componentes, axis=1).dropna()
            if frame.empty:
                return pd.DataFrame()

            log_dxy = sum(
                np.log(frame[nombre]) * peso
                for nombre, (_ticker, peso, _inverso) in _DXY_COMPONENTES.items()
            )
            out = pd.DataFrame({"Close": _DXY_NIVEL_BASE * np.exp(log_dxy)}).dropna()

            try:
                out.to_csv(ruta)
            except OSError:
                pass  # sin permiso de escritura: se sigue con datos en memoria

            return out
        except Exception:
            return pd.DataFrame()

    def _nci_historico(self) -> pd.DataFrame:
        """Histórico diario del Nasdaq Crypto Index (NCI) desde nasdaq.com.

        Yahoo solo entrega el último cierre del NCI (sin histórico), así que se
        descarga de la API pública que alimenta nasdaq.com (``api.nasdaq.com/
        api/quote/NCI/chart``), sin clave y con caché en disco como el resto de
        series. El NCI existe desde abril de 2021; se piden 5 años.

        Returns:
            DataFrame con columna ``Close`` indexado por fecha, o vacío si el
            índice no pudo obtenerse (fallo seguro → la métrica lo trata N/A).
        """
        ruta = DATA_DIR / "NCI.csv"

        if ruta.exists() and not self.refresh:
            try:
                df = pd.read_csv(ruta, index_col=0, parse_dates=True)
                if not df.empty and "Close" in df.columns and not self._cache_desactualizada(df):
                    return df
            except Exception:
                pass  # caché corrupta o vieja → re-descargar

        hoy = pd.Timestamp.today()
        params = {
            "assetclass": "index",
            "fromdate": (hoy - pd.DateOffset(years=5)).strftime("%Y-%m-%d"),
            "todate": hoy.strftime("%Y-%m-%d"),
            "period": "daily",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
            "Accept": "application/json",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/",
        }
        try:
            resp = requests.get(
                "https://api.nasdaq.com/api/quote/NCI/chart",
                params=params, headers=headers, timeout=20,
            )
            resp.raise_for_status()
            chart = (resp.json().get("data") or {}).get("chart") or []
            filas = [(pd.to_datetime(c["z"]["dateTime"]), float(c["y"])) for c in chart]
        except Exception:
            return pd.DataFrame()

        if not filas:
            return pd.DataFrame()

        df = pd.DataFrame(filas, columns=["Date", "Close"]).set_index("Date")
        df = df.sort_index().dropna()
        # Fechas duplicadas (intradía o ajustes): última observación de cada día.
        df = df[~df.index.duplicated(keep="last")]

        if not df.empty:
            try:
                ruta.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(ruta)
            except OSError:
                pass  # sin permiso de escritura: se sigue con datos en memoria

        return df

    def fetch(self) -> bool:
        """Descarga el activo y su benchmark, con caché en disco.

        Returns:
            True si hay datos de precio del activo; False (con `self.error`)
            si no, fallo seguro.
        """
        try:
            self.price_data = self._serie_con_cache(self.ticker_yf, self.period)

            if self.price_data is None or self.price_data.empty:
                self.error = "Sin datos de precio."
                return False

            self.spy_data = self._dato_benchmark()

            return True

        except Exception as exc:
            self.error = str(exc)
            return False
