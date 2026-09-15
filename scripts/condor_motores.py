"""Motor de las pestañas de iron condors: volatilidad, opciones y rangos por periodo.

Cada pestaña es una entrada de ``SUBYACENTES`` (hoy XSP, el Mini-SPX, y QQQ, el ETF
del Nasdaq-100). El módulo no sabe de ninguno en particular: lo que cambia de uno a
otro —los tickers, el nombre de su índice de volatilidad, su caché— viaja en el
``Subyacente``, y lo que se calcula es idéntico, así que se escribe una sola vez.

El subyacente es el instrumento sobre el que se venden los iron condors, así que **todo
se expresa en % del precio** — la unidad con la que se comparan entre sí los strikes,
las bandas de movimiento esperado y los rangos históricos. Los puntos se muestran al
lado porque son los que se teclean en la orden, nunca en lugar del % (en dólares cuando
el subyacente es un ETF, que es como se lee su orden).

Tres bloques, todos con datos reales o ``N/A`` honesto:

  1. VOLATILIDAD — volatilidad realizada (HV) a 10/20/30/60/90 días anualizada,
     el índice de volatilidad del subyacente (VIX para XSP, VXN para QQQ) con su
     percentil sobre la historia pedida, y la prima de volatilidad (VIX / HV30).
  2. OPCIONES — para los vencimientos 0 DTE (la expiración de HOY), 1 DTE (la de
     mañana) y SEMANAL (el primer viernes), la IV ATM, el movimiento esperado
     (1σ) en % y en $, el ATR de referencia con su banda, y las bandas 1σ/2σ con
     su distancia OTM medida sobre los strikes reales de la cadena (las alas del
     condor, que se calculan para el OTM pero no se pintan: en los subyacentes de
     hoy el strike va de $1 en $1 cerca del dinero, así que coinciden con la
     banda redondeada y saldrían dos filas con los mismos números).
  3. RANGOS POR PERIODO — cuánto se mueve el subyacente, en %, en un día, una
     semana y un mes (media, mediana y último cerrado), con el desglose por día
     de la semana y por día del mes. Es la versión en % de las tablas de pips del
     dashboard FX: aquí no hay pips que valgan, hay porcentaje del precio.

MOVIMIENTO ESPERADO (1σ) — se deriva del straddle ATM, no de la IV cruda de
Yahoo, que sale ruidosa en strikes ilíctos (se han visto IV del 240 % en un
strike profundo). Para una opción ATM, ``straddle = 2 · N'(0) · S · σ√T =
0.7979 · S · σ√T``; por tanto ``σ(puntos) = straddle / 0.7979``. La IV de la
misma card se enseña al lado como contraste, pero no entra en el cálculo del σ.

(La verificación cruzada que hubo aquí —straddle contra IV del mismo
vencimiento— se hizo con datos de SPY, antes de que la pestaña pasara a XSP, y
se ha retirado en vez de reetiquetarla: aunque el subyacente es el mismo S&P 500
y las cifras salen parecidas, una comprobación no se hereda cambiando el ticker.
Además, contrastar con la IV obliga a fijar la convención de T (días naturales o
de bolsa) y en un vencimiento con fin de semana por medio las dos convenciones
no dan lo mismo, así que la comparación no era tan limpia como parecía.)

NORMALIZACIÓN DE RANGOS — el detalle que evita mentir con una media de 5 años:
el rango de cada periodo se divide por el **cierre del día anterior al inicio
del periodo**, es decir el movimiento contado desde donde se entra. El gap
cuenta, porque el gap es riesgo real de quien vende un condor. Nunca se divide
por el precio de hoy: eso sesgaría la media por el nivel de precio (el error que
ya mordió con BTC en el estudio de reversión semanal).

ATR — se añade el ATR de Wilder (14 periodos) como contraste entre lo que el
mercado PAGA (el 1σ de la card) y lo que el mercado HACE (el rango real). Se lee
en el último periodo CERRADO, nunca en el que se está formando: el ATR que sirve
para decidir una entrada es el que se ve al entrar —el de ayer al cierre en los
ciclos diarios, el de la semana pasada en el semanal—, no el que se completa
después con la sesión que ya estás operando. Es el mismo principio que la
normalización de rangos. ÚNICA excepción, pedida por el usuario: el ciclo de
1 DTE usa el ATR **con la sesión de hoy dentro** (``incluye_hoy``), porque esa
posición sí vive el día entero y un día muy volátil tiene que notarse en el
informe mientras se forma; la card lo dice en su fila de fecha (``↳ hoy`` frente
a ``↳ cierre``). El 0 DTE y el semanal no se tocan. Ojo: el ATR usa el true
range, que INCLUYE el hueco de apertura, mientras que el rango de la card «Día»
es alto−bajo a secas y no lo incluye; las dos cifras se parecen, pero no son lo
mismo.

Caché: los precios de cada subyacente viven en ``data/{clave}_precios.csv`` (él y su
índice de volatilidad, con las columnas prefijadas por su nombre —``XSP_Close`` frente
a ``QQQ_Close``— para que cada fichero se lea solo) con auto-refresh si quedan por
detrás del último día hábil, el mismo patrón que el resto del dashboard. La cadena de
opciones NO se cachea: es una foto del momento y caduca en minutos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Final, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

# Caché en disco: `data/` en la raíz del proyecto (un nivel por encima de scripts/).
DATA_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "data"

# Sesiones de bolsa al año para anualizar la volatilidad realizada.
DIAS_ANIO: Final[int] = 252

# Horario regular del mercado de EE.UU. en su propio huso. Solo lo necesita el spot de
# los subyacentes que cotizan TAMBIÉN fuera de hora (ver ``cotiza_extendido``): fuera de
# este tramo su «último precio» no es el que la cadena de opciones está mirando.
ZONA_MERCADO: Final[ZoneInfo] = ZoneInfo("America/New_York")
APERTURA_MERCADO: Final[time] = time(9, 30)
CIERRE_MERCADO: Final[time] = time(16, 0)

# Ventanas de la volatilidad realizada (días de cotización).
VENTANAS_HV: Final[tuple[int, ...]] = (10, 20, 30, 60, 90)

# Periodos del ATR de Wilder (días en el diario, semanas en el semanal).
VENTANA_ATR: Final[int] = 14

# Constante de la derivación del straddle ATM: 2 · N'(0) = 0.7979.
_STRADDLE_A_SIGMA: Final[float] = 0.7979

# IV fuera de este rango se considera basura de Yahoo (strikes ilíctos).
_IV_MIN: Final[float] = 0.01
_IV_MAX: Final[float] = 3.0

# Umbral de antigüedad del último dato para considerar la caché al día: si la
# última sesión es anterior al penúltimo día hábil, se re-descarga (mismo
# criterio que `bonos_motores`).
_ULTIMO_OK: Final[pd.Timestamp] = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)

_NOMBRES_DOW: Final[dict[int, str]] = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
}


# ---------------------------------------------------------------------------
# Subyacentes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Subyacente:
    """Un instrumento sobre el que se venden iron condors.

    Todo lo que distingue una pestaña de otra vive aquí; el resto del módulo es común a
    todas. La ``etiqueta`` hace además de prefijo de las columnas de su caché
    (``XSP_Close`` frente a ``QQQ_Close``), de modo que cada fichero de ``data/`` se lee
    solo y no depende de con qué pestaña se abrió.

    Attributes:
        clave: Identificador corto (``xsp``): ids de las cards, clave del nav y nombre
            del fichero de caché (``data/{clave}_precios.csv``). Acaba en un ``id`` de
            HTML y en una ruta, así que es minúscula y ASCII.
        etiqueta: Nombre que se pinta (``XSP``) y prefijo de las columnas en disco.
        ticker: Símbolo de Yahoo del subyacente (``^XSP``, ``QQQ``).
        ticker_vol: Símbolo de Yahoo de su índice de volatilidad (``^VIX``, ``^VXN``).
        nombre_vol: Cómo se llama ese índice (``VIX``, ``VXN``): se pinta en las filas y
            es el prefijo de su columna en la caché.
        cotiza_extendido: True si el subyacente cotiza TAMBIÉN fuera del horario regular
            (un ETF). Su «último precio» puede ser entonces de after-hours, mientras que
            la cadena de opciones está cotizada contra el cierre de la sesión; fuera de
            horario se usa el cierre, que es el precio con el que la cadena está de
            acuerdo. Un índice no tiene este problema (``False``).
        anios: Años de historia para el percentil y las medias de rango.
    """

    clave: str
    etiqueta: str
    ticker: str
    ticker_vol: str
    nombre_vol: str
    cotiza_extendido: bool
    anios: int = 5

    @property
    def columnas_disco(self) -> dict[str, str]:
        """``{nombre en el CSV: nombre interno}`` de la caché de precios.

        Las columnas se guardan con el nombre del subyacente delante para que el CSV se
        lea solo; dentro del módulo se trabaja con nombres genéricos, de modo que los
        cálculos no sepan de quién es la serie. El renombrado vive únicamente aquí, en
        las dos funciones que tocan el disco.
        """
        return {
            f"{self.etiqueta}_Close": "Close",
            f"{self.etiqueta}_High": "High",
            f"{self.etiqueta}_Low": "Low",
            f"{self.nombre_vol}_Close": "Vol_Close",
        }

    @property
    def columnas_internas(self) -> dict[str, str]:
        """El inverso de ``columnas_disco``: ``{nombre interno: nombre en el CSV}``."""
        return {interno: disco for disco, interno in self.columnas_disco.items()}

    @property
    def ruta_cache(self) -> Path:
        """Fichero de caché de precios de este subyacente dentro de ``data/``."""
        return DATA_DIR / f"{self.clave}_precios.csv"


# Las pestañas de iron condors. El orden en que se pintan NO se decide aquí (es cosa de
# `generar_dashboard`, la raíz de composición): esto es solo el registro.
SUBYACENTES: Final[dict[str, Subyacente]] = {
    "xsp": Subyacente(
        clave="xsp",
        etiqueta="XSP",
        ticker="^XSP",
        ticker_vol="^VIX",
        nombre_vol="VIX",
        cotiza_extendido=False,
    ),
    "qqq": Subyacente(
        clave="qqq",
        etiqueta="QQQ",
        ticker="QQQ",
        ticker_vol="^VXN",
        nombre_vol="VXN",
        cotiza_extendido=True,
    ),
}

# La clave acaba en un `id` de HTML y en un nombre de fichero: una mayúscula o un acento
# rompería el primero o el segundo. Se comprueba al importar para que el fallo salte aquí
# y no a mitad del render.
for _subyacente in SUBYACENTES.values():
    if not (_subyacente.clave.isascii() and _subyacente.clave.islower()
            and _subyacente.clave.isalnum()):
        raise ValueError(
            f"Clave de subyacente inválida: {_subyacente.clave!r} "
            "(tiene que ser minúscula, ASCII y alfanumérica: va a un id de HTML y a una ruta)"
        )


# ---------------------------------------------------------------------------
# Estructuras de resultado
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ATR:
    """Average True Range de referencia, leído en el último periodo CERRADO.

    Contesta a «¿cuánto se mueve el subyacente de verdad?» frente al 1σ de la card, que
    contesta «¿cuánto dice el mercado que se va a mover?». Se lee en el cierre
    anterior a propósito: el ATR que decide una entrada es el que se ve al
    entrar, no el que se completa después con la sesión que ya se está operando.

    Attributes:
        valor: ATR en puntos del subyacente.
        pct: ATR como fracción del precio de referencia (0.012 = 1.2 %).
        desde: Última SESIÓN del periodo del que sale el ATR (ISO).
        etiqueta: ``ATR(14) diario`` o ``ATR(14) semanal``.
        incluye_hoy: True si la cifra lleva dentro la sesión de HOY (en curso o
            ya cerrada). False si sale de un periodo cerrado anterior.

    La única excepción a la regla del cierre es el ciclo de **1 DTE**, que sí
    incluye la sesión de hoy: esa posición vive el día entero, así que un día muy
    volátil tiene que notarse en el informe mientras se forma. Se paga con que la
    cifra se mueve durante la sesión y deja de ser comparable con la del 0 DTE,
    que sigue anclada al cierre de ayer.
    """

    valor: Optional[float] = None
    pct: Optional[float] = None
    desde: Optional[str] = None
    etiqueta: str = ""
    incluye_hoy: bool = False


@dataclass(frozen=True)
class Vencimiento:
    """Un vencimiento de opciones con sus bandas de movimiento esperado.

    Attributes:
        etiqueta: ``0 DTE``, ``1 DTE`` o ``Semanal``.
        fecha: Fecha de expiración (ISO).
        dte: Días naturales hasta la expiración.
        strike_atm: Strike más cercano al dinero usado para leer la IV.
        iv: IV ATM en fracción anualizada (0.136 = 13.6 %), o None.
        sigma_puntos: Movimiento esperado 1σ en puntos del subyacente, o None.
        sigma_pct: Movimiento esperado 1σ en % del precio, o None.
        banda1: ``(inferior, superior)`` a 1σ en puntos, o None.
        banda2: ``(inferior, superior)`` a 2σ en puntos, o None.
        ala1: Strikes REALES de la cadena más próximos a ±1σ (put, call), o None.
            No se pintan en la card (en los subyacentes de hoy el strike va de $1 en
            $1 cerca del dinero, así que coinciden con la banda redondeada y salían
            dos filas idénticas), pero se calculan: el OTM se mide sobre ellos, que
            son los strikes que de verdad se teclean en la orden.
        ala2: Strikes reales más próximos a ±2σ (put, call), o None.
        banda_atr: ``(inferior, superior)`` a ±1 ATR sobre el mismo spot, o None.
            Es el contraste de la banda σ: el mismo spot, pero movido por lo que
            el precio ha recorrido de verdad en vez de por lo que el mercado
            espera que recorra.
        atr: ATR de referencia de este ciclo (diario o semanal), o None.
    """

    etiqueta: str
    fecha: str
    dte: int
    strike_atm: Optional[float] = None
    iv: Optional[float] = None
    sigma_puntos: Optional[float] = None
    sigma_pct: Optional[float] = None
    banda1: Optional[tuple[float, float]] = None
    banda2: Optional[tuple[float, float]] = None
    ala1: Optional[tuple[float, float]] = None
    ala2: Optional[tuple[float, float]] = None
    banda_atr: Optional[tuple[float, float]] = None
    atr: Optional[ATR] = None


@dataclass(frozen=True)
class EstadisticaPeriodo:
    """Rango de un periodo (día, semana o mes) resumido en %.

    Attributes:
        etiqueta: ``Día``, ``Semana`` o ``Mes``.
        media_pct: Rango medio del periodo en % (None si no hay datos).
        mediana_pct: Mediana del rango en %.
        ultimo_pct: Rango del último periodo CERRADO en % (el periodo en curso
            se excluye: estaría incompleto y tiraría la lectura hacia abajo).
        ultimo_fin: Última SESIÓN del último periodo cerrado (ISO). Se guarda la
            fecha de cotización y no el ``start_time`` del periodo porque en las
            semanas y los meses pandas devuelve fechas que no son de sesión (la
            semana que termina el viernes 11 empieza, para pandas, el sábado 5).
        n: Número de periodos usados en la media.
    """

    etiqueta: str
    media_pct: Optional[float]
    mediana_pct: Optional[float]
    ultimo_pct: Optional[float]
    ultimo_fin: Optional[str]
    n: int


@dataclass(frozen=True)
class AnalisisCondor:
    """Resultado completo de una pestaña de iron condors.

    Lleva dentro el ``subyacente`` del que salen las cifras, y no es un adorno: el panel
    construye los rótulos a partir de él, así que las cifras de un subyacente no pueden
    pintarse bajo el nombre de otro.

    Attributes:
        subyacente: El instrumento analizado, o sea de quién son estas cifras.
        precio: Último precio disponible (spot si la cadena responde).
        fecha_precio: Fecha (y hora, si es spot) del precio mostrado.
        hv: ``{ventana: fracción anualizada}`` de volatilidad realizada.
        vol: Último cierre de su índice de volatilidad (VIX en XSP, VXN en QQQ).
        vol_percentil: Percentil de ese índice dentro de la historia (0-100).
        prima_vol: Índice de volatilidad / HV30 (la prima; < 1 = vender barato).
        vencimientos: Los vencimientos (0 DTE, 1 DTE y semanal) con sus bandas.
        periodos: Estadísticas de día, semana y mes.
        por_dia_semana: ``[(nombre, arriba %, abajo %, total %)]`` de lunes a
            viernes: medias de las excursiones diarias, partidas en los dos
            lados. Al ser medias, ``arriba + abajo = total`` cuadra exacto.
        por_dia_mes: ``[(día, rango medio %)]`` ordenado de mayor a menor.
        notas: Avisos honestos de qué no se pudo calcular y por qué.
    """

    subyacente: Subyacente
    precio: Optional[float] = None
    fecha_precio: str = ""
    hv: dict[int, Optional[float]] = field(default_factory=dict)
    vol: Optional[float] = None
    vol_percentil: Optional[float] = None
    prima_vol: Optional[float] = None
    vencimientos: list[Vencimiento] = field(default_factory=list)
    periodos: list[EstadisticaPeriodo] = field(default_factory=list)
    por_dia_semana: list[tuple[str, float, float, float]] = field(default_factory=list)
    por_dia_mes: list[tuple[int, float]] = field(default_factory=list)
    notas: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Precios (con caché en disco)
# ---------------------------------------------------------------------------

def _cache_ok(s: Subyacente) -> Optional[pd.DataFrame]:
    """Carga la caché de precios del subyacente si sirve; si no, None.

    El orden de las comprobaciones importa: el esquema se valida ANTES que la fecha. Un
    CSV con otras columnas no es una caché «fresca pero rara», es un fichero que no se
    puede usar. Detectar eso al principio evita el fallo tardío que salía si no: se leía
    con nombres genéricos un CSV con columnas prefijadas y reventaba con un KeyError a
    mitad del análisis, con todo lo demás ya descargado y sin llegar a escribir el HTML.

    Args:
        s: Subyacente cuya caché se quiere.

    Returns:
        Los precios con nombres de columna genéricos, o None si no hay caché utilizable
        (no existe, está corrupta, es de otro subyacente o se queda atrás).
    """
    if not s.ruta_cache.exists():
        return None
    try:
        df = pd.read_csv(s.ruta_cache, index_col=0, parse_dates=True)
    except Exception:
        return None  # caché corrupta → re-descargar
    if df.empty:
        return None

    columnas = s.columnas_disco
    if set(df.columns) != set(columnas):
        # Se dice en voz alta a propósito: `data/` está fuera de git, así que este aviso
        # es el único rastro que queda de que la caché se descartó.
        print(f"  Caché de {s.etiqueta} con columnas inesperadas "
              f"({', '.join(map(str, df.columns))}): se re-descarga.")
        return None

    if df.index.max().normalize() < _ULTIMO_OK:
        return None  # caché desactualizada → re-descargar

    return df.rename(columns=columnas)


def _a_fechas(serie: pd.DataFrame) -> pd.DataFrame:
    """Reindexa un histórico de Yahoo por fecha (sin hora y sin zona horaria).

    yfinance 1.7 devuelve el índice con zona horaria y con la hora de sesión,
    de modo que el subyacente y su índice de volatilidad llegan con timestamps
    distintos para el MISMO día: al combinarlos, pandas no encuentra ni una fecha
    en común y el join se queda vacío. Se normaliza a fecha pura antes de cruzar
    las dos series.

    Args:
        serie: Histórico diario de yfinance.

    Returns:
        El mismo histórico indexado por fecha (medianoche, sin zona). Si dos
        velas cayeran en la misma fecha, se conserva la última.
    """
    serie = serie.copy()
    indice = pd.to_datetime(serie.index)
    if indice.tz is not None:
        indice = indice.tz_localize(None)
    serie.index = indice.normalize()
    return serie[~serie.index.duplicated(keep="last")]


def _descargar_precios(s: Subyacente) -> Optional[pd.DataFrame]:
    """Descarga la historia del subyacente (OHLC) y de su volatilidad (cierre) de Yahoo.

    Args:
        s: Subyacente a descargar.

    Returns:
        DataFrame con ``Close``, ``High``, ``Low`` y ``Vol_Close`` indexado por fecha, o
        None si Yahoo no responde o falta alguna columna (fallo seguro: la card muestra
        N/A, nunca un valor inventado).
    """
    try:
        indice = yf.Ticker(s.ticker).history(period=f"{s.anios}y", interval="1d")
        vol = yf.Ticker(s.ticker_vol).history(period=f"{s.anios}y", interval="1d")
    except Exception:
        return None  # sin red / Yahoo caído

    if indice.empty or vol.empty:
        return None

    indice, vol = _a_fechas(indice), _a_fechas(vol)

    try:
        df = pd.DataFrame({
            "Close": indice["Close"],
            "High": indice["High"],
            "Low": indice["Low"],
            "Vol_Close": vol["Close"],
        })
    except KeyError:
        return None  # Yahoo cambió el esquema de columnas

    # Join interno: un día sin el índice de volatilidad (o sin el subyacente) no se
    # rellena a ojo, se descarta.
    df = df.dropna(how="any")
    return df if not df.empty else None


def analizar_precios(s: Subyacente, refresh: bool = False) -> Optional[pd.DataFrame]:
    """Precios del subyacente y de su índice de volatilidad, o None sin datos.

    Args:
        s: Subyacente a analizar.
        refresh: True fuerza la re-descarga (ignora la caché en disco).

    Returns:
        DataFrame de ``_descargar_precios``, o None si no hay forma de obtenerlo.
    """
    if not refresh:
        cached = _cache_ok(s)
        if cached is not None:
            return cached

    df = _descargar_precios(s)
    if df is None:
        return None

    try:
        # Se guarda con las columnas del subyacente (`XSP_Close`, `QQQ_Close`) para que
        # el CSV se lea solo; dentro del módulo se usan los nombres genéricos.
        s.ruta_cache.parent.mkdir(parents=True, exist_ok=True)
        df.rename(columns=s.columnas_internas).to_csv(s.ruta_cache)
    except OSError:
        pass  # sin permiso de escritura: se sigue con datos en memoria

    return df


# ---------------------------------------------------------------------------
# Volatilidad realizada, VIX y prima de volatilidad
# ---------------------------------------------------------------------------

def _volatilidad_realizada(close: pd.Series, ventana: int) -> Optional[float]:
    """Volatilidad realizada anualizada de las últimas ``ventana`` sesiones.

    Args:
        close: Serie de cierres.
        ventana: Número de sesiones de la ventana.

    Returns:
        Desviación típica de los retornos logarítmicos diarios anualizada
        (√252) en fracción, o None si no hay suficientes datos.
    """
    if len(close) <= ventana:
        return None
    retornos = np.log(close / close.shift(1)).dropna().tail(ventana)
    if len(retornos) < 2:
        return None
    valor = float(retornos.std(ddof=1) * np.sqrt(DIAS_ANIO))
    return valor if np.isfinite(valor) and valor > 0 else None


def analizar_volatilidad(df: pd.DataFrame) -> dict:
    """Volatilidad realizada a varias ventanas, índice de volatilidad y percentil.

    Args:
        df: DataFrame de ``analizar_precios``.

    Returns:
        Dict con ``hv`` ({ventana: fracción}), ``vol``, ``vol_percentil`` (0-100) y
        ``prima_vol`` (índice / HV30; < 1 significa que el mercado paga menos volatilidad
        de la que el activo se mueve). Los campos que no se puedan calcular salen None.
    """
    close = df["Close"]
    hv = {v: _volatilidad_realizada(close, v) for v in VENTANAS_HV}

    vol_serie = df["Vol_Close"].dropna()
    vol = float(vol_serie.iloc[-1]) if len(vol_serie) else None
    # El índice se publica en puntos de volatilidad (16.84 = 16.84 % anualizado),
    # así que se compara directamente con la HV (fracción → %).
    percentil = (
        float((vol_serie <= vol).mean() * 100) if vol is not None and len(vol_serie) else None
    )

    prima = None
    hv30 = hv.get(30)
    if vol is not None and hv30:
        prima = float((vol / 100) / hv30)

    return {"hv": hv, "vol": vol, "vol_percentil": percentil, "prima_vol": prima}


# ---------------------------------------------------------------------------
# Cadena de opciones: IV ATM, movimiento esperado y alas
# ---------------------------------------------------------------------------

def _precio_mid(fila: pd.Series) -> Optional[float]:
    """Precio medio de una opción: (bid+ask)/2, o lastPrice si no hay cotización.

    Args:
        fila: Fila de la cadena (columnas ``bid``, ``ask``, ``lastPrice``).

    Returns:
        El precio utilizable, o None si la opción no cotiza (sin bid, sin ask y
        sin último precio: no se inventa un cero).
    """
    bid, ask, ultimo = fila.get("bid"), fila.get("ask"), fila.get("lastPrice")
    if pd.notna(bid) and pd.notna(ask) and float(bid) > 0 and float(ask) > 0:
        return float((bid + ask) / 2)
    if pd.notna(ultimo) and float(ultimo) > 0:
        return float(ultimo)
    return None


def _iv_valida(iv: object) -> Optional[float]:
    """IV de Yahoo si entra en rango razonable, o None si es basura.

    Yahoo devuelve IV absurdas en strikes ilíctos (se han visto 240 %). Fuera de
    [1 %, 300 %] se considera inservible antes que mostrarla.
    """
    if iv is None or pd.isna(iv):
        return None
    valor = float(iv)
    return valor if _IV_MIN <= valor <= _IV_MAX else None


def _strike_cercano(strikes: pd.Series, objetivo: float) -> Optional[float]:
    """Strike real de la cadena más próximo a un precio objetivo."""
    if strikes is None or strikes.empty:
        return None
    idx = (strikes - objetivo).abs().idxmin()
    valor = float(strikes.loc[idx])
    return valor if np.isfinite(valor) else None


def _analizar_vencimiento(ticker: yf.Ticker, exp: str, etiqueta: str, spot: float,
                          atr: Optional[ATR] = None) -> Vencimiento:
    """Lee la cadena de un vencimiento y calcula sus bandas de movimiento.

    Args:
        ticker: Ticker de yfinance ya construido.
        exp: Fecha de expiración (ISO).
        etiqueta: ``0 DTE``, ``1 DTE`` o ``Semanal``.
        spot: Precio actual del subyacente.
        atr: ATR de referencia del ciclo al que pertenece este vencimiento.

    Returns:
        El ``Vencimiento`` con lo que se haya podido calcular; los campos sin
        dato fiable quedan a None (la card los pinta como N/A).
    """
    dte = int((pd.Timestamp(exp).normalize() - pd.Timestamp.today().normalize()).days)
    try:
        cadena = ticker.option_chain(exp)
    except Exception:
        return Vencimiento(etiqueta=etiqueta, fecha=exp, dte=dte, atr=atr)

    calls, puts = cadena.calls, cadena.puts
    if calls.empty or puts.empty:
        return Vencimiento(etiqueta=etiqueta, fecha=exp, dte=dte, atr=atr)

    # Strike ATM: el más cercano al dinero que cotice en call Y en put. Se exige
    # cotización en ambos porque el movimiento esperado sale del straddle.
    candidatos = []
    for strike in sorted(calls["strike"].unique(), key=lambda s: abs(float(s) - spot)):
        c = calls[calls["strike"] == strike]
        p = puts[puts["strike"] == strike]
        if c.empty or p.empty:
            continue
        mid_c, mid_p = _precio_mid(c.iloc[0]), _precio_mid(p.iloc[0])
        if mid_c is not None and mid_p is not None:
            candidatos.append((strike, c.iloc[0], p.iloc[0], mid_c, mid_p))
        if len(candidatos) >= 3:
            break  # con los 3 strikes más cercanos al dinero sobra

    if not candidatos:
        return Vencimiento(etiqueta=etiqueta, fecha=exp, dte=dte, atr=atr)

    strike_atm, fila_c, fila_p, mid_c, mid_p = candidatos[0]
    strike_atm = float(strike_atm)

    # IV ATM: media de call y put en el mismo strike (menos dependiente del
    # sesgo de un solo lado que leer una única pata).
    ivs = [iv for iv in (_iv_valida(fila_c.get("impliedVolatility")),
                         _iv_valida(fila_p.get("impliedVolatility"))) if iv is not None]
    iv = float(np.mean(ivs)) if ivs else None

    # Movimiento esperado: σ = straddle / 0.7979 (ver docstring del módulo).
    straddle = mid_c + mid_p
    sigma_puntos = float(straddle / _STRADDLE_A_SIGMA) if straddle > 0 else None
    sigma_pct = (sigma_puntos / spot) if sigma_puntos and spot else None

    banda1 = banda2 = ala1 = ala2 = banda_atr = None
    if sigma_puntos and sigma_puntos > 0:
        banda1 = (spot - sigma_puntos, spot + sigma_puntos)
        banda2 = (spot - 2 * sigma_puntos, spot + 2 * sigma_puntos)
        # Alas: strikes REALES de la cadena, los que se pueden teclear en la
        # orden. Si un extremo se sale de la cadena, ese lado queda a None.
        ala1 = (_strike_cercano(puts["strike"], banda1[0]),
                _strike_cercano(calls["strike"], banda1[1]))
        ala2 = (_strike_cercano(puts["strike"], banda2[0]),
                _strike_cercano(calls["strike"], banda2[1]))

    # Banda del ATR sobre el MISMO spot que las bandas σ, para que la única
    # diferencia entre las dos sea el estadístico (lo recorrido vs lo esperado)
    # y no el punto de referencia.
    if atr is not None and atr.valor is not None and np.isfinite(atr.valor) and atr.valor > 0:
        banda_atr = (spot - atr.valor, spot + atr.valor)

    return Vencimiento(
        etiqueta=etiqueta,
        fecha=exp,
        dte=dte,
        strike_atm=strike_atm,
        iv=iv,
        sigma_puntos=sigma_puntos,
        sigma_pct=sigma_pct,
        banda1=banda1,
        banda2=banda2,
        ala1=ala1,
        ala2=ala2,
        banda_atr=banda_atr,
        atr=atr,
    )


def analizar_opciones(s: Subyacente, spot: float, atr_diario: Optional[ATR] = None,
                      atr_diario_hoy: Optional[ATR] = None,
                      atr_semanal: Optional[ATR] = None) -> list[Vencimiento]:
    """Vencimientos 0 DTE, 1 DTE y semanal del subyacente con su movimiento esperado.

    Se eligen por criterio de calendario, no a ojo:
      - **0 DTE**: la expiración de HOY (el ciclo de entrada del día).
      - **1 DTE**: la de mañana.
      - **Semanal**: la primera expiración que cae en viernes.

    0 DTE y 1 DTE se buscan por **días exactos**, no por «el vencimiento más
    cercano»: una card que se llama 0 DTE tiene que ser 0 DTE de verdad, y
    enseñar bajo esa etiqueta un vencimiento a 2 días sería mentir en pantalla.
    Consecuencia asumida: en fin de semana no hay 0 DTE ni 1 DTE y esas cards no
    se pintan (queda solo la semanal), que es lo honesto.

    Args:
        s: Subyacente cuya cadena se lee.
        spot: Precio actual del subyacente (referencia del dinero y de las bandas).
        atr_diario: ATR diario cerrado —el de ayer al cierre—, el del ciclo de
            0 DTE.
        atr_diario_hoy: ATR diario con la sesión de HOY dentro, el del ciclo de
            1 DTE. Si no llega, el 1 DTE cae al cerrado antes que quedarse sin
            ATR (degradar antes que romper la card).
        atr_semanal: ATR semanal, el que se le da al vencimiento del viernes.

    Returns:
        Lista de ``Vencimiento`` (vacía si Yahoo no da la cadena).
    """
    try:
        ticker = yf.Ticker(s.ticker)
        expiraciones = list(ticker.options)
    except Exception:
        return []

    if not expiraciones:
        return []

    hoy = pd.Timestamp.today().normalize()
    vivas = [e for e in expiraciones if pd.Timestamp(e).normalize() >= hoy]
    if not vivas:
        return []

    # Un dict DTE -> fecha: buscar «el de 0 días» y «el de 1 día» por igualdad
    # deja el criterio a la vista y hace imposible colar un vencimiento que no
    # toca bajo una etiqueta que promete otra cosa. El valor es el texto ISO que
    # da Yahoo (no un ``pd.Timestamp``): al pintar una fecha de pandas sale con
    # hora —«2026-09-14 00:00:00»— y en la card queda sucia.
    por_dte: dict[int, str] = {
        (pd.Timestamp(e).normalize() - hoy).days: e for e in vivas
    }
    semanal = next(
        (e for e in vivas if pd.Timestamp(e).weekday() == 4 and pd.Timestamp(e).normalize() > hoy),
        None,
    )

    # Un vencimiento, una card: si dos criterios apuntan a la MISMA fecha (los
    # jueves, «mañana» ya es el viernes semanal), se queda con la primera
    # etiqueta de la lista en vez de repetir la misma fecha en dos cards. El ATR
    # viaja con cada ciclo: el 0 DTE mira el diario cerrado, el 1 DTE el diario
    # con la sesión de hoy dentro (es la posición que vive el día entero) y el
    # viernes el suyo.
    plan: list[tuple[str, str, Optional[ATR]]] = []
    for etiqueta, fecha, atr in (
        ("0 DTE", por_dte.get(0), atr_diario),
        ("1 DTE", por_dte.get(1),
         atr_diario_hoy if atr_diario_hoy is not None else atr_diario),
        ("Semanal", semanal, atr_semanal),
    ):
        if fecha is not None and fecha not in {f for _, f, _ in plan}:
            plan.append((etiqueta, fecha, atr))

    return [
        _analizar_vencimiento(ticker, exp, etiqueta, spot, atr)
        for etiqueta, exp, atr in plan
    ]


# ---------------------------------------------------------------------------
# Rangos por periodo (día, semana, mes) en %
# ---------------------------------------------------------------------------

def _partir_cerrados(marco: pd.Series | pd.DataFrame,
                     ultima_fecha: pd.Timestamp) -> pd.Series | pd.DataFrame:
    """Descarta el último periodo de la serie si todavía no ha cerrado.

    Un periodo en curso (el mes de hoy, la semana de hoy, la sesión de hoy) está
    incompleto y arrastraría la media hacia abajo. Se excluye hasta que cierre.

    Args:
        marco: Serie o DataFrame (OHLC) indexado por ``Period``.
        ultima_fecha: Última fecha con datos.

    Returns:
        El marco sin el periodo en curso (o tal cual si ya cerró).
    """
    if marco.empty:
        return marco
    # ``>=`` y no ``>``: el ``end_time`` de un periodo diario normalizado es la
    # propia fecha del día, así que con ``>`` el día en curso nunca se excluía.
    if marco.index[-1].end_time.normalize() >= ultima_fecha:
        return marco.iloc[:-1]
    return marco


def _atr_wilder(altos: pd.Series, bajos: pd.Series, cierres: pd.Series,
                ventana: int = VENTANA_ATR) -> pd.Series:
    """ATR de Wilder: el true range suavizado con una exponencial de α = 1/ventana.

    El true range de un periodo es el mayor de ``alto − bajo``, ``|alto − cierre
    previo|`` y ``|bajo − cierre previo|``. Los dos últimos son los que meten el
    HUECO de apertura dentro del rango, y el hueco es riesgo real de quien vende
    un condor: el subyacente abre donde quiere, no donde cerró.

    El arranque de ``ewm(adjust=False)`` es el primer true range y no la media de
    los primeros 14 de la definición clásica; con 5 años de historia el peso de
    ese arranque es despreciable, así que no se corrige.

    Args:
        altos: Máximos por periodo.
        bajos: Mínimos por periodo.
        cierres: Cierres por periodo.
        ventana: Periodos del suavizado.

    Returns:
        Serie del ATR, una observación más corta que la entrada: el primer
        periodo no tiene cierre previo y se descarta en vez de rellenarlo.
    """
    cierre_previo = cierres.shift(1)
    valido = cierre_previo.notna()
    if not valido.any():
        return pd.Series(dtype=float)

    altos, bajos, cierre_previo = altos[valido], bajos[valido], cierre_previo[valido]
    rango = pd.concat(
        [altos - bajos, (altos - cierre_previo).abs(), (bajos - cierre_previo).abs()],
        axis=1,
    ).max(axis=1)
    return rango.ewm(alpha=1 / ventana, adjust=False).mean()


def _atr_de(df: pd.DataFrame, freq: str, etiqueta: str,
            ultima_fecha: pd.Timestamp,
            incluir_en_curso: bool = False) -> Optional[ATR]:
    """ATR de un ciclo (día o semana) leído en su último periodo CERRADO.

    Args:
        df: DataFrame de ``analizar_precios``.
        freq: Agrupación de pandas (``D`` para diario, ``W-FRI`` para semanal).
        etiqueta: Nombre del ATR para la card.
        ultima_fecha: Última fecha con datos (descarta el periodo en curso).
        incluir_en_curso: True NO descarta el periodo en curso y calcula el ATR
            con la sesión de hoy dentro. Es lo que pide el ciclo de 1 DTE: un día
            volátil mueve la cifra mientras se forma, a costa de que la cifra se
            mueva. El ``pct`` se sigue tomando sobre el cierre del periodo, así
            que en la sesión en curso es el último precio, no un cierre.

    Returns:
        El ``ATR``, o None si no hay historia suficiente o el precio no es válido.
    """
    periodos = df.index.to_period(freq)
    ohlc = pd.DataFrame({
        "High": df["High"].groupby(periodos).max(),
        "Low": df["Low"].groupby(periodos).min(),
        "Close": df["Close"].groupby(periodos).last(),
        # Última SESIÓN real del periodo: en las semanas, el viernes; si el
        # viernes fue festivo, el jueves. Se guarda la fecha de cotización y no
        # el ``end_time`` del periodo, que en una semana festiva daría un
        # viernes sin sesión (el mismo detalle que en ``EstadisticaPeriodo``).
        # El nombre lo dice: no es un precio más del marco.
        "Fin_Periodo": df.index.to_series().groupby(periodos).max(),
    })
    marco = ohlc if incluir_en_curso else _partir_cerrados(ohlc, ultima_fecha)
    if marco.empty:
        return None

    serie = _atr_wilder(marco["High"], marco["Low"], marco["Close"])
    if serie.empty:
        return None

    valor = float(serie.iloc[-1])
    referencia = float(marco["Close"].iloc[-1])
    if not np.isfinite(valor) or referencia <= 0:
        return None

    fin = marco["Fin_Periodo"].iloc[-1]
    return ATR(
        valor=valor,
        # El % se toma sobre el cierre del propio periodo usado, nunca sobre el
        # precio de hoy: incluso el ATR en curso se mide contra el último precio
        # de su serie, para que la cifra y su % salgan del mismo sitio.
        pct=valor / referencia,
        desde=f"{fin:%Y-%m-%d}",
        etiqueta=etiqueta,
        # «Incluye hoy» solo si la última sesión del marco ES hoy: en festivo la
        # variante en curso coincide con la cerrada (las dos acaban el viernes) y
        # marcarla como «hoy» sería mentir en la fila de la fecha.
        incluye_hoy=incluir_en_curso and fin.normalize() == pd.Timestamp.today().normalize(),
    )


def analizar_atr(df: pd.DataFrame) -> dict[str, Optional[ATR]]:
    """ATR diario, diario-con-hoy y semanal, cada uno en su periodo de referencia.

    Args:
        df: DataFrame de ``analizar_precios``.

    Returns:
        ``{"diario": ATR|None, "diario_hoy": ATR|None, "semanal": ATR|None}``.
        El diario es el de ayer al cierre (el que se ve al entrar hoy); el
        ``diario_hoy`` mete la sesión de hoy dentro y es el que usa el ciclo de
        1 DTE; el semanal, el de la semana pasada.
    """
    ultima = df.index.max().normalize()
    return {
        "diario": _atr_de(df, "D", "ATR(14) diario", ultima),
        "diario_hoy": _atr_de(df, "D", "ATR(14) diario", ultima, incluir_en_curso=True),
        "semanal": _atr_de(df, "W-FRI", "ATR(14) semanal", ultima),
    }


def _rangos_por_periodo(df: pd.DataFrame, freq: str) -> pd.Series:
    """Serie de rangos % por periodo, normalizados por el cierre previo.

    El rango de cada periodo es ``(máximo − mínimo) / cierre anterior al inicio
    del periodo``: el movimiento contado desde donde se entra, que es lo que le
    importa a quien vende un condor (el gap cuenta como riesgo).

    Args:
        df: DataFrame de ``analizar_precios``.
        freq: Frecuencia de agrupación de pandas (``D``, ``W-FRI``, ``M``).

    Returns:
        Serie ``{Period: rango %}``; los periodos sin cierre previo en la
        historia (el primero) se descartan, no se rellenan.
    """
    periodos = df.index.to_period(freq)
    altos = df["High"].groupby(periodos).max()
    bajos = df["Low"].groupby(periodos).min()
    cierres = df["Close"]

    filas: dict[pd.Period, float] = {}
    for periodo in altos.index:
        previos = cierres[cierres.index < periodo.start_time]
        if previos.empty:
            continue  # sin referencia previa no hay % honesto que dar
        referencia = float(previos.iloc[-1])
        if referencia <= 0:
            continue
        rango = float(altos.loc[periodo] - bajos.loc[periodo]) / referencia * 100
        if np.isfinite(rango):
            filas[periodo] = rango

    return pd.Series(filas).sort_index()


def _excursiones_diarias(df: pd.DataFrame, ultima_fecha: pd.Timestamp) -> pd.DataFrame:
    """Recorrido de cada sesión partido en lo que fue hacia arriba y hacia abajo.

    Mismo criterio que ``_rangos_por_periodo`` (cifras en % del cierre anterior a
    la sesión, y solo sesiones cerradas), pero separando los dos lados. El total
    de una sesión no dice cuánto se fue en cada dirección, y a quien vende un
    condor no le da igual: las dos alas no se defienden igual y el movimiento de
    una sesión se reparte entre las dos de forma muy desigual.

    Args:
        df: DataFrame de ``analizar_precios``.
        ultima_fecha: Última fecha con datos (descarta la sesión en curso).

    Returns:
        DataFrame indexado por periodo diario con tres columnas en %:
        ``arriba`` (máximo − cierre previo), ``abajo`` (cierre previo − mínimo) y
        ``total`` (máximo − mínimo). ``total`` es la suma exacta de las otras
        dos, así que la tabla no puede contradecirse consigo misma.
    """
    periodos = df.index.to_period("D")
    tabla = pd.DataFrame({
        "alto": df["High"].groupby(periodos).max(),
        "bajo": df["Low"].groupby(periodos).min(),
        "previo": df["Close"].shift(1).groupby(periodos).last(),
    }).dropna()
    tabla = tabla[tabla["previo"] > 0]  # sin referencia previa no hay % honesto
    tabla = _partir_cerrados(tabla, ultima_fecha)
    if tabla.empty:
        return tabla

    tabla["arriba"] = (tabla["alto"] - tabla["previo"]) / tabla["previo"] * 100
    tabla["abajo"] = (tabla["previo"] - tabla["bajo"]) / tabla["previo"] * 100
    tabla["total"] = tabla["arriba"] + tabla["abajo"]
    return tabla[["arriba", "abajo", "total"]]


def _estadistica(etiqueta: str, serie: pd.Series, fin_por_periodo: pd.Series) -> EstadisticaPeriodo:
    """Resume la serie de rangos de un periodo en media, mediana y último.

    Args:
        etiqueta: ``Día``, ``Semana`` o ``Mes``.
        serie: Serie de rangos % ya filtrada a periodos cerrados.
        fin_por_periodo: ``{Period: última fecha de sesión del periodo}``.

    Returns:
        La ``EstadisticaPeriodo``; sin datos, todos los valores a None y n=0.
    """
    if serie.empty:
        return EstadisticaPeriodo(etiqueta, None, None, None, None, 0)

    ultimo_periodo = serie.index[-1]
    fin = fin_por_periodo.get(ultimo_periodo)

    return EstadisticaPeriodo(
        etiqueta=etiqueta,
        media_pct=float(serie.mean()),
        mediana_pct=float(serie.median()),
        ultimo_pct=float(serie.iloc[-1]),
        ultimo_fin=fin.strftime("%Y-%m-%d") if fin is not None else None,
        n=int(len(serie)),
    )


def analizar_rangos(df: pd.DataFrame) -> dict:
    """Rangos medios en % por día, semana y mes, con sus desgloses.

    Args:
        df: DataFrame de ``analizar_precios``.

    Returns:
        Dict con ``periodos`` (lista de ``EstadisticaPeriodo``), ``por_dia_semana``
        (lunes a viernes, con las dos excursiones y su suma) y ``por_dia_mes``
        (ordenado de mayor a menor rango).
    """
    ultima = df.index.max().normalize()

    def fin_de(freq: str) -> pd.Series:
        """Última fecha de sesión de cada periodo de la frecuencia dada."""
        return df.index.to_series().groupby(df.index.to_period(freq)).max()

    fin_d, fin_w, fin_m = fin_de("D"), fin_de("W-FRI"), fin_de("M")
    diarios = _partir_cerrados(_rangos_por_periodo(df, "D"), ultima)
    semanales = _partir_cerrados(_rangos_por_periodo(df, "W-FRI"), ultima)
    mensuales = _partir_cerrados(_rangos_por_periodo(df, "M"), ultima)

    # Desglose por día de la semana, partido en los dos lados: media de lo que
    # fue hacia arriba, media de lo que fue hacia abajo y su suma. Las tres
    # columnas suman exacto porque son medias de cifras que suman y la media es
    # lineal; con medianas no sumarían y habría que advertirlo en la card.
    por_dow: list[tuple[str, float, float, float]] = []
    por_dia_mes: list[tuple[int, float]] = []
    excursiones = _excursiones_diarias(df, ultima)
    if not excursiones.empty:
        tabla_dow = excursiones.groupby(excursiones.index.dayofweek).mean()
        por_dow = [
            (
                _NOMBRES_DOW[dow],
                float(tabla_dow.loc[dow, "arriba"]),
                float(tabla_dow.loc[dow, "abajo"]),
                float(tabla_dow.loc[dow, "total"]),
            )
            for dow in sorted(_NOMBRES_DOW)
            if dow in tabla_dow.index
        ]
    if not diarios.empty:
        tabla_mes = diarios.groupby(diarios.index.day).mean()
        por_dia_mes = sorted(
            ((int(dia), float(valor)) for dia, valor in tabla_mes.items()),
            key=lambda par: par[1],
            reverse=True,
        )

    return {
        "periodos": [
            _estadistica("Día", diarios, fin_d),
            _estadistica("Semana", semanales, fin_w),
            _estadistica("Mes", mensuales, fin_m),
        ],
        "por_dia_semana": por_dow,
        "por_dia_mes": por_dia_mes,
    }


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def _en_horario_regular() -> bool:
    """True si el mercado de EE.UU. está en su sesión regular (9:30-16:00 ET, L-V).

    Los festivos no se contemplan, y da igual: en un festivo no hay sesión, pero tampoco
    precio nuevo, así que el último cierre sigue siendo la respuesta correcta.
    """
    ahora = pd.Timestamp.now(tz=ZONA_MERCADO)
    if ahora.weekday() >= 5:
        return False
    return APERTURA_MERCADO <= ahora.time() <= CIERRE_MERCADO


def _spot(s: Subyacente, precio_cierre: float) -> tuple[float, str, Optional[str]]:
    """Precio del subyacente para las bandas: spot en vivo si se puede, si no el cierre.

    Args:
        s: Subyacente del que se quiere el precio.
        precio_cierre: Último cierre del histórico.

    Returns:
        ``(precio, texto_de_fecha, nota)``: la nota avisa cuando se usa el cierre en lugar
        del spot (mercado cerrado, fuera de horario en un ETF, o Yahoo sin respuesta).
    """
    # Un ETF cotiza también fuera del horario regular, así que su «último precio» puede
    # ser de after-hours mientras la cadena de opciones sigue cotizada contra el cierre de
    # la sesión. Las bandas van contra el precio con el que la cadena está de acuerdo, que
    # es ese cierre. Un índice no tiene el problema: no cotiza fuera de hora, su último
    # precio ya es el de la sesión.
    if s.cotiza_extendido and not _en_horario_regular():
        return precio_cierre, "último cierre", (
            f"Fuera del horario regular de EE.UU.: las bandas van sobre el último cierre "
            f"de {s.etiqueta}, que es contra el que cotiza la cadena de opciones "
            "(el ETF sí sigue cotizando fuera de hora)."
        )

    try:
        vivo = yf.Ticker(s.ticker).fast_info.get("lastPrice")
        if vivo is not None and float(vivo) > 0:
            return float(vivo), f"spot {pd.Timestamp.now():%Y-%m-%d %H:%M}", None
    except Exception:
        pass
    return precio_cierre, "último cierre", "Spot no disponible: bandas sobre el último cierre."


def analizar_condor(s: Subyacente, refresh: bool = False) -> AnalisisCondor:
    """Análisis completo de un subyacente para su pestaña: vol, opciones y rangos.

    Args:
        s: Subyacente a analizar.
        refresh: True fuerza la re-descarga de precios (ignora la caché).

    Returns:
        El ``AnalisisCondor``; si no hay precios, un resultado con ``precio=None``
        y la nota del fallo (la card se pinta como sin datos, sin inventar nada).
    """
    df = analizar_precios(s, refresh=refresh)
    if df is None or df.empty:
        return AnalisisCondor(
            subyacente=s,
            notas=[f"Sin datos de {s.etiqueta}: Yahoo no responde y no hay caché utilizable."],
        )

    precio_cierre = float(df["Close"].iloc[-1])
    spot, texto_fecha, nota_spot = _spot(s, precio_cierre)

    vol = analizar_volatilidad(df)
    rangos = analizar_rangos(df)
    atr = analizar_atr(df)
    vencimientos = analizar_opciones(s, spot, atr["diario"], atr["diario_hoy"], atr["semanal"])

    notas = [nota_spot] if nota_spot else []
    if not vencimientos:
        notas.append("Cadena de opciones no disponible: sin IV ni bandas de vencimiento.")
    elif all(v.iv is None for v in vencimientos):
        notas.append("La cadena respondió pero sin IV utilizable (strikes sin cotizar).")
    if df.index.max().normalize() < pd.Timestamp.today().normalize():
        notas.append(f"Último dato de precio: {df.index.max():%Y-%m-%d} (mercado cerrado).")
    # Honestidad sobre el percentil: la IV de los vencimientos cortos es una foto
    # del momento (Yahoo no da histórico de cadena), así que no tiene percentil.
    notas.append(
        f"El percentil se calcula sobre el {s.nombre_vol}: la IV de la cadena no tiene histórico."
    )

    return AnalisisCondor(
        subyacente=s,
        precio=spot,
        fecha_precio=texto_fecha,
        hv=vol["hv"],
        vol=vol["vol"],
        vol_percentil=vol["vol_percentil"],
        prima_vol=vol["prima_vol"],
        vencimientos=vencimientos,
        periodos=rangos["periodos"],
        por_dia_semana=rangos["por_dia_semana"],
        por_dia_mes=rangos["por_dia_mes"],
        notas=notas,
    )
