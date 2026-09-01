"""Cards HTML de los tramos del Tesoro de EE.UU. (5 cards, datos FRED/DGS).

Una card por tramo (3M, 2Y, 5Y, 10Y, 30Y) con las métricas del tramo. El precio
que se muestra es TEÓRICO: derivado del yield diario (FRED/DGS) valorando un
bono hipotético con cupón fijo y vencimiento constante (convención
constant-maturity estándar, la misma que usan los índices de renta fija). No es
un precio de mercado; es lo que valdría el bono a un vencimiento fijo.

Interpretación de cada fila (honesta, nada inventado):
  - ``Precio``: precio teórico del bono hipotético (par 100 el día de emisión).
  - ``Yield``: nivel actual del tramo (FRED/DGS).
  - ``Retorno YTD/1M/1Y``, ``Vol 30D/1Y``, ``Drawdown``: calculados sobre la
    serie de PRECIOS teórica — lo que realmente gana o pierde quien tiene el
    bono (subida de yield = caída de precio), mismo motor ``metricas_market``
    que el resto del dashboard.
  - ``Δ Yield 1M/3M``: variación absoluta del nivel en puntos básicos.
  - ``Spread vs 2Y``: yield(tramo) − yield(2Y) (la card 2Y usa 2Y−3M, el frente
    de la curva). ``Δ Curve 1M``: variación de ese spread en el último mes.
  - ``Carry (1Y)`` y ``Roll-down (1Y)``: "lo que ganas por esperar" — carry =
    yield − 3M (proxy del repo, el coste de financiación); roll-down = pendiente
    hacia el tramo adyacente más corto × duration (aprox. lineal, curva
    estática). El 3M no tiene tramo más corto → N/A.
  - ``Duration`` y ``DV01 ($/1M)``: las métricas de mesa de bonos
    (sensibilidad por punto básico), calculadas sobre el bono hipotético.
Cualquier dato ausente → N/A honesto.
"""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")  # sin ventana; solo exportar PNG

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from macro_catalogo import BONOS_TRAMOS
from macro_metrics import metricas_market
from macro_panel_html import (  # reutiliza formateadores del tema FX
    _color_retorno,
    _fila,
    _fila_retorno,
    _GRIS,
    fmt_pct_frac,
    fmt_precio,
)

# Paleta del tema FX (la misma que las cards macro).
_VERDE = "#3fb950"
_NARANJA = "#ff9f43"

_MES_ATRAS = 21  # días hábiles ≈ 1 mes
_TRIM_ATRAS = 63  # días hábiles ≈ 3 meses


def _color_spread(valor: Optional[float]) -> str:
    """Color del spread: verde si la curva es normal, naranja si invertida."""
    if valor is None:
        return _GRIS
    return _VERDE if valor >= 0 else _NARANJA


def _fmt_pb(valor: Optional[float]) -> str:
    """Variación en puntos básicos con signo (p.ej. ``+12.0 pb``)."""
    if valor is None or np.isnan(valor):
        return "N/A"
    return f"{valor:+.1f} pb"


def _precio_serie(serie: pd.Series, venc_anos: float) -> pd.Series:
    """Serie de precios teórica del tramo, derivada del yield (FRED/DGS).

    Se valora un bono hipotético con cupón SEMESTRAL fijo (igual al yield del
    primer día, emitido a par = 100) y vencimiento constante ``venc_anos``,
    revalorado cada día con el yield del día. Tramos de < 1 año (TB3M) se
    valoran como T-bill cero-cupón. Es la convención constant-maturity
    estándar de los índices de renta fija: no inventa precios de mercado,
    reconstruye el precio que la curva de rendimientos implica.

    Args:
        serie: Serie diaria del yield del tramo en % (0.0473 = 4.73 %).
        venc_anos: Vencimiento del bono hipotético en años.

    Returns:
        Serie de precios (nominal 100 = par), indexada igual que ``serie``.
    """
    # FRED entrega el yield en % (3.90 = 3.90 %): para valorar, la fracción es
    # y/100. El cupón de emisión también se fija en fracción del nominal.
    y0 = float(max(serie.iloc[0], 0.0)) / 100.0

    if venc_anos < 1:
        # T-bill cero-cupón: precio = 100 / (1 + y·T) (bond-equivalent yield).
        return serie.apply(lambda y: 100.0 / (1.0 + max(y, 0.0) / 100.0 * venc_anos))

    cupon_sem = y0 / 2.0 * 100.0  # cupón semestral fijo por 100 nominal (a par)
    periodos = np.arange(1, int(2 * venc_anos) + 1)
    ultimo = periodos[-1]

    def _precio(y: float) -> float:
        y_sem = max(y, 0.0) / 100.0 / 2.0
        cupones = float(np.sum(cupon_sem / (1.0 + y_sem) ** periodos))
        principal = 100.0 / (1.0 + y_sem) ** ultimo
        return cupones + principal

    return serie.apply(_precio)


def _duration_dv01(serie: pd.Series, venc_anos: float) -> tuple[Optional[float], Optional[float]]:
    """Duration modificada y DV01 del bono hipotético al yield actual.

    Métricas de mesa de bonos:
      - Duration modificada (años): sensibilidad del precio a variaciones del
        yield, ~(Σ t·PV/price) / (1+y/2) para flujos semestrales.
      - DV01: cambio de precio ante +1 pb (exacto, revalorando a y+1pb),
        expresado en dólares por $1M de nominal (convención de las mesas:
        "un 1M de 10Y mueve ~$84 por punto básico").

    Args:
        serie: Serie diaria del yield del tramo en % (0.0473 = 4.73 %).
        venc_anos: Vencimiento del bono hipotético en años.

    Returns:
        Tupla ``(duration_modificada, dv01_por_1M)`` con el yield actual, o
        ``(None, None)`` si no se puede valorar.
    """
    y = float(serie.iloc[-1]) / 100.0  # fracción (FRED entrega %)
    y_sem = y / 2.0

    def _precio_tbill(yy: float) -> float:
        return 100.0 / (1.0 + yy * venc_anos)

    if venc_anos < 1:
        # T-bill cero-cupón: Macaulay ≈ T; DV01 exacto.
        p = _precio_tbill(y)
        p_up = _precio_tbill(y + 0.0001)
        # Magnitud positiva (convención de las mesas): un +1 pb de yield
        # mueve el precio |p_up − p|; el signo (pérdida para el largo) es
        # implícito en la relación yield/precio.
        dv01 = abs((p_up - p) * 10000)  # $ por $1M de nominal
        mod = venc_anos / (1.0 + y * venc_anos)
        return mod, dv01

    cupon_sem = float(serie.iloc[0]) / 100.0 / 2.0 * 100.0  # cupón semestral por 100
    periodos = np.arange(1, int(2 * venc_anos) + 1)
    ultimo = periodos[-1]

    def _flujos(yy: float) -> tuple[np.ndarray, float]:
        ys = yy / 2.0
        cupones = cupon_sem / (1.0 + ys) ** periodos
        principal = (cupon_sem + 100.0) / (1.0 + ys) ** ultimo
        return cupones, principal

    def _precio(yy: float) -> float:
        cupones, principal = _flujos(yy)
        return float(np.sum(cupones) + principal)

    try:
        p = _precio(y)
        if p <= 0 or not np.isfinite(p):
            return None, None
        p_up = _precio(y + 0.0001)
        dv01 = abs((p_up - p) * 10000)  # $ por $1M de nominal (magnitud)

        # Macaulay = Σ(t·PV) / precio, con t en años (semestres/2).
        cupones, principal = _flujos(y)
        pvs = np.append(cupones, principal)
        tiempos = np.append(periodos / 2.0, venc_anos)
        macaulay = float(np.sum(tiempos * pvs) / p)
        mod = macaulay / (1.0 + y_sem)
        return mod, dv01
    except (ZeroDivisionError, OverflowError):
        return None, None


def _adyacente_corto(tramo: str) -> Optional[str]:
    """Tramo inmediatamente más corto de la curva (para el roll-down)."""
    orden = sorted(BONOS_TRAMOS, key=lambda k: float(BONOS_TRAMOS[k]["venc_anos"]))
    if tramo not in orden or orden.index(tramo) == 0:
        return None
    return orden[orden.index(tramo) - 1]


def _card_tramo(df: pd.DataFrame, tramo: str, spec: dict) -> str:
    """Card HTML de un tramo de la curva.

    Args:
        df: Curva completa (``bonos_motores.analizar_curva``), columnas = tramos.
        tramo: Clave del tramo (``TB10Y``).
        spec: Especificación del tramo en ``BONOS_TRAMOS`` (label, series).

    Returns:
        String HTML de la card del tramo.
    """
    serie = df[tramo].dropna()
    if serie.empty:
        return (
            '<div class="macro-card" id="m-%s">'
            "<h3>%s</h3>"
            '<div class="sub">%s · sin datos disponibles</div>'
            "</div>" % (tramo, spec["label"], spec["series"])
        )

    yield_actual = float(serie.iloc[-1])
    venc_anos = float(spec["venc_anos"])

    # Precio teórico derivado del yield: los retornos/vol/drawdown se calculan
    # sobre ESTE precio (lo que gana o pierde el que tiene el bono), no sobre
    # el yield (una subida de yield es una pérdida de precio, no una ganancia).
    precios = _precio_serie(serie, venc_anos)
    precio_actual = float(precios.iloc[-1])
    cupon_anual = float(serie.iloc[0])  # cupón fijo de emisión (bono a par)
    market = metricas_market(precios, benchmark=None, dias_anio=252)
    duration, dv01 = _duration_dv01(serie, venc_anos)

    # Retorno YTD (mismo criterio que el panel macro).
    primer_anio = serie.index[-1].year
    ytd_serie = precios[precios.index.year == primer_anio]
    retorno_ytd = (
        float(precios.iloc[-1] / ytd_serie.iloc[0] - 1)
        if not ytd_serie.empty
        else None
    )

    # Δ Yield 1M / 3M en pb.
    delta_1m = delta_3m = None
    if len(serie) > _MES_ATRAS and np.isfinite(serie.iloc[-1 - _MES_ATRAS]):
        delta_1m = float((serie.iloc[-1] - serie.iloc[-1 - _MES_ATRAS]) * 100)
    if len(serie) > _TRIM_ATRAS and np.isfinite(serie.iloc[-1 - _TRIM_ATRAS]):
        delta_3m = float((serie.iloc[-1] - serie.iloc[-1 - _TRIM_ATRAS]) * 100)

    # Spread y su variación mensual. Contra la 2Y para todos los tramos menos
    # la propia 2Y (ahí el spread sería 0 por definición): la 2Y usa el frente
    # de la curva (2Y − 3M), que es la medida estándar del tramo corto.
    spread = None
    delta_spread = None
    ancla = "TB2Y"
    par = "TB3M" if tramo == ancla else ancla
    etiqueta_spread = "Spread 2Y–3M" if tramo == ancla else "Spread vs 2Y"
    if par in df.columns:
        par_serie = df[par].dropna()
        comunes = pd.concat([serie, par_serie], axis=1, join="inner").dropna()
        if len(comunes):
            spread = float((comunes.iloc[-1, 0] - comunes.iloc[-1, 1]) * 100)
            if len(comunes) > _MES_ATRAS:
                prev = comunes.iloc[-1 - _MES_ATRAS]
                if np.isfinite(prev.iloc[0]) and np.isfinite(prev.iloc[1]):
                    delta_spread = float(
                        ((comunes.iloc[-1, 0] - comunes.iloc[-1, 1])
                         - (prev.iloc[0] - prev.iloc[1])) * 100
                    )

    # Carry y roll-down (métricas de mesa de "cuánto ganas por esperar").
    # Carry = yield − coste de financiación a corto (el 3M como proxy del repo,
    # lo más honesto con los datos DGS disponibles). Roll-down = pendiente de la
    # curva hacia el tramo adyacente más corto × duration (aprox. lineal, curva
    # estática): lo que el bono se aprecia al rodar hacia un vencimiento menor.
    carry = None
    roll = None
    corto = "TB3M"
    if tramo != corto and corto in df.columns:
        corto_serie = df[corto].dropna()
        comunes = pd.concat([serie, corto_serie], axis=1, join="inner").dropna()
        if len(comunes):
            carry = float((comunes.iloc[-1, 0] - comunes.iloc[-1, 1]) * 100)  # pb

    adyacente = _adyacente_corto(tramo)
    if adyacente and adyacente in df.columns and duration is not None:
        adj_serie = df[adyacente].dropna()
        comunes = pd.concat([serie, adj_serie], axis=1, join="inner").dropna()
        if len(comunes):
            pendiente = (comunes.iloc[-1, 0] - comunes.iloc[-1, 1])  # en % del yield
            roll = float(pendiente * duration * 100)  # pb/año (aprox. lineal)

    filas = [
        _fila("Precio", fmt_precio(precio_actual)),
        _fila("Yield", f'<span style="color:#c9d1d9">{yield_actual:.2f}%</span>'),
        _fila_retorno("Retorno YTD", retorno_ytd),
        _fila_retorno("Retorno 1M", market.return_1m),
        _fila_retorno("Retorno 1Y", market.return_1y),
        _fila("Vol 30D", fmt_pct_frac(market.volatility_30d)),
        _fila("Vol 1Y", fmt_pct_frac(market.volatility_1y)),
        _fila("Drawdown", fmt_pct_frac(market.max_drawdown, signo=True)),
        _fila("Δ Yield 1M", f'<span{_color_retorno(delta_1m)}>{_fmt_pb(delta_1m)}</span>'),
        _fila("Δ Yield 3M", f'<span{_color_retorno(delta_3m)}>{_fmt_pb(delta_3m)}</span>'),
        _fila(etiqueta_spread, f'<span style="color:{_color_spread(spread)}">{_fmt_pb(spread)}</span>'),
        _fila("Δ Curve 1M", _fmt_pb(delta_spread)),
        _fila("Carry (1Y)", f'<span{_color_retorno(carry)}>{_fmt_pb(carry)}</span>'),
        _fila("Roll-down (1Y)", f'<span{_color_retorno(roll)}>{_fmt_pb(roll)}</span>'),
        _fila("Duration", "N/A" if duration is None else f"{duration:.1f} a"),
        _fila("DV01 ($/1M)", "N/A" if dv01 is None else f"${dv01:.1f}"),
    ]

    return f"""
    <div class="macro-card" id="m-{tramo}">
        <div class="macro-head">
            <h3>{spec["label"]} <span class="ticker">({spec["series"]})</span></h3>
            <div class="sub">Datos del Tesoro (FRED) · precio teórico, cupón fijo {cupon_anual:.2f}%</div>
        </div>
        <table class="macro-table">
            {"\n".join(filas)}
        </table>
    </div>
    """


def panel_curva(df: Optional[pd.DataFrame]) -> str:
    """Grid con las cards de los 5 tramos del Tesoro (o card honesta de sin-datos).

    Args:
        df: DataFrame de ``bonos_motores.analizar_curva`` (5 tramos en %,
            indexado por fecha), o None si no hay datos.

    Returns:
        String HTML con las 5 cards dentro del grid macro.
    """
    if df is None or df.empty:
        return (
            '<div class="macro-card" id="m-curva-tesoro">'
            "<h3>Curva del Tesoro EE.UU.</h3>"
            '<div class="sub">DGS* FRED · sin datos disponibles</div>'
            "</div>"
        )

    cards = "".join(
        _card_tramo(df, tramo, spec) for tramo, spec in BONOS_TRAMOS.items()
    )
    return cards
