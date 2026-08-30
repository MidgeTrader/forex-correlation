"""Panel HTML por activo macro, con las métricas de panel.

Convierte el resultado de los 6 motores (``macro_motores.analizar_activo``)
en una tarjeta HTML autocontenida del tema del dashboard FX (fondo oscuro
``#161b22``, acentos ``#58a6ff``/``#3fb950``/``#f85149``) con:
  - cabecera: etiqueta, ticker yfinance, precio actual y retorno 1D
  - tabla de métricas (precio, retornos, volatilidad, ratios, riesgo,
    probabilidades bootstrap y escenarios Monte Carlo)
  - dos gráficos matplotlib en base64: evolución de precio y retorno acumulado

UNIDADES — crítico, cada fuente devuelve en una escala distinta:
  - ``ProbabilityMetrics``: ``prob_*``, ``expected_*``, ``return_p*`` ya en % (0-100).
  - ``MarketMetrics``, ``ReturnsMetrics``, ``StatisticsMetrics``,
    ``TimeSeriesMetrics`` y los estocásticos de ``ModelsMetrics`` (``mc_*``,
    ``scenario_*``): fracciones (×100 para %).
Cualquier dato ausente se muestra como ``N/A`` honesto — nunca se inventa.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # sin ventana; solo exportar PNG

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from macro_catalogo import BENCHMARK_LABELS, carry_de_par  # noqa: E402
from data_fred import datos_inflacion, datos_tipos, fecha_tipos  # noqa: E402

plt.style.use("dark_background")

# Paleta del tema FX (misma que usa forex_to_html.py).
_VERDE = "#3fb950"
_ROJO = "#f85149"
_AZUL = "#58a6ff"
_GRIS = "#8b949e"


# ---------------------------------------------------------------------------
# Formateo (unidades)
# ---------------------------------------------------------------------------

def _es_na(valor: object) -> bool:
    """True si el valor es None o NaN (float/int)."""
    return valor is None or (isinstance(valor, (int, float)) and valor != valor)


def fmt_precio(valor: Optional[float]) -> str:
    """Precio con miles y 2 decimales, o N/A."""
    return "N/A" if _es_na(valor) else f"{valor:,.2f}"


def fmt_pct_frac(valor: Optional[float], signo: bool = False) -> str:
    """Fracción → porcentaje (×100). ``signo`` antepone +/−."""
    if _es_na(valor):
        return "N/A"
    pct = valor * 100
    if signo:
        return f"{pct:+.1f}%"
    return f"{pct:.1f}%"


def fmt_pct_100(valor: Optional[float], signo: bool = False) -> str:
    """Valor que ya es porcentaje 0-100 (prob_*, return_p*, momentum)."""
    if _es_na(valor):
        return "N/A"
    if signo:
        return f"{valor:+.1f}%"
    return f"{valor:.1f}%"


def fmt_num(valor: Optional[float], decimales: int = 2) -> str:
    """Número sin unidad (ratio, correlación, beta, p-value)."""
    if _es_na(valor):
        return "N/A"
    return f"{valor:.{decimales}f}"


def fmt_var_diaria(valor: Optional[float]) -> str:
    """Varianza diaria en % con 3 decimales (fracción muy pequeña), o N/A."""
    if _es_na(valor):
        return "N/A"
    return f"{valor * 100:.3f}%"


def fmt_dias(valor: Optional[float]) -> str:
    """Duración en días (entero) o N/A."""
    if _es_na(valor):
        return "N/A"
    return f"{int(round(valor))}d"


def _color_retorno(valor: Optional[float]) -> str:
    """Clase CSS de color para una fracción de retorno: pos/neg/neutral."""
    if _es_na(valor):
        return ""
    if valor > 0:
        return f' style="color:{_VERDE}"'
    if valor < 0:
        return f' style="color:{_ROJO}"'
    return ""


# ---------------------------------------------------------------------------
# Gráficos base64 (estilo oscuro del tema FX)
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    """Serializa la figura matplotlib a data-URI PNG en base64."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _estilo_grafico(ax) -> None:
    """Estilo limpio del dashboard: sin spines ni ticks, fondo del tema."""
    ax.set_facecolor("#161b22")
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _grafico_precio(price_data: pd.DataFrame) -> str:
    """Evolución del cierre (línea + relleno degradado), en base64."""
    closes = price_data["Close"].dropna()
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    _estilo_grafico(ax)
    ax.plot(closes.index, closes.values, color=_AZUL, linewidth=1.2)
    ax.fill_between(closes.index, closes.values, closes.values.min(),
                    color=_AZUL, alpha=0.12)
    ax.set_title("Evolución de precio", color=_GRIS, fontsize=8,
                 loc="left", pad=4)
    fig.patch.set_facecolor("#161b22")
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Construcción del panel
# ---------------------------------------------------------------------------

def _fila(etiqueta: str, valor_html: str) -> str:
    """Fila de la tabla de métricas: etiqueta + valor."""
    return f'<tr><td style="color:{_GRIS}">{etiqueta}</td><td>{valor_html}</td></tr>'


def _fila_retorno(etiqueta: str, fraccion: Optional[float]) -> str:
    """Fila de retorno en % con color según signo."""
    return _fila(etiqueta, f'<span{_color_retorno(fraccion)}>{fmt_pct_frac(fraccion, signo=True)}</span>')


def _fila_prob(etiqueta: str, pct_100: Optional[float], alto_es_riesgo: bool) -> str:
    """Fila de probabilidad (%) coloreada por umbral 50."""
    if _es_na(pct_100):
        return _fila(etiqueta, "N/A")
    color = _VERDE if (pct_100 >= 50) != alto_es_riesgo else _ROJO
    return _fila(etiqueta, f'<span style="color:{color}">{pct_100:.1f}%</span>')


def _tail_ratio(retornos: pd.Series) -> Optional[float]:
    """Tail ratio: percentil 95 de retornos ÷ |percentil 5|.

    ``P95 / abs(P5)``. >1 cola derecha más gruesa en magnitud (distribución
    favorable), <1 cola izquierda más pesada (riesgo de caídas extremas).
    N/A si el percentil 5 es 0.
    """
    if retornos.empty:
        return None
    p95 = retornos.quantile(0.95)
    p05 = retornos.quantile(0.05)
    if _es_na(p95) or _es_na(p05) or p05 == 0:
        return None
    return float(p95 / abs(p05))


def panel_activo(key: str, data, results: Optional[dict]) -> str:
    """Tarjeta HTML completa de un activo macro.

    Args:
        key: Clave del activo (p.ej. ``GOLD``) — identifica los IDs propios.
        data: ``MacroDataYF`` ya con ``fetch()`` ejecutado.
        results: Dict de métricas de los motores (None si no hay datos).

    Returns:
        String HTML de la tarjeta del activo (o tarjeta "sin datos" honesta).
    """
    if results is None:
        return (
            '<div class="macro-card" id="m-%s">'
            "<h3>%s</h3>"
            '<div class="sub">%s · sin datos disponibles</div>'
            "</div>" % (key, data.label, data.ticker_yf)
        )

    market = results["market"]
    returns = results["returns"]
    probability = results["probability"]
    timeseries = results["timeseries"]
    statistics = results["statistics"]

    closes = data.price_data["Close"].dropna()

    # Retornos que los motores no calculan: 1D y YTD (2 líneas).
    # Guardas de longitud: con series cortas o a 1 de enero no hay dato → N/A.
    retorno_1d = closes.iloc[-1] / closes.iloc[-2] - 1 if len(closes) >= 2 else None
    primer_anio = closes.index[-1].year if len(closes) else None
    ytd_serie = closes[closes.index.year == primer_anio] if primer_anio is not None else pd.Series(dtype=float)
    retorno_ytd = closes.iloc[-1] / ytd_serie.iloc[0] - 1 if not ytd_serie.empty else None

    precio_html = fmt_precio(market.current_price)
    ret1d_html = (
        f'<span{_color_retorno(retorno_1d)}>{fmt_pct_frac(retorno_1d, signo=True)}</span>'
    )

    benchmark = BENCHMARK_LABELS.get(data.benchmark_yf, data.benchmark_yf)

    # En los pares FX, la fila "Drawdown máximo" se sustituye por el Carry
    # (diferencial de tipos base − cotizada, tipos actuales de FRED vía
    # data_fred; fallo seguro a los últimos verificados). Los activos macro
    # (metales, índices...) y las criptomonedas no tienen carry: conservan el
    # drawdown máximo.
    if data.categoria == "fx":
        tipos_fx, _ = datos_tipos()
        carry = carry_de_par(key, tipos=tipos_fx)
    else:
        carry = None
    if data.categoria == "fx" and carry is not None:
        fila_carry = _fila(
            f"Carry ({fecha_tipos()})",
            f'<span{_color_retorno(carry)}>{fmt_pct_100(carry, signo=True)}</span>',
        )
    else:
        fila_carry = _fila("Drawdown máximo", fmt_pct_frac(market.max_drawdown, signo=True))

    # En los pares FX y las criptomonedas se sustituyen métricas de las cards:
    # "Retorno acumulado" → media de 50 semanas + canales 1σ/2σ, y
    # Sharpe/Sortino → forma de la distribución (Skewness, Kurtosis exceso,
    # Tail Ratio). Los activos macro conservan sus filas originales.
    if data.categoria in ("fx", "crypto"):
        # 50 semanas × días de negociación por semana (5 en 252, 7 en 365).
        ventana_50s = closes.tail(50 * round(data.dias_anio / 52))
        media_50s = ventana_50s.mean()
        sigma_50s = ventana_50s.std()
        filas_reemplazo = [
            _fila("Media 50 sem", fmt_precio(media_50s)),
            _fila("1σ (50 sem)",
                  f"{fmt_precio(media_50s + sigma_50s)} - {fmt_precio(media_50s - sigma_50s)}"),
            _fila("2σ (50 sem)",
                  f"{fmt_precio(media_50s + 2 * sigma_50s)} - {fmt_precio(media_50s - 2 * sigma_50s)}"),
        ]
        filas_forma = [
            _fila("Varianza diaria", fmt_var_diaria(statistics.variance)),
            _fila("Skewness", fmt_num(statistics.skewness)),
            _fila("Kurtosis (exceso)", fmt_num(statistics.kurtosis)),
            _fila("Tail Ratio", fmt_num(_tail_ratio(closes.pct_change().dropna()))),
            _fila("VaR 95%", fmt_pct_frac(statistics.pct_daily_p5, signo=True)),
            _fila("CVaR 95%", fmt_pct_frac(statistics.cvar_95, signo=True)),
        ]
    else:
        filas_reemplazo = [
            _fila("Retorno acumulado", fmt_pct_frac(returns.cumulative_return, signo=True)),
        ]
        filas_forma = [
            _fila("Sharpe", fmt_num(market.sharpe)),
            _fila("Sortino", fmt_num(market.sortino_ratio)),
        ]

    # Real yield differential = (tipo nominal − inflación YoY) base − cotizada.
    # Solo aplica a pares de divisas (par FX); las criptomonedas no tienen
    # tipos de interés y no llevan esta fila. La inflación (FRED) solo está si
    # hay key y la serie responde; si alguna divisa no tiene (AUD/NZD no
    # publican IPC mensual en FRED), la fila se muestra con N/A honesto — nada
    # se inventa.
    if data.categoria == "fx":
        inflacion = datos_inflacion()
        base, cotizada = key[:3], key[3:6]
        real_base = (tipos_fx[base] - inflacion[base]
                     if base in tipos_fx and base in inflacion else None)
        real_cotizada = (tipos_fx[cotizada] - inflacion[cotizada]
                         if cotizada in tipos_fx and cotizada in inflacion else None)
        if real_base is not None and real_cotizada is not None:
            valor_yield = fmt_pct_100(real_base - real_cotizada, signo=True)
        else:
            valor_yield = "N/A"
        filas_yield = [_fila("Real Yield Differential", valor_yield)]
    else:
        filas_yield = []

    filas = [
        _fila_retorno("Retorno YTD", retorno_ytd),
        _fila_retorno("Retorno 1M", market.return_1m),
        _fila_retorno("Retorno 1Y", market.return_1y),
        _fila("Volatilidad 30D", fmt_pct_frac(market.volatility_30d)),
        _fila("Volatilidad 1Y", fmt_pct_frac(market.volatility_1y)),
        *filas_forma,
        fila_carry,
        *filas_yield,
        _fila("Drawdown actual", fmt_pct_frac(returns.current_drawdown, signo=True)),
        _fila("Momentum (0-100)", fmt_pct_100(market.momentum_score)),
        _fila("CAGR", fmt_pct_frac(returns.cagr, signo=True)),
        *filas_reemplazo,
        _fila(f"Beta vs {benchmark}", fmt_num(market.beta)),
        _fila(f"Correlación vs {benchmark}", fmt_num(market.correlation_spx)),
        _fila(f"R² vs {benchmark}", fmt_pct_frac(timeseries.r2_vs_spy)),
        _fila(f"Alpha vs {benchmark}", fmt_pct_frac(timeseries.ols_alpha_annual, signo=True)),
        _fila_prob("P(retorno 12M < 0)", probability.prob_negative, alto_es_riesgo=True),
        _fila_prob(f"P(ganar a {benchmark} 12M)", probability.prob_outperform_spx, alto_es_riesgo=False),
    ]

    tabla = "\n".join(filas)
    img_precio = _grafico_precio(data.price_data)

    return f"""
    <div class="macro-card" id="m-{key}">
        <div class="macro-head">
            <h3>{data.label} <span class="ticker">({data.ticker_yf})</span></h3>
            <div class="macro-precio">{precio_html} <span class="ret1d">{ret1d_html}</span></div>
            <div class="sub">Periodo {data.period} · Benchmark {benchmark}</div>
        </div>
        <table class="macro-table">
            {tabla}
        </table>
        <img src="data:image/png;base64,{img_precio}" alt="Precio {data.label}">
    </div>
    """
