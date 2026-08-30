"""Métricas de panel autocontenidas para las tarjetas macro.

Cada tarjeta macro muestra un subconjunto de métricas (Market, Returns,
Statistics, TimeSeries y Probability) calculadas localmente con pandas/numpy —
sin dependencias privadas: el repo se clona y regenera en cualquier máquina.

Parámetros fijos: rf 4 %, bootstrap 10.000 simulaciones × 252 días con seed 42,
ventanas 21/126/252. Cualquier dato ausente → None (→ N/A en el panel).
Nunca se inventa un valor.

Unidades: fracciones salvo las probabilidades bootstrap (``prob_*`` ya en %
0-100). `panel_activo` formatea cada grupo con su escala.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------

def clean_number(value) -> Optional[float]:
    """Convierte a float o None (si es None, no finito o no convertible)."""
    if value is None:
        return None

    try:
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def mean_available(values) -> Optional[float]:
    """Media ignorando valores None; None si no hay ninguno."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


# ---------------------------------------------------------------------------
# Market — precio, retornos por ventana, volatilidad, momentum, drawdown,
# Sharpe/Sortino y beta/correlación vs benchmark.
# ---------------------------------------------------------------------------

@dataclass
class MarketMetrics:
    current_price: Optional[float] = None
    return_1m: Optional[float] = None
    return_1y: Optional[float] = None
    volatility_30d: Optional[float] = None
    volatility_1y: Optional[float] = None
    momentum_score: Optional[float] = None
    sharpe: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    beta: Optional[float] = None
    correlation_spx: Optional[float] = None


def metricas_market(
    prices: pd.Series,
    benchmark: Optional[pd.Series],
    dias_anio: int = 252,
) -> MarketMetrics:
    """Métricas de mercado del activo (mismo cálculo que MarketEngine).

    Args:
        prices: Serie de cierres del activo.
        benchmark: Serie de cierres del benchmark (o None si no hay).
        dias_anio: Días de negociación al año (252 hábiles; 365 en crypto,
            que cotiza todos los días). Afecta a la anualización de la
            volatilidad (√dias_anio) y a las ventanas 1M/1Y.
    """
    prices = prices.dropna()
    returns = prices.pct_change().dropna()

    m = MarketMetrics()
    if prices.empty:
        return m

    current = clean_number(prices.iloc[-1])
    m.current_price = current

    def trailing_return(days: int) -> Optional[float]:
        if len(prices) <= days:
            return None
        old = clean_number(prices.iloc[-days - 1])
        if old is None or old == 0:
            return None
        return current / old - 1

    m.return_1m = trailing_return(dias_anio // 12)
    m.return_1y = trailing_return(dias_anio)

    if len(returns) >= 30:
        m.volatility_30d = returns.tail(30).std() * np.sqrt(dias_anio)
    if len(returns) >= dias_anio:
        m.volatility_1y = returns.tail(dias_anio).std() * np.sqrt(dias_anio)

    # Momentum score (normalización, no predicción).
    momentum: list[float] = []
    for ret in (m.return_1m, trailing_return(dias_anio // 2), m.return_1y):
        if ret is not None:
            momentum.append(float(np.clip(50 + ret * 100, 0, 100)))

    if len(prices) >= 50 and len(prices) >= 200:
        ma_50 = prices.tail(50).mean()
        ma_200 = prices.tail(200).mean()
        momentum.append(75 if ma_50 > ma_200 else 35)

    m.momentum_score = mean_available(momentum)

    # Drawdown máximo y Sharpe/Sortino (risk-free 4 %).
    m.max_drawdown = clean_number((prices / prices.cummax() - 1).min())

    if len(returns) >= dias_anio:
        year_rets = returns.tail(dias_anio)
        annual_return = (1 + year_rets).prod() - 1
        volatility = year_rets.std() * np.sqrt(dias_anio)
        risk_free = 0.04

        if volatility > 0:
            m.sharpe = (annual_return - risk_free) / volatility

        downside = year_rets[year_rets < 0]
        if len(downside) > 0:
            downside_dev = downside.std() * np.sqrt(252)
            if downside_dev > 0:
                m.sortino_ratio = (annual_return - risk_free) / downside_dev

    # Beta / correlación vs benchmark (frame inner join, >= 60 obs).
    if benchmark is not None:
        spy_returns = benchmark.dropna().pct_change().dropna()
        combined = pd.concat([returns, spy_returns], axis=1, join="inner").dropna()

        if len(combined) >= 60:
            asset = combined.iloc[:, 0]
            bench = combined.iloc[:, 1]

            covariance = np.cov(asset, bench, ddof=1)[0, 1]
            variance = np.var(bench, ddof=1)

            if variance > 0:
                m.beta = covariance / variance

            std_asset = asset.std(ddof=1)
            std_bench = bench.std(ddof=1)
            if variance > 0 and std_asset > 0 and std_bench > 0:
                m.correlation_spx = covariance / (std_asset * std_bench)

    return m


# ---------------------------------------------------------------------------
# Returns — retorno compuesto y estado del drawdown.
# ---------------------------------------------------------------------------

@dataclass
class ReturnsMetrics:
    cumulative_return: Optional[float] = None
    cagr: Optional[float] = None
    current_drawdown: Optional[float] = None


def metricas_returns(prices: pd.Series, dias_anio: int = 252) -> ReturnsMetrics:
    """CAGR, retorno acumulado y drawdown actual (mismo cálculo que ReturnsEngine).

    Args:
        prices: Serie de cierres del activo.
        dias_anio: Días de negociación al año (252 hábiles; 365 en crypto).
    """
    prices = prices.dropna()
    returns = prices.pct_change().dropna()

    r = ReturnsMetrics()
    if len(returns) < 30:
        return r

    start = clean_number(prices.iloc[0])
    end = clean_number(prices.iloc[-1])

    if start is None or end is None or start <= 0 or end <= 0 or len(returns) == 0:
        return r

    r.cumulative_return = end / start - 1
    r.cagr = (end / start) ** (dias_anio / len(returns)) - 1

    peak_value = clean_number(prices.cummax().iloc[-1])
    if peak_value is not None and peak_value > 0:
        r.current_drawdown = clean_number(end / peak_value - 1)

    return r


# ---------------------------------------------------------------------------
# Statistics — momentos y riesgo de cola de los retornos diarios.
# ---------------------------------------------------------------------------

@dataclass
class StatisticsMetrics:
    variance: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    pct_daily_p5: Optional[float] = None
    cvar_95: Optional[float] = None


def metricas_statistics(prices: pd.Series) -> StatisticsMetrics:
    """Skewness, curtosis exceso, VaR y CVaR 95 % (mismo cálculo que StatisticsEngine)."""
    returns = prices.dropna().pct_change().dropna()

    s = StatisticsMetrics()
    if len(returns) < 30:
        return s

    s.variance = float(returns.var())
    s.skewness = float(returns.skew())
    s.kurtosis = float(returns.kurt())

    s.pct_daily_p5 = float(returns.quantile(0.05))

    threshold = returns.quantile(0.05)
    tail = returns[returns <= threshold]
    if not tail.empty:
        s.cvar_95 = float(tail.mean())

    return s


# ---------------------------------------------------------------------------
# TimeSeries — regresión OLS diaria vs benchmark.
# ---------------------------------------------------------------------------

@dataclass
class TimeSeriesMetrics:
    r2_vs_spy: Optional[float] = None
    ols_alpha_annual: Optional[float] = None


def metricas_timeseries(prices: pd.Series, benchmark: Optional[pd.Series]) -> TimeSeriesMetrics:
    """R² y alpha anualizada de la regresión returns_activo ~ returns_benchmark."""
    returns = prices.dropna().pct_change().dropna()

    t = TimeSeriesMetrics()
    if len(returns) < 30 or benchmark is None:
        return t

    spy_returns = benchmark.dropna().pct_change().dropna()
    if spy_returns.empty:
        return t

    frame = pd.concat([returns, spy_returns], axis=1, join="inner").dropna()
    if len(frame) < 60:
        return t

    x = frame.iloc[:, 1].to_numpy(dtype=float)  # retornos del benchmark
    y = frame.iloc[:, 0].to_numpy(dtype=float)  # retornos del activo

    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return t

    t.ols_alpha_annual = clean_number(intercept * 252)

    y_pred = slope * x + intercept
    residuals = y - y_pred

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    if ss_tot > 0:
        t.r2_vs_spy = clean_number(1 - ss_res / ss_tot)

    return t


# ---------------------------------------------------------------------------
# Probability — bootstrap real (10.000 caminos × 252 días, seed 42).
# ---------------------------------------------------------------------------

@dataclass
class ProbabilityMetrics:
    prob_negative: Optional[float] = None
    prob_outperform_spx: Optional[float] = None


def metricas_probability(
    prices: pd.Series,
    benchmark: Optional[pd.Series],
    dias_anio: int = 252,
) -> ProbabilityMetrics:
    """P(retorno 12M < 0) y P(retorno 12M > retorno 12M del benchmark).

    Remuestreo con reemplazo de los pares de retornos diarios (activo,
    benchmark) sobre el frame alineado (inner join). Seed fija para
    reproducibilidad (secuencia RNG fija).

    Args:
        prices: Serie de cierres del activo.
        benchmark: Serie de cierres del benchmark (o None si no hay).
        dias_anio: Días del año de simulación (252 hábiles; 365 en crypto).
    """
    returns = prices.dropna().pct_change().dropna()

    p = ProbabilityMetrics()
    if len(returns) < 60 or benchmark is None:
        return p

    spy_returns = benchmark.dropna().pct_change().dropna()
    frame = pd.concat([returns, spy_returns], axis=1, join="inner").dropna()

    if len(frame) < 60:
        return p

    asset = frame.iloc[:, 0].values
    bench = frame.iloc[:, 1].values

    rng = np.random.default_rng(42)
    n = len(asset)
    days = dias_anio
    n_sims = 10_000

    outperf = 0
    neg = 0

    for _ in range(n_sims):
        idx = rng.integers(0, n, size=days)

        a = asset[idx]
        b = bench[idx]

        ret_a = float(np.prod(1 + a) - 1)
        ret_b = float(np.prod(1 + b) - 1)

        if ret_a > ret_b:
            outperf += 1

        if ret_a < 0:
            neg += 1

    p.prob_negative = float(neg / n_sims) * 100
    p.prob_outperform_spx = float(outperf / n_sims) * 100

    return p


# ---------------------------------------------------------------------------
# Punto de entrada: dict con el mismo shape que espera macro_motores/panel.
# ---------------------------------------------------------------------------

def metricas(
    prices: pd.Series,
    benchmark: Optional[pd.Series],
    dias_anio: int = 252,
) -> dict:
    """Todas las métricas de las tarjetas, en un dict.

    Args:
        prices: Serie ``Close`` del activo.
        benchmark: Serie ``Close`` del benchmark (o None si no hay).
        dias_anio: Días de negociación al año (252 hábiles; 365 en crypto).

    Returns:
        Dict con claves ``market``, ``returns``, ``statistics``,
        ``timeseries`` y ``probability``.
    """
    return {
        "market": metricas_market(prices, benchmark, dias_anio),
        "probability": metricas_probability(prices, benchmark, dias_anio),
        "statistics": metricas_statistics(prices),
        "returns": metricas_returns(prices, dias_anio),
        "timeseries": metricas_timeseries(prices, benchmark),
    }
