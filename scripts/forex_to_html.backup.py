import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import base64
from io import BytesIO
from datetime import datetime
import json
import os
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore") # Para evitar avisos de convergencia de ARIMA
try:
    from arch import arch_model
except ImportError:
    pass
from scipy.stats import norm

# --- FUNCIONES ESTOCÁSTICAS AVANZADAS ---

def simulate_gbm(prices, n_days=30, n_sims=1000):
    prices = prices.dropna()
    if len(prices) < 2: return {"median": [], "p5": [], "p95": [], "prob_up": 0.5}
    returns = prices.pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    last_price = prices.iloc[-1]
    
    # Simular retornos log-normales
    dt = 1
    sims = np.zeros((n_sims, n_days))
    for i in range(n_sims):
        rand_shocks = np.random.normal(mu, sigma, n_days)
        price_path = last_price * np.exp(np.cumsum(rand_shocks))
        sims[i, :] = price_path
        
    return {
        "median": np.nan_to_num(np.median(sims, axis=0)).tolist(),
        "p5": np.nan_to_num(np.percentile(sims, 5, axis=0)).tolist(),
        "p95": np.nan_to_num(np.percentile(sims, 95, axis=0)).tolist(),
        "prob_up": float(np.mean(sims[:, -1] > last_price))
    }

def simulate_garch(prices, n_days=30, n_sims=1000):
    prices = prices.dropna()
    if len(prices) < 2: return {"median": [], "p5": [], "p95": [], "prob_up": 0.5}
    returns = prices.pct_change().dropna()
    last_price = prices.iloc[-1]
    
    try:
        # Escalar para convergencia estable
        rescale_factor = 100
        returns_scaled = returns * rescale_factor
        
        model = arch_model(returns_scaled, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
        res = model.fit(disp='off')
        
        # Pronóstico de varianza para n_days
        forecast = res.forecast(horizon=n_days, method='simulation', simulations=n_sims)
        sim_returns = forecast.simulations.values[0] / rescale_factor
        sim_returns = np.nan_to_num(sim_returns, nan=0.0)
        sim_returns = np.clip(sim_returns, -0.05, 0.05)
        
        sims = last_price * np.exp(np.cumsum(sim_returns, axis=1))
        
        return {
            "median": np.nan_to_num(np.median(sims, axis=0)).tolist(),
            "p5": np.nan_to_num(np.percentile(sims, 5, axis=0)).tolist(),
            "p95": np.nan_to_num(np.percentile(sims, 95, axis=0)).tolist(),
            "prob_up": float(np.mean(sims[:, -1] > last_price))
        }
    except Exception:
        return simulate_gbm(prices, n_days, n_sims)

def simulate_merton(prices, n_days=30, n_sims=1000):
    prices = prices.dropna()
    if len(prices) < 5: return {"median": [], "p5": [], "p95": [], "prob_up": 0.5}
    returns = prices.pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    last_price = prices.iloc[-1]
    
    jumps = returns[np.abs(returns - mu) > 3 * sigma]
    lam = len(jumps) / len(returns) if len(returns) > 0 else 0.01
    mu_j = jumps.mean() if not jumps.empty else 0
    sigma_j = jumps.std() if not jumps.empty else 0.02
    
    sims = np.zeros((n_sims, n_days))
    for i in range(n_sims):
        W = np.random.normal(0, 1, n_days)
        N = np.random.poisson(lam, n_days)
        J = np.random.normal(mu_j, sigma_j, n_days)
        
        path = np.zeros(n_days)
        curr = last_price
        for t in range(n_days):
            jump_comp = N[t] * J[t]
            curr = curr * np.exp(mu + sigma * W[t] + jump_comp)
            path[t] = curr
        sims[i, :] = path
        
    return {
        "median": np.nan_to_num(np.median(sims, axis=0)).tolist(),
        "p5": np.nan_to_num(np.percentile(sims, 5, axis=0)).tolist(),
        "p95": np.nan_to_num(np.percentile(sims, 95, axis=0)).tolist(),
        "prob_up": float(np.mean(sims[:, -1] > last_price))
    }

def calculate_markov_matrix(prices):
    returns = prices.pct_change().dropna()
    # Definir estados: UP (> 0.05%), DOWN (< -0.05%), FLAT
    threshold = 0.0005
    states = []
    for r in returns:
        if r > threshold: states.append(0)   # 0: UP
        elif r < -threshold: states.append(1) # 1: DOWN
        else: states.append(2)               # 2: FLAT
        
    matrix = np.zeros((3, 3))
    for i in range(len(states)-1):
        matrix[states[i], states[i+1]] += 1
        
    # Normalizar probabilisticamente
    for i in range(3):
        r_sum = np.sum(matrix[i, :])
        if r_sum > 0:
            matrix[i, :] = matrix[i, :] / r_sum
        else:
            matrix[i, :] = 1/3
            
    return matrix.tolist()

def get_consensus_signal(bias, p_up, p_down, markov, seasonality_avg, price, m_hi, m_lo):
    score = 0
    reasons = []
    
    # 1. Trend Bias (Weight 0.25)
    if "LARGO" in bias or "BULL" in bias: 
        score += 0.25
        reasons.append("Tendencia Alcista")
    elif "CORTO" in bias or "BEAR" in bias: 
        score -= 0.25
        reasons.append("Tendencia Bajista")
        
    # 2. Stochastic Models (Weight 0.35)
    if p_up > 0.60: 
        score += 0.35
        reasons.append("Modelos Estocásticos Alcistas")
    elif p_up < 0.40: 
        score -= 0.35
        reasons.append("Modelos Estocásticos Bajistas")
        
    # 3. Markov Transitions (Weight 0.15)
    if markov[0][0] > 0.45: score += 0.15
    if markov[1][1] > 0.45: score -= 0.15
    
    # 4. Monthly Range Context (Weight 0.15)
    range_size = m_hi - m_lo
    if range_size > 0:
        pos = (price - m_lo) / range_size
        if pos > 0.90: 
            score -= 0.15
            reasons.append("Agotamiento (Extremo Superior)")
        elif pos < 0.10: 
            score += 0.15
            reasons.append("Sobreventa (Extremo Inferior)")
            
    # 5. Seasonality (Weight 0.10)
    if seasonality_avg > 0.5: 
        score += 0.10
        reasons.append("Estacionalidad Favorable")
    elif seasonality_avg < -0.5: 
        score -= 0.10
        reasons.append("Estacionalidad Desfavorable")
        
    # Result Narrative
    direccion = ""
    intensidad = ""
    if score >= 0.4: 
        intensidad = "inclinada"
        direccion = "LARGAS"
        signal = "COMPRA FUERTE"
        color = "#3fb950"
    elif score >= 0.1: 
        intensidad = "ligera"
        direccion = "LARGAS"
        signal = "COMPRA"
        color = "#a5d6ff"
    elif score <= -0.4: 
        intensidad = "inclinada"
        direccion = "CORTAS"
        signal = "VENTA FUERTE"
        color = "#f85149"
    elif score <= -0.1: 
        intensidad = "ligera"
        direccion = "CORTAS"
        signal = "VENTA"
        color = "#ffa657"
    else: 
        signal = "NEUTRAL"
        color = "#8b949e"
        return signal, color, "Los datos actuales sugieren un escenario mixto sin una tendencia clara definida."

    summary = f"La tendencia, los modelos estocásticos, el modelo de Markov y la estacionalidad indican una {intensidad} tendencia a posiciones {direccion}."
    return signal, color, summary

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def analyze_forex_to_html():
    symbols = [
        "USDJPY=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "NZDJPY=X", "CADJPY=X",
        "USDCAD=X", "GBPUSD=X", "EURUSD=X", "AUDUSD=X", "NZDUSD=X", "EURGBP=X",
        "USDCHF=X", "EURCAD=X", "SEKJPY=X", "SGDJPY=X", "EURAUD=X", "GBPCAD=X",
        "EURNZD=X", "CADCHF=X", "GBPCHF=X", "AUDCAD=X", "AUDCHF=X", "CHFJPY=X",
        "NZDCHF=X", "ZARJPY=X", "EURNOK=X", "EURCHF=X", "CHFSEK=X", "GBPNOK=X", "NZDNOK=X"
    ]
    
    print(f"Descargando datos para {len(symbols)} pares...")
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=2)
    
    try:
        raw_data = yf.download(symbols, start=start_date, end=end_date, progress=False)
        data_close = (raw_data['Adj Close'] if 'Adj Close' in raw_data.columns else raw_data['Close']).ffill()
        highs = raw_data['High'].ffill()
        lows = raw_data['Low'].ffill()
        data_range = ((highs - lows) / data_close) * 100
        m_high = highs.resample('ME').max()
        m_low = lows.resample('ME').min()
        m_close = data_close.resample('ME').last()
        data_monthly_range = ((m_high - m_low) / m_close) * 100
    except Exception as e:
        print(f"Error: {e}")
        return

    new_cols = []
    for col in data_close.columns:
        name = col[-1] if isinstance(col, tuple) else col
        new_cols.append(name.replace('=X', ''))
    
    data_close.columns = new_cols
    data_range.columns = new_cols
    data_monthly_range.columns = new_cols
    highs.columns = new_cols
    lows.columns = new_cols
    
    data_close = data_close.loc[:, ~data_close.columns.duplicated()]
    data_range = data_range.loc[:, ~data_range.columns.duplicated()]
    data_monthly_range = data_monthly_range.loc[:, ~data_monthly_range.columns.duplicated()]
    highs = highs.loc[:, ~highs.columns.duplicated()]
    lows = lows.loc[:, ~lows.columns.duplicated()]
    
    returns = data_close.pct_change()
    corr_matrix = returns.corr()

    # 1. Heatmap Global
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="RdBu", center=0, annot=False, cbar_kws={'label': 'Correlación'})
    plt.title("Matriz Global", fontsize=16)
    img_corr = fig_to_base64(plt.gcf())
    plt.close()

    # 2. Día de la Semana (Fuerza por Activo y Global)
    returns['DayOfWeek'] = returns.index.day_name()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    # Datos para Fuerza por Activo
    dow_strength = {}
    for day in dow_order:
        day_rets = returns[returns['DayOfWeek'] == day].drop(columns=['DayOfWeek', 'DayOfMonth'], errors='ignore')
        if not day_rets.empty:
            corr_mat = day_rets.corr()
            for col in corr_mat.columns:
                if col not in dow_strength: dow_strength[col] = []
                avg_c = corr_mat[col].drop(labels=[col]).mean()
                dow_strength[col].append(float(avg_c) if not np.isnan(avg_c) else 0.0)
        else:
            for col in corr_matrix.columns:
                if col not in dow_strength: dow_strength[col] = []
                dow_strength[col].append(0.0)
    
    # Fuerza Global (Promedio del mercado)
    dow_corrs_global = []
    for day in dow_order:
        day_rets = returns[returns['DayOfWeek'] == day].drop(columns=['DayOfWeek', 'DayOfMonth'], errors='ignore')
        if not day_rets.empty:
            m = day_rets.corr().values
            val = np.nanmean(m[np.triu_indices_from(m, k=1)])
            dow_corrs_global.append(float(val) if not np.isnan(val) else 0.0)
        else: dow_corrs_global.append(0.0)
    
    # Añadimos la Global como un "activo" especial para referencia si se desea
    dow_strength['GLOBAL'] = dow_corrs_global
    dow_strength_json = json.dumps(dow_strength)

    # 3. Estacionalidad de Volatilidad (Día del Mes) y Pips
    returns['DayOfMonth'] = returns.index.day
    abs_returns = returns.drop(columns=['DayOfWeek', 'DayOfMonth'], errors='ignore').abs() * 100
    abs_returns['DayOfMonth'] = returns['DayOfMonth']
    abs_returns['DayOfWeek'] = returns['DayOfWeek']
    
    # Calcular movimiento en Pips (High - Low)
    pips_df = (highs - lows).copy()
    for col in pips_df.columns:
        multiplier = 100 if "JPY" in col else 10000
        pips_df[col] = pips_df[col] * multiplier
    
    pips_df['DayOfMonth'] = returns['DayOfMonth']
    pips_df['DayOfWeek'] = returns['DayOfWeek']
    
    vol_seasonality = abs_returns.groupby('DayOfMonth').mean(numeric_only=True)
    vol_seasonality_dow = abs_returns.groupby('DayOfWeek').mean(numeric_only=True).reindex(dow_order)
    pips_seasonality = pips_df.groupby('DayOfMonth').mean(numeric_only=True)
    pips_seasonality_dow = pips_df.groupby('DayOfWeek').mean(numeric_only=True).reindex(dow_order)
    
    pips_seasonality.columns = [c.replace('=X', '') for c in pips_seasonality.columns]
    pips_seasonality_dow.columns = [c.replace('=X', '') for c in pips_seasonality_dow.columns]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(vol_seasonality.T, cmap="YlOrRd", annot=False, cbar_kws={'label': 'Volatilidad %'})
    plt.title("Volatilidad / Día Mes", fontsize=16)
    img_dom = fig_to_base64(plt.gcf())
    plt.close()

    # 4. Rango Diario
    avg_range = data_range.mean().sort_values()
    plt.figure(figsize=(10, 8))
    avg_range.plot(kind='barh', color=sns.color_palette("flare", len(avg_range)))
    plt.title("Rango Diario (H-L %)", fontsize=16)
    plt.grid(axis='x', alpha=0.2)
    img_range = fig_to_base64(plt.gcf())
    plt.close()

    # 4b. Rango Mensual
    avg_m_range = data_monthly_range.mean().sort_values()
    plt.figure(figsize=(10, 8))
    avg_m_range.plot(kind='barh', color=sns.color_palette("crest", len(avg_m_range)))
    plt.title("Rango Mensual (H-L %)", fontsize=16)
    plt.grid(axis='x', alpha=0.2)
    img_m_range = fig_to_base64(plt.gcf())
    plt.close()

    # 5. Ranking Correlaciones
    stacked_corr = corr_matrix.stack()
    stacked_corr = stacked_corr[stacked_corr < 0.99]
    top_pos = stacked_corr.sort_values(ascending=False).head(20)[::2]
    top_neg = stacked_corr.sort_values(ascending=True).head(20)[::2]
    ranking_data = pd.concat([top_pos, top_neg])
    ranking_labels = [f"{a} / {b}" for a, b in ranking_data.index]
    
    plt.figure(figsize=(10, 8))
    colors = ['#3fb950' if x > 0 else '#f85149' for x in ranking_data.values]
    sns.barplot(x=ranking_data.values, y=ranking_labels, palette=colors, hue=ranking_labels, legend=False)
    plt.title("Ranking Correlaciones", fontsize=16)
    img_ranking = fig_to_base64(plt.gcf())
    plt.close()

    # 6. Descarga y Preparación de Precios por Temporalidad
    print("Descargando temporalidades adicionales (esto puede tardar un momento)...")
    
    # 1d y 1w (ya tenemos 1d en data_close)
    prices_1d = data_close.tail(200)
    prices_1w = data_close.resample('W').last().tail(100)
    
    # 1h (máximo 730 días permitido por yfinance)
    raw_1h = yf.download(symbols, period="1y", interval="1h", progress=False)
    prices_1h_all = (raw_1h['Adj Close'] if 'Adj Close' in raw_1h.columns else raw_1h['Close']).ffill()
    
    # Limpiar columnas 1h
    prices_1h_all.columns = [c.replace('=X', '') for c in prices_1h_all.columns]
    prices_1h = prices_1h_all.tail(200)
    
    # 4h (remuestreo de 1h)
    prices_4h = prices_1h_all.resample('4H').last().tail(200)

    # Cálculo de Kernel Regression (Nadaraya-Watson) para 1D
    def kernel_smooth(series, bandwidth=3.0):
        y = series.values
        x = np.arange(len(y))
        smoothed = []
        for xi in x:
            weights = np.exp(-0.5 * ((x - xi) / bandwidth)**2)
            smoothed.append(np.sum(weights * y) / np.sum(weights))
        return smoothed

    # Estructura de DATOS para JS
    price_pack = {
        "1h": {"data": prices_1h.to_dict(orient='list'), "labels": [d.strftime('%H:%M %d/%m') for d in prices_1h.index]},
        "4h": {"data": prices_4h.to_dict(orient='list'), "labels": [d.strftime('%H:%M %d/%m') for d in prices_4h.index]},
        "1d": {
            "data": prices_1d.to_dict(orient='list'), 
            "labels": [d.strftime('%d/%m/%y') for d in prices_1d.index]
        },
        "1w": {"data": prices_1w.to_dict(orient='list'), "labels": [d.strftime('%d/%m/%y') for d in prices_1w.index]}
    }
    price_pack_json = json.dumps(price_pack)
    # Rango Mensual: Max/Min REALES de las últimas 22 sesiones
    # Equivalente a: ref_price * (1 + monthly_range_pct/100) — misma fórmula que el Resumen
    monthly_levels = {}
    for col in data_close.columns:
        p_m = data_close[col].tail(22)
        ref_price = float(p_m.iloc[0])  # Precio de apertura del periodo (igual que en Resumen)
        high_22   = float(highs[col].tail(22).max())
        low_22    = float(lows[col].tail(22).min())

        # Estos % coinciden exactamente con lo que muestra el panel Resumen
        pct_high = round((high_22 - ref_price) / ref_price * 100, 2)
        pct_low  = round((low_22  - ref_price) / ref_price * 100, 2)

        monthly_levels[col] = {
            "proj_high": round(high_22, 5),
            "proj_low":  round(low_22,  5),
            "avg_up":    pct_high,
            "avg_down":  pct_low,
            "ref":       round(ref_price, 5)
        }
    monthly_levels_json = json.dumps(monthly_levels)

    # 7. Estadísticas Extra por Activo para la nueva KPI
    stats_dict = {}
    gbm_mc = {}
    garch_mc = {}
    merton_mc = {}
    markov_dict = {}
    
    current_day = end_date.day
    current_dow = end_date.strftime('%A')
    
    # Pre-calcular correlaciones extremas para cada par
    for col in data_close.columns:
        # Pips factor
        multiplier = 100 if "JPY" in col else 10000
        
        # Correlaciones
        corrs = corr_matrix[col].drop(labels=[col]).dropna()
        if not corrs.empty:
            max_c_pair = corrs.idxmax()
            min_c_pair = corrs.idxmin()
            max_c_val = f"{max_c_pair} ({corrs[max_c_pair]:.2f})"
            min_c_val = f"{min_c_pair} ({corrs[min_c_pair]:.2f})"
        else:
            max_c_val = "N/A"
            min_c_val = "N/A"
        
        # Volatilidad máxima
        best_dom = vol_seasonality[col].idxmax() if not vol_seasonality[col].isna().all() else 0
        best_dow = vol_seasonality_dow[col].idxmax() if not vol_seasonality_dow[col].isna().all() else "N/A"
        
        # Sesgos (Largo/Corto)
        ret_20d = (data_close[col].iloc[-1] / data_close[col].iloc[-20] - 1) * 100
        bias_20d = "LARGO" if ret_20d > 0 else "CORTO"
        bias_20d_color = "#3fb950" if bias_20d == "LARGO" else "#f85149"
        
        ret_5d = (data_close[col].iloc[-1] / data_close[col].iloc[-5] - 1) * 100
        bias_5d = "LARGO" if ret_5d > 0 else "CORTO"
        bias_5d_color = "#3fb950" if bias_5d == "LARGO" else "#f85149"
        
        # Rango Anual % (+/- desde inicio de periodo)
        prices_year = data_close[col].tail(252)
        ref_price_y = prices_year.iloc[0]
        max_y = ((highs[col].tail(252).max() - ref_price_y) / ref_price_y) * 100
        min_y = ((lows[col].tail(252).min() - ref_price_y) / ref_price_y) * 100
        
        # Rango Mensual % (+/- desde INICIO de periodo 22 días)
        prices_month = data_close[col].tail(22)
        ref_price_m = prices_month.iloc[0]
        max_m = ((highs[col].tail(22).max() - ref_price_m) / ref_price_m) * 100
        min_m = ((lows[col].tail(22).min() - ref_price_m) / ref_price_m) * 100
        
        # Promedios de Pips
        avg_pips = pips_df[col].mean()
        
        # Pips Semanales (H-L de la semana)
        weekly_h = highs[col].resample('W').max()
        weekly_l = lows[col].resample('W').min()
        avg_pips_w = ((weekly_h - weekly_l) * multiplier).mean()
        
        # ------------------------------------------------------------------
        # PROYECCIÓN A 5 DÍAS (experimental)
        # Combina: sesgo tendencia, pips medios diarios, std volatilidad 30d
        # ------------------------------------------------------------------
        series = data_close[col].dropna()
        last_price = series.iloc[-1]
        
        # ------------------------------------------------------------------
        # PROYECCIÓN ARIMA (Próximos 5 días)
        # ------------------------------------------------------------------
        series_arima = series.tail(100).dropna()
        proj_arima = last_price # Inicialización de seguridad
        
        try:
            # Fit ARIMA
            model = ARIMA(series_arima, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=5)
            proj_arima = float(forecast.iloc[-1])
            
            # Ajustamos el rango a la volatilidad real de los últimos 10 días
            # Esto evita que el rango se dispare por errores viejos del modelo
            vol_10d = series.tail(11).pct_change().std() * last_price # Vol de 10 retornos
            
            # Proyectamos ±1 Sigma (10d) para los 5 días
            proj_high_arima = proj_arima + (vol_10d * np.sqrt(5))
            proj_low_arima  = proj_arima - (vol_10d * np.sqrt(5))
            
            # ACELERADOR DE SEGURIDAD: No permitir que el rango exceda ±3% del precio actual
            # Esto evita proyecciones absurdas en cruces con poca liquidez o fallos de modelo
            max_dev = last_price * 0.03 
            proj_high_arima = min(proj_high_arima, last_price + max_dev)
            proj_low_arima  = max(proj_low_arima, last_price - max_dev)
            
        except Exception:
            # Fallback simple
            rets_5 = series_arima.pct_change().tail(10).mean()
            proj_arima = last_price * (1 + rets_5 * 5)
            proj_high_arima = proj_arima * 1.005
            proj_low_arima  = proj_arima * 0.995

        pips_dec = 2 if "JPY" in col else 5
        pct_h = ((proj_high_arima / last_price) - 1) * 100
        pct_l = ((proj_low_arima  / last_price) - 1) * 100
        
        forecast_h_str = f"{proj_high_arima:.{pips_dec}f}"
        forecast_l_str = f"{proj_low_arima:.{pips_dec}f}"
        forecast_h_pct_str = f"{'+' if pct_h>=0 else ''}{pct_h:.3f}%"
        forecast_l_pct_str = f"{'+' if pct_l>=0 else ''}{pct_l:.3f}%"
        
        # ------------------------------------------------------------------
        # Últimos datos
        last_range_pct = data_range[col].iloc[-1]
        last_range_pips = (highs[col].iloc[-1] - lows[col].iloc[-1]) * multiplier
        
        # Promedios
        m_daily_range = avg_range[col]
        m_monthly_range = avg_m_range[col]
           # Calcular modelos estocásticos y consenso para esta columna
        p_y = data_close[col].tail(252)
        gbm_res = simulate_gbm(p_y)
        garch_res = simulate_garch(p_y)
        merton_res = simulate_merton(p_y)
        
        gbm_mc[col] = gbm_res
        garch_mc[col] = garch_res
        merton_mc[col] = merton_res
        
        avg_up = (gbm_res['prob_up'] + garch_res['prob_up'] + merton_res['prob_up']) / 3
        avg_down = 1 - avg_up
        
        # Markov
        markov_mat = calculate_markov_matrix(p_y)
        markov_dict[col] = markov_mat
        
        # Estacionalidad del mes actual
        s_rets = returns.groupby(returns.index.month)[col].mean()
        season_avg = float(s_rets.get(end_date.month, 0)) * 100
        
        # Consenso
        m_hi = float(highs[col].tail(22).max())
        m_lo = float(lows[col].tail(22).min())
        c_signal, c_color, c_summary = get_consensus_signal(bias_5d, avg_up, avg_down, markov_mat, season_avg, last_price, m_hi, m_lo)
        
        stats_dict[col] = {

            "price": f"{last_price:.4f}",
            "range_pct": f"{last_range_pct:.2f}%",
            "range_pips": f"{last_range_pips:.0f}",
            "avg_daily": f"{m_daily_range:.2f}%",
            "avg_pips_daily": f"{int(avg_pips)}",
            "avg_pips_week": f"{int(avg_pips_w)}",
            "max_corr": max_c_val,
            "min_corr": min_c_val,
            "best_dom": f"{int(best_dom)}",
            "best_dow": best_dow,
            "bias_20d": bias_20d,
            "bias_20d_color": bias_20d_color,
            "bias_5d": bias_5d,
            "bias_5d_color": bias_5d_color,
            "yearly_range": f"{round(lows[col].tail(252).min(), 4)} - {round(highs[col].tail(252).max(), 4)}",
            "monthly_range": f"{m_lo:.4f} - {m_hi:.4f}",
            "p_up": f"{avg_up*100:.1f}%",
            "p_down": f"{avg_down*100:.1f}%",
            "f_h": f"{proj_high_arima:.5f}",
            "f_l": f"{proj_low_arima:.5f}",
            "f_h_p": f"{pct_h:+.2f}%",
            "f_l_p": f"{pct_l:+.2f}%",
            "c_signal": c_signal,
            "c_color": c_color,
            "c_summary": c_summary
        }
    
    pair_stats_json = json.dumps(stats_dict)

    # JSONs de Volatilidad y Pips (Restaurados)
    vol_data_json = vol_seasonality.to_json(orient='columns')
    vol_dow_json = vol_seasonality_dow.to_json(orient='columns')
    pips_data_json = pips_seasonality.to_json(orient='columns')
    pips_dow_json = pips_seasonality_dow.to_json(orient='columns')

    # 8. Datos para Historial de Retornos Diarios (Dispersión 3D) - Último Año
    returns_history = {}
    for col in data_close.columns:
        # Últimos 252 días
        rets_y = returns[col].tail(252)
        prices_y = data_close[col].tail(252)
        labels_y = [d.strftime('%Y-%m-%d') for d in rets_y.index]
        
        data_pts = {
            "x": [], "y": [], "z": [], "text": [], "colors": []
        }
        for i in range(len(rets_y)):
            r = rets_y.iloc[i]
            p = prices_y.iloc[i]
            if not pd.isna(r) and not pd.isna(p):
                rx = round(float(r * 100), 3)
                py = round(float(p), 5)
                dt = labels_y[i]
                data_pts["x"].append(dt)
                data_pts["y"].append(py)
                data_pts["z"].append(rx)
                data_pts["text"].append(f"{dt}<br>Ret: {rx}%<br>Price: {py}")
                data_pts["colors"].append('#3fb950' if rx >= 0 else '#f85149')
        returns_history[col] = data_pts
    
    scatter_data_json = json.dumps(returns_history)
    
    # 9. Kernel Regression Histórica (Precios 1Y) para KPI dedicada
    kernel_history = {}
    for col in data_close.columns:
        p_y = data_close[col].tail(252)
        # Usamos el suavizado histórico
        k_smooth = kernel_smooth(p_y, bandwidth=5.0)
        
        kernel_history[col] = {
            "labels": [d.strftime('%d/%m/%y') for d in p_y.index],
            "price": [round(float(v), 5) for v in p_y.values],
            "kernel": [round(float(v), 5) for v in k_smooth]
        }
    kernel_history_json = json.dumps(kernel_history)
    
    # 10. Estacionalidad Mensual de Precios (Retorno % Medio)
    monthly_ret_seasonal = {}
    returns['Month'] = returns.index.month
    for col in data_close.columns:
        # Agrupar por mes y calcular la media del retorno * 100
        m_rets = returns.groupby('Month')[col].mean() * 100
        # Asegurar que tenemos los 12 meses
        m_rets_list = [round(float(m_rets.get(m, 0.0)), 3) for m in range(1, 13)]
        monthly_ret_seasonal[col] = m_rets_list
    
    monthly_seasonality_json = json.dumps(monthly_ret_seasonal)

    gbm_json = json.dumps(gbm_mc)
    garch_json = json.dumps(garch_mc)
    merton_json = json.dumps(merton_mc)
    markov_json = json.dumps(markov_dict)


    # 11. Estructura Jerárquica para Sunburst (Market Performance)
    # Jerarquía: Market -> Base Currency -> Instrument
    base_currencies = set()
    for col in data_close.columns:
        base_currencies.add(col[:3])
    
    ids = ["Market"]
    labels = ["MARKET PERFORMANCE"]
    parents = [""]
    values = [0]
    colors = [0]
    hovers = ["Market Performance"]
    
    # Agregar Monedas Base
    for bc in sorted(base_currencies):
        ids.append(bc)
        labels.append(bc)
        parents.append("Market")
        values.append(0)
        colors.append(0)
        hovers.append(f"Currency: {bc}")
    
    # Agregar Instrumentos
    for col in data_close.columns:
        bc = col[:3]
        last_ret = float(returns[col].iloc[-1] * 100)
        ids.append(col)
        labels.append(col)
        parents.append(bc)
        values.append(1) # Tamaño fijo para cada par
        colors.append(last_ret)
        hovers.append(f"{col}: {last_ret:.3f}%")
        
    hierarchy_data = {
        "ids": ids,
        "labels": labels,
        "parents": parents,
        "values": values,
        "colors": colors,
        "hovers": hovers
    }
    market_hierarchy_json = json.dumps(hierarchy_data)

    # 12. Datos para Sankey (Flujo de Impacto de Mercado) - Global
    # Fuentes: Monedas Base -> Pares
    base_list = sorted(list(base_currencies))
    pair_list = sorted(data_close.columns.tolist())
    all_nodes = base_list + pair_list
    node_indices = {n: i for i, n in enumerate(all_nodes)}
    
    s_sources = []
    s_targets = []
    s_values = []
    s_colors = []
    
    for col in pair_list:
        bc = col[:3]
        impact = abs(float(returns[col].iloc[-1] * 100))
        if impact < 0.001: impact = 0.01 # Gracia visual
        s_sources.append(node_indices[bc])
        s_targets.append(node_indices[col])
        s_values.append(impact)
        s_colors.append('rgba(63, 185, 80, 0.4)' if returns[col].iloc[-1] >= 0 else 'rgba(248, 81, 73, 0.4)')
        
    sankey_data = {
        "labels": all_nodes,
        "sources": s_sources,
        "targets": s_targets,
        "values": s_values,
        "colors": s_colors
    }
    market_sankey_json = json.dumps(sankey_data)
    
    # 13. Datos para Bubble Timeline (Evolución de Volatilidad y Pips) - Por Activo
    bubble_history = {}
    for col in data_close.columns:
        # Últimos 45 días
        b_pips = pips_df[col].tail(45)
        b_range = data_range[col].tail(45)
        b_rets = returns[col].tail(45)
        b_dates = [d.strftime('%d/%m') for d in b_pips.index]
        
        bubble_history[col] = {
            "x": list(range(len(b_pips))),
            "y": [round(float(v), 2) for v in b_pips.values],
            "size": [round(float(v) * 20, 2) for v in b_range.values], # Factor de escala
            "color": ['#3fb950' if v >= 0 else '#f85149' for v in b_rets.values],
            "text": [f"Fecha: {d}<br>Pips: {p:.0f}<br>Rango: {r:.2f}%" for d, p, r in zip(b_dates, b_pips.values, b_range.values)],
            "dates": b_dates
        }
    pair_bubble_json = json.dumps(bubble_history)
    
    # 14. Datos para Superficie de Volatilidad 3D (Hour vs Day vs Vol)
    surface_history = {}
    dow_map = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4}
    
    # Usar datos de 1h para granularidad
    prices_1h_all.index = pd.to_datetime(prices_1h_all.index)
    prices_1h_all['Hour'] = prices_1h_all.index.hour
    prices_1h_all['DOW'] = prices_1h_all.index.day_name()
    
    for col in data_close.columns:
        if col in prices_1h_all.columns:
            # Calcular retornos absolutos (volatilidad) por hora
            rets_1h = prices_1h_all[col].pct_change().abs() * 100
            temp_df = pd.DataFrame({'Vol': rets_1h, 'Hour': prices_1h_all['Hour'], 'DOW': prices_1h_all['DOW']}).dropna()
            
            # Crear matriz 24x5
            z_data = np.zeros((24, 5))
            for dow_name, dow_idx in dow_map.items():
                for h in range(24):
                    val = temp_df[(temp_df['DOW'] == dow_name) & (temp_df['Hour'] == h)]['Vol'].mean()
                    z_data[h, dow_idx] = float(val) if not np.isnan(val) else 0.0
            
            surface_history[col] = z_data.tolist()
            
    pair_surface_json = json.dumps(surface_history)

    html_template = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Forex Dashboard PRO</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 10px; }}
            .container {{ max-width: 1950px; margin: auto; }}
            header {{ text-align: center; padding: 10px 0; border-bottom: 1px solid #30363d; margin-bottom: 15px; }}
            
            /* Grid Principal */
            .row-4-cols {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 15px; }}
            .row-3-cols {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 15px; }}
            .row-1-col {{ display: grid; grid-template-columns: 1fr; gap: 12px; margin-bottom: 15px; }}
            
            /* Fila Especial superior */
            .top-row {{ display: grid; grid-template-columns: 1.5fr 1fr 1fr; gap: 12px; margin-bottom: 15px; }}
            
            .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; display: flex; flex-direction: column; position: relative; }}
            h2 {{ color: #a5d6ff; border-left: 3px solid #58a6ff; padding-left: 8px; margin-top: 0; font-size: 0.95em; margin-bottom: 15px; }}
            img {{ width: 100%; height: auto; border-radius: 4px; object-fit: contain; }}
            
            .controls {{ background: #1c2128; padding: 12px; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 12px; display: flex; align-items: center; gap: 12px; }}
            select {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 8px 15px; border-radius: 6px; cursor: pointer; }}
            
            /* Botones de Temporalidad */
            .tf-buttons {{ position: absolute; top: 12px; right: 12px; display: flex; gap: 5px; }}
            .tf-btn {{ 
                background: #21262d; color: #8b949e; border: 1px solid #30363d; padding: 4px 8px; 
                border-radius: 4px; cursor: pointer; font-size: 0.75em; transition: 0.2s;
            }}
            .tf-btn.active {{ background: #238636; color: white; border-color: #2ea043; }}
            .tf-btn:hover {{ border-color: #58a6ff; }}

            .chart-wrapper {{ height: 220px; position: relative; }}
            .price-chart-wrapper {{ height: 350px; position: relative; }}
            
            .footer {{ text-align: center; color: #8b949e; font-size: 0.8em; margin-top: 30px; border-top: 1px solid #30363d; padding: 10px; }}
            
            /* Modal / Zoom Styles */
            .modal {{
                display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(1, 4, 9, 0.95); z-index: 9999; justify-content: center; align-items: center;
                backdrop-filter: blur(8px);
            }}
            .modal-content {{
                background: #161b22; border: 1px solid #30363d; border-radius: 12px;
                width: 90%; height: 85%; padding: 25px; position: relative;
                box-shadow: 0 0 40px rgba(0,0,0,0.6); display: flex; flex-direction: column;
            }}
            .close-modal {{
                position: absolute; top: 15px; right: 25px; color: #8b949e;
                font-size: 35px; cursor: pointer; transition: 0.2s;
            }}
            .close-modal:hover {{ color: #f85149; }}
            .modal-chart-container {{ flex-grow: 1; min-height: 0; width: 100%; }}
            
            .card:hover .zoom-icon {{ opacity: 1; }}

            @media (max-width: 900px) {{ .top-row, .row-4-cols {{ grid-template-columns: 1fr 1fr; }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1 style="color:#58a6ff; margin:0;">Forex Dashboard PRO</h1>
                <p style="color:#8b949e; margin:5px 0 0 0;">Análisis Pro • Radar Técnico Multi-TF</p>
            </header>

            <div class="controls">
                <strong>Activo Principal:</strong>
                <select id="pairSelector" onchange="updateCharts()"></select>
                <span style="color:#8b949e; font-size: 0.85em;">(Sincronización de Precio y Estacionalidad)</span>
            </div>
            
            <!-- Fila 1: Precio Interactivo + Correlaciones -->
            <div class="top-row">
                <div class="card" onclick="expandChart('chartPrice', 'Evolución de Precio')">
                    <span class="zoom-icon">🔍</span>
                    <h2>Evolución de Precio</h2>
                    <div class="tf-buttons">
                        <button class="tf-btn" onclick="event.stopPropagation(); setTF('1h')">1H</button>
                        <button class="tf-btn" onclick="event.stopPropagation(); setTF('4h')">4H</button>
                        <button class="tf-btn active" onclick="event.stopPropagation(); setTF('1d')">1D</button>
                        <button class="tf-btn" onclick="event.stopPropagation(); setTF('1w')">1W</button>
                    </div>
                    <div class="price-chart-wrapper"><canvas id="chartPrice"></canvas></div>
                </div>
                <div class="card"><h2>Ranking Correlaciones</h2><img src="data:image/png;base64,{img_ranking}"></div>
                <div class="card"><h2>Matriz Global</h2><img src="data:image/png;base64,{img_corr}"></div>
            </div>

            <!-- Fila de Volatilidad, Rangos y DATOS RELEVANTES (4 cols) -->
            <div class="row-4-cols">
                <div class="card"><h2>Volatilidad / Día Mes</h2><img src="data:image/png;base64,{img_dom}"></div>
                <div class="card"><h2>Rango Diario (%)</h2><img src="data:image/png;base64,{img_range}"></div>
                <div class="card"><h2>Rango Mensual (%)</h2><img src="data:image/png;base64,{img_m_range}"></div>
                <div class="card" style="cursor: default;">
                    <h2>Resumen: <span id="statPairName">---</span></h2>
                    <div id="statsContent" style="font-size: 0.8em; color: #8b949e;">
                        <table style="width:100%; border-collapse: collapse; margin-top:2px;">
                            <tr style="border-bottom: 1px solid #30363d;"><td style="padding:3px 0;">Precio Actual</td><td id="s_price" style="text-align:right; color:#c9d1d9; font-weight:bold;">0.00</td></tr>
                            <tr style="border-bottom: 1px solid #30363d;"><td style="padding:3px 0;">Sesgo (5d)</td><td id="s_bias_5d" style="text-align:right; font-weight:bold;">---</td></tr>
                            <tr style="border-bottom: 1px solid #30363d;"><td style="padding:3px 0;">Sesgo (20d)</td><td id="s_bias_20d" style="text-align:right; font-weight:bold;">---</td></tr>
                            <tr style="border-bottom: 1px solid #30363d;"><td style="padding:3px 0;">Max Correlación</td><td id="s_max_c" style="text-align:right; color:#3fb950;">---</td></tr>
                            <tr style="border-bottom: 1px solid #30363d;"><td style="padding:3px 0;">Min Correlación</td><td id="s_min_c" style="text-align:right; color:#f85149;">---</td></tr>
                            <tr style="border-bottom: 1px solid #30363d;"><td style="padding:3px 0;">Día Top Vol (Mes)</td><td id="s_best_dom" style="text-align:right; color:#ff9f43;">---</td></tr>
                            <tr style="border-bottom: 1px solid #30363d;"><td style="padding:2px 0;">Día Top Vol (Sem)</td><td id="s_best_dow" style="text-align:right; color:#ff9f43;">---</td></tr>
                            <tr style="border-bottom: 1px solid #30363d;"><td style="padding:2px 0;">Rango Anual %</td><td id="s_y_range" style="text-align:right; color:#58a6ff; font-weight:bold;">---</td></tr>
                            <tr style="border-bottom: 1px solid #30363d;"><td style="padding:2px 0;">Rango Mensual %</td><td id="s_m_range" style="text-align:right; color:#a5d6ff; font-weight:bold;">---</td></tr>
                            <tr style="border-bottom: 1px solid #30363d;"><td style="padding:2px 0;">Media Diaria</td><td id="s_avg_d" style="text-align:right; color:#c9d1d9;">0%</td></tr>
                            <tr style="border-bottom: 1px solid #30363d;"><td style="padding:2px 0;">Media Pips Diarios</td><td id="s_avg_pips" style="text-align:right; color:#ff9f43; font-weight:bold;">0</td></tr>
                            <tr style="border-bottom: 1px solid #30363d;"><td style="padding:2px 0;">Media Pips Semanales</td><td id="s_avg_pips_w" style="text-align:right; color:#a371f7; font-weight:bold;">0</td></tr>
                            <tr style="border-bottom: 2px solid #58a6ff; background: rgba(88, 166, 255, 0.05);"><td style="padding:4px 0; color:#58a6ff; font-weight:bold;">PROB. BULLISH MES</td><td id="s_prob_up" style="text-align:right; color:#3fb950; font-weight:bold; font-size:1.1em;">0%</td></tr>
                            <tr style="border-bottom: 1px solid #30363d; background: rgba(248, 81, 73, 0.05);"><td style="padding:4px 0; color:#f85149; font-weight:bold;">PROB. BEARISH MES</td><td id="s_prob_down" style="text-align:right; color:#f85149; font-weight:bold; font-size:1.1em;">0%</td></tr>
                            <tr><td colspan="2" style="padding-top:6px; padding-bottom:2px; font-size:0.85em; color:#58a6ff; letter-spacing:1px;">▶ FORECAST ARIMA (5D)</td></tr>
                            <tr style="border-bottom: 1px solid #30363d;"><td style="padding:2px 0;">Precio Objetivo (5d)</td><td id="s_forecast_5d" style="text-align:right; font-weight:bold; font-size:0.95em;">---</td></tr>
                            <tr><td style="padding:2px 0;">Potencial %</td><td id="s_forecast_pct" style="text-align:right; font-weight:bold;">---</td></tr>
                        </table>
                    </div>
                </div>
            </div>

            <div class="row-4-cols">
                <div class="card" onclick="expandChart('chartDom', 'Volatilidad / Mes (%)')">
                    <span class="zoom-icon">🔍</span>
                    <h2>Volatilidad / Mes (%)</h2>
                    <div class="chart-wrapper"><canvas id="chartDom"></canvas></div>
                </div>
                <div class="card" onclick="expandChart('chartDow', 'Volatilidad / Sem (%)')">
                    <span class="zoom-icon">🔍</span>
                    <h2>Volatilidad / Sem (%)</h2>
                    <div class="chart-wrapper"><canvas id="chartDow"></canvas></div>
                </div>
                <div class="card" onclick="expandChart('chartPips', 'Pips / Mes')">
                    <span class="zoom-icon">🔍</span>
                    <h2>Pips / Mes</h2>
                    <div class="chart-wrapper"><canvas id="chartPips"></canvas></div>
                </div>
                <div class="card" onclick="expandChart('chartPipsDow', 'Pips / Semana')">
                    <span class="zoom-icon">🔍</span>
                    <h2>Pips / Semana</h2>
                    <div class="chart-wrapper"><canvas id="chartPipsDow"></canvas></div>
                </div>
            </div>

            <div class="row-4-cols">
                <div class="card" onclick="expandChart('chartMarkov', 'Matriz de Transición (Markov)')">
                    <span class="zoom-icon">🔍</span>
                    <h2 style="margin-bottom:2px;">Matriz Markov</h2>
                    <p style="font-size:9px; color:#8b949e; margin-bottom:10px; line-height:1.2;">Probabilidad de que el estado actual cambie mañana.</p>
                    <div class="chart-wrapper"><canvas id="chartMarkov"></canvas></div>
                </div>
                <div class="card" onclick="expandChart('chartScatter', 'DISPERSIÓN')">
                    <span class="zoom-icon">🔍</span>
                    <h2>DISPERSIÓN</h2>
                    <div class="chart-wrapper"><div id="chartScatter" style="width:100%; height:100%;"></div></div>
                </div>
                <div class="card" onclick="expandChart('chartKernelHist', 'Kernel Regression (1Y)')">
                    <span class="zoom-icon">🔍</span>
                    <h2>Kernel Regression (1Y)</h2>
                    <div class="chart-wrapper"><canvas id="chartKernelHist"></canvas></div>
                </div>
                <div class="card" onclick="expandChart('chartSeasonality', 'Estacionalidad Mensual')">
                    <span class="zoom-icon">🔍</span>
                    <h2>Estacionalidad Mensual</h2>
                    <div class="chart-wrapper"><canvas id="chartSeasonality"></canvas></div>
                </div>
            </div>

            <div class="row-4-cols">
                <div class="card" onclick="expandChart('chartSunburst', 'MAPA DE RENDIMIENTO')">
                    <span class="zoom-icon">🔍</span>
                    <h2>MAPA DE RENDIMIENTO</h2>
                    <div class="chart-wrapper"><div id="chartSunburst" style="width:100%; height:100%;"></div></div>
                </div>
                <div class="card" onclick="expandChart('chartSankey', 'FLUJO DE IMPACTO')">
                    <span class="zoom-icon">🔍</span>
                    <h2>FLUJO DE IMPACTO</h2>
                    <div class="chart-wrapper"><div id="chartSankey" style="width:100%; height:100%;"></div></div>
                </div>
                <div class="card" onclick="expandChart('chartBubble', 'EVOLUCIÓN DE LA VOLATILIDAD')">
                    <span class="zoom-icon">🔍</span>
                    <h2>EVOLUCIÓN VOLATILIDAD</h2>
                    <div class="chart-wrapper"><div id="chartBubble" style="width:100%; height:100%;"></div></div>
                </div>
                <div class="card" onclick="expandChart('chartSurface', 'SUPERFICIE DE VOLATILIDAD')">
                    <span class="zoom-icon">🔍</span>
                    <h2>SUPERFICIE VOLATILIDAD</h2>
                    <div class="chart-wrapper"><div id="chartSurface" style="width:100%; height:100%;"></div></div>
                </div>
            </div>

            <div class="row-4-cols">
                <div class="card" onclick="expandChart('chartGBM', 'GBM - Caminos Aleatorios')">
                    <span class="zoom-icon">🔍</span>
                    <h2>GBM (Mov. Browniano)</h2>
                    <div class="chart-wrapper"><div id="chartGBM" style="width:100%; height:100%;"></div></div>
                </div>
                <div class="card" onclick="expandChart('chartGARCH', 'GARCH - Volatilidad Estocástica')">
                    <span class="zoom-icon">🔍</span>
                    <h2>GARCH (Vol. Estocástica)</h2>
                    <div class="chart-wrapper"><div id="chartGARCH" style="width:100%; height:100%;"></div></div>
                </div>
                <div class="card" onclick="expandChart('chartMerton', 'Jump-Diffusion / Heston')">
                    <span class="zoom-icon">🔍</span>
                    <h2>Jump-Diffusion / Merton</h2>
                    <div class="chart-wrapper"><div id="chartMerton" style="width:100%; height:100%;"></div></div>
                </div>
                <div class="card" id="consensusCard">
                    <h2 style="color:#58a6ff; margin-bottom:10px; border-bottom:1px solid #30363d; padding-bottom:5px;">CONSENSO ESTRATÉGICO</h2>
                    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; text-align:center;">
                        <div style="background:rgba(88,166,255,0.05); padding:12px; border-radius:6px; width:90%; border:1px solid rgba(88,166,255,0.1);">
                            <p id="c_summary" style="font-size:0.95em; color:#c9d1d9; line-height:1.5; font-weight:400; font-style:italic;">Calculando factores...</p>
                        </div>
                    </div>
                </div>
            </div>

            <footer class="footer">Dashboard Analítico Elite • {datetime.now().strftime('%H:%M')}</footer>
        </div>

        <div id="chartModal" class="modal">
            <div class="modal-content">
                <span class="close-modal" onclick="closeModal()">&times;</span>
                <h2 id="modalTitle" style="font-size: 1.4em; border-left: 4px solid #58a6ff; padding-left: 10px;">KPI Detallada</h2>
                <div class="modal-chart-container">
                    <canvas id="modalChart"></canvas>
                    <div id="modalPlotly" style="width:100%; height:100%; display:none;"></div>
                </div>
            </div>
        </div>

        <script>
            const dataDom = {vol_data_json};
            const dataDow = {vol_dow_json};
            const dataPips = {pips_data_json};
            const dataPipsDow = {pips_dow_json};
            const dataStrength = {dow_strength_json};
            const scatterData = {scatter_data_json};
            const kernelHist = {kernel_history_json};
            const seasonalData = {monthly_seasonality_json};
            const pricePack = {price_pack_json};
            const pairStats = {pair_stats_json};
            const markovData = {markov_json};
            const monthlyLevels = {monthly_levels_json};
            const marketHierarchy = {market_hierarchy_json};
            const marketSankey = {market_sankey_json};
            const bubbleData = {pair_bubble_json};
            const surfaceData = {pair_surface_json};
            const gbmData = {gbm_json};
            const garchData = {garch_json};
            const mertonData = {merton_json};
            
            const pairs = Object.keys(dataDom).sort();
            const selector = document.getElementById('pairSelector');
            let currentTF = '1d';
            let chartDom, chartDow, chartPips, chartPipsDow, chartPrice, chartMarkov, chartScatter, chartKernelHist, chartSeasonality, 
                chartGBM, chartGARCH, chartMerton, modalChart;
            
            // Plugin para dibujar etiquetas en las líneas horizontales de rango
            const horizontalLineLabelsPlugin = {{
                id: 'horizontalLineLabels',
                afterDraw(chart) {{
                    const {{ ctx, chartArea: {{ left }}, scales: {{ y }} }} = chart;
                    const p = selector.value;
                    const levels = monthlyLevels[p];
                    if (!levels) return;

                    chart.data.datasets.forEach((dataset) => {{
                        if (dataset.label === 'Máx Mes' || dataset.label === 'Mín Mes') {{
                            const val = dataset.data[0];
                            const yPos = y.getPixelForValue(val);
                            if (yPos < chart.chartArea.top || yPos > chart.chartArea.bottom) return;

                            ctx.save();
                            const isHigh = dataset.label === 'Máx Mes';
                            const pct = isHigh ? levels.avg_up : levels.avg_down;
                            const sign = pct > 0 ? '+' : '';
                            const text = `${{isHigh ? 'MAX' : 'MIN'}}: ${{val.toFixed(5)}} (${{sign}}${{pct}}%)`;
                            
                            ctx.font = 'bold 11px monospace';
                            const textWidth = ctx.measureText(text).width;
                            ctx.fillStyle = 'rgba(13, 17, 23, 0.9)';
                            ctx.fillRect(left + 2, yPos - 12, textWidth + 8, 14);
                            ctx.fillStyle = dataset.borderColor;
                            ctx.fillText(text, left + 6, yPos - 1);
                            ctx.restore();
                        }}
                    }});
                }}
            }};
            Chart.register(horizontalLineLabelsPlugin);

            const chartMap = {{
                'chartPrice': () => chartPrice,
                'chartDom': () => chartDom,
                'chartDow': () => chartDow,
                'chartPips': () => chartPips,
                'chartPipsDow': () => chartPipsDow,
                'chartStrength': () => chartStrength,
                'chartScatter': () => 'plotly',
                'chartKernelHist': () => chartKernelHist,
                'chartSeasonality': () => chartSeasonality,
                'chartGBM': () => 'plotly',
                'chartGARCH': () => 'plotly',
                'chartMerton': () => 'plotly',
                'chartMarkov': () => chartMarkov,
                'chartSunburst': () => 'plotly',
                'chartSankey': () => 'plotly',
                'chartBubble': () => 'plotly',
                'chartSurface': () => 'plotly'
            }};

            function expandChart(id, title) {{
                lastExpandedId = id;
                const p = selector.value;
                const modal = document.getElementById('chartModal');
                const modalTitle = document.getElementById('modalTitle');
                const modalCanvas = document.getElementById('modalChart');
                const modalPlotly = document.getElementById('modalPlotly');
                
                modalTitle.innerText = title;
                modal.style.display = 'flex';
                
                if (modalChart) {{ modalChart.destroy(); modalChart = null; }}
                modalCanvas.style.display = 'none';
                modalPlotly.style.display = 'none';

                if (id === 'chartScatter' || id === 'chartSunburst' || id === 'chartSankey' || id === 'chartBubble' || id === 'chartSurface' || 
                    id === 'chartGBM' || id === 'chartGARCH' || id === 'chartMerton') {{
                    modalPlotly.style.display = 'block';
                    
                    if (id === 'chartGBM' || id === 'chartGARCH' || id === 'chartMerton') {{
                        const d = (id === 'chartGBM') ? gbmData[p] : (id === 'chartGARCH' ? garchData[p] : mertonData[p]);
                        if (!d || !d.median || d.median.every(v => v === 0)) return;
                        const color = 'rgb(31, 111, 235)';
                        const labels = Array.from({{length: 30}}, (_, i) => "D+" + (i + 1));
                        Plotly.newPlot('modalPlotly', [
                            {{ x: labels, y: d.median, name: 'Mediana', line: {{ color: color, width: 3 }}, type: 'scatter', mode: 'lines+markers' }},
                            {{ x: labels, y: d.p95, name: 'P95', line: {{ color: 'transparent' }}, type: 'scatter', mode: 'lines' }},
                            {{ x: labels, y: d.p5, name: 'P5', line: {{ color: 'transparent' }}, type: 'scatter', mode: 'lines', fill: 'tonexty', fillcolor: color.replace('rgb', 'rgba').replace(')', ', 0.2)') }}
                        ], {{
                            paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
                            margin: {{ l: 50, r: 20, b: 50, t: 30 }},
                            xaxis: {{ gridcolor: '#30363d', color: '#c9d1d9', title: 'Días' }},
                            yaxis: {{ gridcolor: '#30363d', color: '#c9d1d9' }},
                            font: {{ color: '#c9d1d9' }}, hovermode: 'x unified', showlegend: true
                        }});
                    }} else if (id === 'chartSurface') {{
                        Plotly.newPlot('modalPlotly', [{{
                            z: surfaceData[p], x: [0, 1, 2, 3, 4], type: 'surface', colorscale: 'Viridis'
                        }}], {{ 
                            scene: {{ 
                                xaxis: {{ title: 'Día', ticktext: ['LUN','MAR','MIE','JUE','VIE'], tickvals: [0,1,2,3,4], color: '#8b949e' }},
                                yaxis: {{ title: 'Hora', ticktext: ['0h','6h','12h','18h'], tickvals: [0,6,12,18], color: '#8b949e' }},
                                zaxis: {{ title: 'Volatilidad', color: '#8b949e' }},
                                bgcolor: '#161b22' 
                            }}, 
                            paper_bgcolor: '#161b22', margin: {{ l: 0, r: 0, b: 0, t: 0 }} 
                        }});
                    }} else if (id === 'chartScatter') {{
                        const hist = scatterData[p];
                        Plotly.newPlot('modalPlotly', [{{
                            x: hist.x, y: hist.y, z: hist.z, mode: 'markers', marker: {{ size: 6, color: hist.colors }}, text: hist.text, type: 'scatter3d'
                        }}], {{ scene: {{ bgcolor: '#161b22' }}, paper_bgcolor: '#161b22', margin: {{ l: 0, r: 0, b: 0, t: 0 }} }});
                    }} else if (id === 'chartSunburst') {{
                        Plotly.newPlot('modalPlotly', [{{
                            type: "sunburst", ids: marketHierarchy.ids, labels: marketHierarchy.labels, parents: marketHierarchy.parents, values: marketHierarchy.values,
                            marker: {{ colors: marketHierarchy.colors, colorscale: "RdYlGn", cmid: 0 }}, text: marketHierarchy.hovers, hoverinfo: "text", branchvalues: "remainder"
                        }}], {{ margin: {{l: 0, r: 0, b: 0, t: 0}}, paper_bgcolor: '#161b22', font: {{color: '#c9d1d9'}} }});
                    }} else if (id === 'chartSankey') {{
                        Plotly.newPlot('modalPlotly', [{{
                            type: "sankey", node: {{ pad: 15, thickness: 20, label: marketSankey.labels, color: "#58a6ff" }},
                            link: {{ source: marketSankey.sources, target: marketSankey.targets, value: marketSankey.values, color: marketSankey.colors }}
                        }}], {{ paper_bgcolor: '#161b22', font: {{ color: '#c9d1d9' }}, margin: {{ l: 10, r: 10, b: 10, t: 10 }} }});
                    }} else if (id === 'chartBubble') {{
                        const b = bubbleData[p];
                        Plotly.newPlot('modalPlotly', [{{
                            x: b.x, y: b.y, mode: 'markers', marker: {{ size: b.size.map(s => s*1.5), color: b.color, opacity: 0.6 }}, text: b.text, hoverinfo: 'text'
                        }}], {{ paper_bgcolor: '#161b22', plot_bgcolor: '#161b22', xaxis: {{ color: '#8b949e' }}, yaxis: {{ color: '#8b949e' }}, font: {{ color: '#c9d1d9' }} }});
                    }}
                    return;
                }}

                const sourceChart = chartMap[id] ? chartMap[id]() : null;
                if (!sourceChart) return;
                modalCanvas.style.display = 'block';
                const ctxM = modalCanvas.getContext('2d');
                
                const ttc = (sourceChart.config.options.plugins && sourceChart.config.options.plugins.tooltip) ? 
                            sourceChart.config.options.plugins.tooltip.callbacks : null;

                modalChart = new Chart(ctxM, {{
                    type: sourceChart.config.type,
                    data: JSON.parse(JSON.stringify(sourceChart.config.data)),
                    options: {{
                        ...sourceChart.config.options,
                        maintainAspectRatio: false,
                        plugins: {{
                            ...sourceChart.config.options.plugins,
                            legend: {{ display: true, labels: {{ color: '#c9d1d9' }} }},
                            tooltip: {{
                                enabled: true, backgroundColor: 'rgba(22, 27, 34, 0.95)',
                                borderColor: '#58a6ff', borderWidth: 1, callbacks: ttc
                            }}
                        }}
                    }}
                }});
            }}

            function closeModal() {{
                document.getElementById('chartModal').style.display = 'none';
                if (modalChart) modalChart.destroy();
                Plotly.purge('modalPlotly');
            }}

            // Cerrar al clickar fuera
            window.onclick = function(event) {{
                const modal = document.getElementById('chartModal');
                if (event.target == modal) closeModal();
            }}

            pairs.forEach(p => {{
                let o = document.createElement('option'); o.value = p; o.innerHTML = p; selector.appendChild(o);
            }});

            function createMarkovChart(id, matrix) {{
                const ctx = document.getElementById(id).getContext('2d');
                return new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: ['Si HOY es UP', 'Si HOY es DOWN', 'Si HOY es FLAT'],
                        datasets: [
                            {{ label: 'Siguiente: UP', data: [matrix[0][0], matrix[1][0], matrix[2][0]], backgroundColor: '#3fb950' }},
                            {{ label: 'Siguiente: DOWN', data: [matrix[0][1], matrix[1][1], matrix[2][1]], backgroundColor: '#f85149' }},
                            {{ label: 'Siguiente: FLAT', data: [matrix[0][2], matrix[1][2], matrix[2][2]], backgroundColor: '#8b949e' }}
                        ]
                    }},
                    options: {{
                        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                        plugins: {{ 
                            legend: {{ display: true, position: 'bottom', labels: {{ color: '#c9d1d9', font: {{ size: 8 }} }} }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        const label = context.dataset.label || '';
                                        const value = (context.parsed.x * 100).toFixed(1) + '%';
                                        return label + ': ' + value;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e', font: {{ size: 8 }} }}, max: 1.0 }},
                            y: {{ grid: {{ display: false }}, ticks: {{ color: '#c9d1d9', font: {{ size: 9 }} }} }}
                        }}
                    }}
                }});
            }}

            function createConeChart(id, label, dataObj, color) {{
                const labels = Array.from({{length: 30}}, (_, i) => "D+" + (i + 1));
                const traces = [
                    {{ x: labels, y: dataObj.median, name: 'Mediana', line: {{ color: color, width: 2 }}, type: 'scatter', mode: 'lines' }},
                    {{ x: labels, y: dataObj.p95, name: 'P95', line: {{ color: 'transparent' }}, type: 'scatter', mode: 'lines' }},
                    {{ x: labels, y: dataObj.p5, name: 'P5', line: {{ color: 'transparent' }}, type: 'scatter', mode: 'lines', fill: 'tonexty', fillcolor: color.replace('rgb', 'rgba').replace(')', ', 0.15)') }}
                ];
                const layout = {{
                    margin: {{ l: 35, r: 10, b: 25, t: 10 }},
                    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', showlegend: false,
                    xaxis: {{ gridcolor: 'transparent', color: '#8b949e', tickfont: {{ size: 9 }}, nticks: 10 }},
                    yaxis: {{ gridcolor: '#30363d', color: '#8b949e', tickfont: {{ size: 9 }} }},
                    hovermode: 'x unified'
                }};
                Plotly.newPlot(id, dataObj.median ? traces : [], layout, {{ displayModeBar: false }});
                return id;
            }}

            function createChart(id, type, label, labels, initialData, color, unit) {{
                const ctx = document.getElementById(id).getContext('2d');
                return new Chart(ctx, {{
                    type: type,
                    data: {{
                        labels: labels,
                        datasets: [{{ 
                            label: label, 
                            data: initialData, 
                            backgroundColor: color, 
                            borderColor: color,
                            borderWidth: type === 'line' ? 2 : 1,
                            pointRadius: 0,
                            fill: type === 'line' ? false : true,
                            borderRadius: type === 'bar' ? 3 : 0
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{ mode: 'index', intersect: false }},
                        plugins: {{ 
                            legend: {{ display: false }},
                            tooltip: {{ 
                                enabled: true,
                                backgroundColor: 'rgba(22, 27, 34, 0.9)',
                                titleColor: '#58a6ff',
                                callbacks: {{ label: (c) => c.parsed.y + (unit || '') }} 
                            }}
                        }},
                        scales: {{ 
                            y: {{ beginAtZero: false, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e', font: {{ size: 10 }} }} }},
                            x: {{ grid: {{ display: false }}, ticks: {{ color: '#8b949e', font: {{ size: 10 }}, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }} }}
                        }}
                    }}
                }});
            }}

            function setTF(tf) {{
                currentTF = tf;
                document.querySelectorAll('.tf-btn').forEach(b => {{
                    b.classList.remove('active');
                    if(b.innerText === tf.toUpperCase()) b.classList.add('active');
                }});
                updatePriceChart();
            }}

            function updatePriceChart() {{
                const p = selector.value;
                const tfData = pricePack[currentTF];
                const levels = monthlyLevels[p];
                
                chartPrice.data.labels = tfData.labels;
                chartPrice.data.datasets[0].data = tfData.data[p];
                
                // Limpiar datasets extra si existen
                if (chartPrice.data.datasets.length > 1) {{
                    chartPrice.data.datasets.splice(1);
                }}

                if (levels) {{
                    // Líneas de proyección estadística histórica desde apertura del periodo
                    const n = tfData.labels.length;
                    chartPrice.data.datasets.push({{
                        label: 'Máx Mes',
                        data: Array(n).fill(levels.proj_high),
                        borderColor: '#39ff14',
                        borderWidth: 2,
                        borderDash: [8, 4],
                        pointRadius: 0,
                        fill: false
                    }});
                    chartPrice.data.datasets.push({{
                        label: 'Mín Mes',
                        data: Array(n).fill(levels.proj_low),
                        borderColor: '#ff0033',
                        borderWidth: 2,
                        borderDash: [8, 4],
                        pointRadius: 0,
                        fill: false
                    }});
                }}
                
                chartPrice.update();
            }}

            function initAll() {{
                const f = pairs[0];
                const initialPrice = pricePack[currentTF];
                chartPrice = createChart('chartPrice', 'line', 'Precio', initialPrice.labels, initialPrice.data[f], '#58a6ff', '');
                
                chartDom = createChart('chartDom', 'bar', 'Vol %', Object.keys(dataDom[f]).map(k=>"D"+k), Object.values(dataDom[f]), '#ff9f43', '%');
                chartDow = createChart('chartDow', 'bar', 'Vol Sem %', Object.keys(dataDow[f]), Object.values(dataDow[f]), '#58a6ff', '%');
                chartPips = createChart('chartPips', 'bar', 'Pips Mes', Object.keys(dataPips[f]).map(k=>"D"+k), Object.values(dataPips[f]), '#3fb950', 'Pips');
                chartPipsDow = createChart('chartPipsDow', 'bar', 'Pips Sem', Object.keys(dataPipsDow[f]), Object.values(dataPipsDow[f]), '#a371f7', 'Pips');
                
                
                // Gráfico 3D con Plotly
                const hist = scatterData[f];
                const trace = {{
                    x: hist.x,
                    y: hist.y,
                    z: hist.z,
                    mode: 'markers',
                    marker: {{
                        size: 4,
                        color: hist.colors,
                        opacity: 0.8
                    }},
                    text: hist.text,
                    hoverinfo: 'text',
                    type: 'scatter3d'
                }};

                const layout = {{
                    margin: {{ l: 0, r: 0, b: 0, t: 0 }},
                    scene: {{
                        xaxis: {{ title: 'Fecha', color: '#8b949e', gridcolor: '#30363d' }},
                        yaxis: {{ title: 'Precio', color: '#8b949e', gridcolor: '#30363d' }},
                        zaxis: {{ title: 'Retorno %', color: '#8b949e', gridcolor: '#30363d' }},
                        bgcolor: '#161b22'
                    }},
                    paper_bgcolor: '#161b22',
                    font: {{ color: '#8b949e', size: 10 }}
                }};
                Plotly.newPlot('chartScatter', [trace], layout);

                // Sunburst Global
                const traceSB = {{
                    type: "sunburst",
                    ids: marketHierarchy.ids,
                    labels: marketHierarchy.labels,
                    parents: marketHierarchy.parents,
                    values: marketHierarchy.values,
                    marker: {{
                        colors: marketHierarchy.colors,
                        colorscale: "RdYlGn",
                        cmid: 0
                    }},
                    text: marketHierarchy.hovers,
                    hoverinfo: "text",
                    branchvalues: "remainder"
                }};
                const layoutSB = {{
                    margin: {{l: 0, r: 0, b: 0, t: 0}},
                    paper_bgcolor: '#161b22',
                    font: {{color: '#8b949e', size: 10}}
                }};
                Plotly.newPlot('chartSunburst', [traceSB], layoutSB);

                // Sankey Global
                const traceSK = {{
                    type: "sankey",
                    node: {{
                        pad: 10, thickness: 15, line: {{ color: "black", width: 0.5 }},
                        label: marketSankey.labels, color: "#58a6ff"
                    }},
                    link: {{
                        source: marketSankey.sources, target: marketSankey.targets,
                        value: marketSankey.values, color: marketSankey.colors
                    }}
                }};
                Plotly.newPlot('chartSankey', [traceSK], {{
                    margin: {{ l: 5, r: 5, b: 5, t: 5 }},
                    paper_bgcolor: '#161b22', font: {{ color: '#8b949e', size: 8 }}
                }});

                // Bubble Case
                const b = bubbleData[f];
                const traceBB = {{
                    x: b.x, y: b.y, mode: 'markers',
                    marker: {{ size: b.size, color: b.color, opacity: 0.6 }},
                    text: b.text, hoverinfo: 'text'
                }};
                Plotly.newPlot('chartBubble', [traceBB], {{
                    margin: {{ l: 30, r: 10, b: 30, t: 10 }},
                    paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
                    xaxis: {{ showgrid: false, color: '#8b949e', ticktext: b.dates, tickvals: b.x }},
                    yaxis: {{ gridcolor: '#30363d', color: '#8b949e' }},
                    font: {{ size: 8 }}
                }});

                // Surface 3D
                Plotly.newPlot('chartSurface', [{{
                    z: surfaceData[f], type: 'surface', colorscale: 'Viridis', showscale: false
                }}], {{
                    margin: {{ l: 0, r: 0, b: 0, t: 0 }},
                    scene: {{
                        xaxis: {{ title: 'Día', ticktext: ['L','M','X','J','V'], tickvals: [0,1,2,3,4], color: '#8b949e', font: {{size: 8}} }},
                        yaxis: {{ title: 'Hora', ticktext: ['0h','12h','23h'], tickvals: [0,12,23], color: '#8b949e', font: {{size: 8}} }},
                        zaxis: {{ visible: false }},
                        bgcolor: '#161b22'
                    }},
                    paper_bgcolor: '#161b22'
                }});

                const kh = kernelHist[f];
                const ctxK = document.getElementById('chartKernelHist').getContext('2d');
                chartKernelHist = new Chart(ctxK, {{
                    type: 'line',
                    data: {{
                        labels: kh.labels,
                        datasets: [{{
                            label: 'Precio',
                            data: kh.price,
                            borderColor: '#58a6ff',
                            borderWidth: 1.5,
                            pointRadius: 0,
                            fill: false
                        }}, {{
                            label: 'Kernel Reg.',
                            data: kh.kernel,
                            borderColor: '#ffcc00',
                            borderWidth: 3,
                            pointRadius: 0,
                            fill: false,
                            borderDash: [5, 5]
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{ mode: 'index', intersect: false }},
                        plugins: {{
                            legend: {{ display: false }},
                            tooltip: {{
                                enabled: true,
                                backgroundColor: 'rgba(22, 27, 34, 0.9)',
                                titleColor: '#58a6ff',
                                callbacks: {{ label: (c) => c.dataset.label + ": " + c.parsed.y.toFixed(5) }}
                            }}
                        }},
                        scales: {{
                            x: {{ grid: {{ display: false }}, ticks: {{ color: '#8b949e', font: {{ size: 9 }}, maxTicksLimit: 12 }} }},
                            y: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }}
                        }}
                    }}
                }});
                
                
                const months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'];
                const ctxSeasonal = document.getElementById('chartSeasonality').getContext('2d');
                chartSeasonality = new Chart(ctxSeasonal, {{
                    type: 'bar',
                    data: {{
                        labels: months,
                        datasets: [{{
                            label: 'Retorno % Medio',
                            data: seasonalData[f],
                            backgroundColor: seasonalData[f].map(v => v >= 0 ? 'rgba(63, 185, 80, 0.6)' : 'rgba(248, 81, 73, 0.6)'),
                            borderRadius: 4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }},
                            tooltip: {{ callbacks: {{ label: (c) => "Avg: " + c.parsed.y + "%" }} }}
                        }},
                        scales: {{
                            x: {{ grid: {{ display: false }}, ticks: {{ color: '#8b949e' }} }},
                            y: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }}, title: {{ display:true, text: 'Retorno %', color: '#8b949e' }} }}
                        }}
                    }}
                }});
                
                updateStats();
                updatePriceChart();
                
                try {{
                    chartGBM = (gbmData && gbmData[f]) ? createConeChart('chartGBM', 'GBM', gbmData[f], 'rgb(31, 111, 235)') : null;
                    chartGARCH = (garchData && garchData[f]) ? createConeChart('chartGARCH', 'GARCH', garchData[f], 'rgb(31, 111, 235)') : null;
                    chartMerton = (mertonData && mertonData[f]) ? createConeChart('chartMerton', 'Jump-Merton', mertonData[f], 'rgb(31, 111, 235)') : null;
                    chartMarkov = (markovData && markovData[f]) ? createMarkovChart('chartMarkov', markovData[f]) : null;
                }} catch(e) {{ console.error("Error initializing advanced charts:", e); }}
            }}

            function updateStats() {{
                const p = selector.value;
                if (!pairStats || !pairStats[p]) return;
                const s = pairStats[p];
                document.getElementById('statPairName').innerText = p;
                document.getElementById('s_price').innerText = s.price;
                
                document.getElementById('s_bias_5d').innerText = s.bias_5d;
                document.getElementById('s_bias_5d').style.color = s.bias_5d_color;
                
                document.getElementById('s_bias_20d').innerText = s.bias_20d;
                document.getElementById('s_bias_20d').style.color = s.bias_20d_color;
                
                document.getElementById('s_max_c').innerText = s.max_corr;
                document.getElementById('s_min_c').innerText = s.min_corr;
                document.getElementById('s_best_dom').innerText = s.best_dom;
                document.getElementById('s_best_dow').innerText = s.best_dow;
                document.getElementById('s_y_range').innerText = s.yearly_range;
                document.getElementById('s_m_range').innerText = s.monthly_range;
                document.getElementById('s_avg_d').innerText = s.avg_daily;
                document.getElementById('s_avg_pips').innerText = s.avg_pips_daily;
                document.getElementById('s_avg_pips_w').innerText = s.avg_pips_week;
                document.getElementById('s_prob_up').innerText = s.p_up;
                document.getElementById('s_prob_down').innerText = s.p_down;
                
                document.getElementById('s_forecast_5d').innerHTML = '<span style="color:#3fb950">' + s.f_h + '</span> / <span style="color:#f85149">' + s.f_l + '</span>';
                document.getElementById('s_forecast_pct').innerHTML = '<span style="color:#3fb950">' + s.f_h_p + '</span> / <span style="color:#f85149">' + s.f_l_p + '</span>';
                
                // Consensus UI
                const sig = document.getElementById('c_signal');
                if(sig) {{
                    sig.innerText = s.c_signal;
                    sig.style.color = s.c_color;
                    sig.style.borderColor = s.c_color;
                    if(s.c_color.startsWith('#')) {{
                        const r = parseInt(s.c_color.slice(1,3), 16), g = parseInt(s.c_color.slice(3,5), 16), b = parseInt(s.c_color.slice(5,7), 16);
                        sig.style.backgroundColor = `rgba(${{r}}, ${{g}}, ${{b}}, 0.1)`;
                    }}
                }}
                document.getElementById('c_summary').innerText = s.c_summary;
            }}

            function updateCharts() {{
                const p = selector.value;
                if (!p) return;
                
                updatePriceChart();
                updateStats();
                
                if (dataDom[p]) {{ chartDom.data.datasets[0].data = Object.values(dataDom[p]); chartDom.update(); }}
                if (dataDow[p]) {{ chartDow.data.datasets[0].data = Object.values(dataDow[p]); chartDow.update(); }}
                if (dataPips[p]) {{ chartPips.data.datasets[0].data = Object.values(dataPips[p]); chartPips.update(); }}
                if (dataPipsDow[p]) {{ chartPipsDow.data.datasets[0].data = Object.values(dataPipsDow[p]); chartPipsDow.update(); }}
                
                if (markovData && markovData[p] && chartMarkov) {{
                    chartMarkov.data.datasets[0].data = [markovData[p][0][0], markovData[p][1][0], markovData[p][2][0]];
                    chartMarkov.data.datasets[1].data = [markovData[p][0][1], markovData[p][1][1], markovData[p][2][1]];
                    chartMarkov.data.datasets[2].data = [markovData[p][0][2], markovData[p][1][2], markovData[p][2][2]];
                    chartMarkov.update();
                }}
                
                ['chartGBM', 'chartGARCH', 'chartMerton'].forEach(id => {{
                    const d = (id === 'chartGBM') ? gbmData[p] : (id === 'chartGARCH' ? garchData[p] : mertonData[p]);
                    if (d) {{
                        const color = 'rgb(31, 111, 235)';
                        const lbls = Array.from({{length: 30}}, (_, i) => "D+" + (i + 1));
                        Plotly.react(id, [
                            {{ x: lbls, y: d.median, name: 'Mediana', line: {{ color: color, width: 2 }}, type: 'scatter', mode: 'lines' }},
                            {{ x: lbls, y: d.p95, name: 'P95', line: {{ color: 'transparent' }}, type: 'scatter', mode: 'lines' }},
                            {{ x: lbls, y: d.p5, name: 'P5', line: {{ color: 'transparent' }}, type: 'scatter', mode: 'lines', fill: 'tonexty', fillcolor: color.replace('rgb', 'rgba').replace(')', ', 0.15)') }}
                        ], {{
                            margin: {{ l: 35, r: 10, b: 25, t: 10 }},
                            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', showlegend: false,
                            xaxis: {{ gridcolor: 'transparent', color: '#8b949e', tickfont: {{ size: 9 }}, nticks: 10 }},
                            yaxis: {{ gridcolor: '#30363d', color: '#8b949e', tickfont: {{ size: 9 }} }},
                            hovermode: 'x unified'
                        }});
                    }}
                }});

                
                if (scatterData[p]) {{
                    const newHist = scatterData[p];
                    Plotly.react('chartScatter', [{{
                        x: newHist.x, y: newHist.y, z: newHist.z,
                        mode: 'markers', marker: {{ size: 4, color: newHist.colors, opacity: 0.8 }},
                        text: newHist.text, hoverinfo: 'text', type: 'scatter3d'
                    }}], {{
                        margin: {{ l: 0, r: 0, b: 0, t: 0 }},
                        scene: {{
                            xaxis: {{ title: 'Fecha', color: '#8b949e', gridcolor: '#30363d' }},
                            yaxis: {{ title: 'Precio', color: '#8b949e', gridcolor: '#30363d' }},
                            zaxis: {{ title: 'Retorno %', color: '#8b949e', gridcolor: '#30363d' }},
                            bgcolor: '#161b22'
                        }},
                        paper_bgcolor: '#161b22', font: {{ color: '#8b949e', size: 10 }}
                    }});
                }}

                if (kernelHist[p]) {{
                    const newKh = kernelHist[p];
                    chartKernelHist.data.labels = newKh.labels;
                    chartKernelHist.data.datasets[0].data = newKh.price;
                    chartKernelHist.data.datasets[1].data = newKh.kernel;
                    chartKernelHist.update();
                }}

                if (seasonalData[p]) {{
                    chartSeasonality.data.datasets[0].data = seasonalData[p];
                    chartSeasonality.data.datasets[0].backgroundColor = seasonalData[p].map(v => v >= 0 ? 'rgba(63, 185, 80, 0.6)' : 'rgba(248, 81, 73, 0.6)');
                    chartSeasonality.update();
                }}

                if (bubbleData[p]) {{
                    const nb = bubbleData[p];
                    Plotly.react('chartBubble', [{{
                        x: nb.x, y: nb.y, mode: 'markers',
                        marker: {{ size: nb.size, color: nb.color, opacity: 0.6 }},
                        text: nb.text, hoverinfo: 'text'
                    }}], {{
                        margin: {{ l: 30, r: 10, b: 30, t: 10 }},
                        paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
                        xaxis: {{ showgrid: false, color: '#8b949e', ticktext: nb.dates, tickvals: nb.x }},
                        yaxis: {{ gridcolor: '#30363d', color: '#8b949e' }},
                        font: {{ size: 8 }}
                    }});
                }}

                if (surfaceData[p]) {{
                    Plotly.react('chartSurface', [{{
                        z: surfaceData[p], type: 'surface', colorscale: 'Viridis', showscale: false
                    }}], {{
                        margin: {{ l: 0, r: 0, b: 0, t: 0 }},
                        scene: {{
                            xaxis: {{ title: 'Día', ticktext: ['L','M','X','J','V'], tickvals: [0,1,2,3,4], color: '#8b949e' }},
                            yaxis: {{ title: 'Hora', ticktext: ['0h','12h','23h'], tickvals: [0,12,23], color: '#8b949e' }},
                            zaxis: {{ visible: false }},
                            bgcolor: '#161b22'
                        }},
                        paper_bgcolor: '#161b22'
                    }});
                }}

                const modal = document.getElementById('chartModal');
                if (modal.style.display === 'flex') {{
                    const currentTitle = document.getElementById('modalTitle').innerText;
                    if (lastExpandedId) expandChart(lastExpandedId, currentTitle);
                }}
            }}

            let lastExpandedId = null;

            window.onload = initAll;
        </script>
    </body>
    </html>
    """
    html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "forex_dashboard.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"\nDashboard interactivo generado: {html_path}")

if __name__ == "__main__":
    analyze_forex_to_html()
