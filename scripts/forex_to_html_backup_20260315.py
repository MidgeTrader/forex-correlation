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
        data_close = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns else raw_data['Close']
        highs = raw_data['High']
        lows = raw_data['Low']
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
    prices_1h_all = raw_1h['Adj Close'] if 'Adj Close' in raw_1h.columns else raw_1h['Close']
    
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
        best_dom = vol_seasonality[col].idxmax()
        best_dow = vol_seasonality_dow[col].idxmax()
        
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
        
        # Últimos datos
        last_price = data_close[col].iloc[-1]
        last_range_pct = data_range[col].iloc[-1]
        last_range_pips = (highs[col].iloc[-1] - lows[col].iloc[-1]) * multiplier
        
        # Promedios
        m_daily_range = avg_range[col]
        m_monthly_range = avg_m_range[col]
        avg_pips = pips_df[col].mean()
        
        stats_dict[col] = {
            "price": f"{last_price:.4f}",
            "range_pct": f"{last_range_pct:.2f}%",
            "range_pips": f"{last_range_pips:.0f}",
            "avg_daily": f"{m_daily_range:.2f}%",
            "avg_monthly": f"{m_monthly_range:.2f}%",
            "avg_pips_daily": f"{avg_pips:.0f}",
            "max_corr": max_c_val,
            "min_corr": min_c_val,
            "best_dom": f"Día {best_dom}",
            "best_dow": f"{best_dow}",
            "bias_20d": bias_20d,
            "bias_20d_color": bias_20d_color,
            "bias_5d": bias_5d,
            "bias_5d_color": bias_5d_color,
            "yearly_range": f"+{max_y:.1f}% / {min_y:.1f}%",
            "monthly_range": f"+{max_m:.1f}% / {min_m:.1f}%",
            "avg_pips_week": f"{avg_pips_w:.0f}"
        }
    
    pair_stats_json = json.dumps(stats_dict)

    # JSONs de Volatilidad y Pips (Restaurados)
    vol_data_json = vol_seasonality.to_json(orient='columns')
    vol_dow_json = vol_seasonality_dow.to_json(orient='columns')
    pips_data_json = pips_seasonality.to_json(orient='columns')
    pips_dow_json = pips_seasonality_dow.to_json(orient='columns')

    # 8. Datos para Historial de Retornos Diarios (Evolución al Cierre) - Último Año
    returns_history = {}
    for col in data_close.columns:
        # Últimos 252 días
        rets_y = returns[col].tail(252)
        
        labels = [d.strftime('%d/%m/%y') for d in rets_y.index]
        data_pts = [round(float(r * 100), 3) if not pd.isna(r) else 0.0 for r in rets_y]
        
        returns_history[col] = {
            "labels": labels,
            "data": data_pts
        }
    
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

    html_template = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Forex Dashboard PRO</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 10px; }}
            .container {{ max-width: 1950px; margin: auto; }}
            header {{ text-align: center; padding: 10px 0; border-bottom: 1px solid #30363d; margin-bottom: 15px; }}
            
            /* Grid Principal */
            .row-4-cols {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 15px; }}
            
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

            @media (max-width: 1400px) {{ .top-row, .row-4-cols {{ grid-template-columns: 1fr 1fr; }} }}
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
                <div class="card" onclick="expandChart('chartStrength', 'Fuerza Semanal')">
                    <span class="zoom-icon">🔍</span>
                    <h2>Fuerza Semanal (Índice Corr.)</h2>
                    <div class="chart-wrapper"><canvas id="chartStrength"></canvas></div>
                </div>
                <div class="card" onclick="expandChart('chartScatter', 'Historial por Dispersión %')">
                    <span class="zoom-icon">🔍</span>
                    <h2>Dispersión %</h2>
                    <div class="chart-wrapper"><canvas id="chartScatter"></canvas></div>
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

            <footer class="footer">Dashboard Analítico Elite • {datetime.now().strftime('%H:%M')}</footer>
        </div>

        <div id="chartModal" class="modal">
            <div class="modal-content">
                <span class="close-modal" onclick="closeModal()">&times;</span>
                <h2 id="modalTitle" style="font-size: 1.4em; border-left: 4px solid #58a6ff; padding-left: 10px;">KPI Detallada</h2>
                <div class="modal-chart-container">
                    <canvas id="modalChart"></canvas>
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
            const monthlyLevels = {monthly_levels_json};
            
            const pairs = Object.keys(dataDom).sort();
            const selector = document.getElementById('pairSelector');
            let currentTF = '1d';
            let chartDom, chartDow, chartPips, chartPipsDow, chartPrice, chartStrength, chartScatter, chartKernelHist, chartSeasonality, modalChart;
            
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
                'chartScatter': () => chartScatter,
                'chartKernelHist': () => chartKernelHist,
                'chartSeasonality': () => chartSeasonality
            }};

            function expandChart(id, title) {{
                const modal = document.getElementById('chartModal');
                const modalTitle = document.getElementById('modalTitle');
                const sourceChart = chartMap[id] ? chartMap[id]() : null;
                
                if (!sourceChart) return;

                modalTitle.innerText = title;
                modal.style.display = 'flex';

                if (modalChart) modalChart.destroy();

                const ctxM = document.getElementById('modalChart').getContext('2d');
                
                // Asegurar que hay callbacks de tooltip, o usar uno por defecto
                const tooltipCallbacks = (sourceChart.config.options.plugins && 
                                          sourceChart.config.options.plugins.tooltip && 
                                          sourceChart.config.options.plugins.tooltip.callbacks) ? 
                                          sourceChart.config.options.plugins.tooltip.callbacks : 
                                          {{ label: (c) => c.dataset.label + ": " + c.parsed.y }};

                modalChart = new Chart(ctxM, {{
                    type: sourceChart.config.type,
                    data: sourceChart.config.data,
                    options: {{
                        ...sourceChart.config.options,
                        maintainAspectRatio: false,
                        interaction: {{
                            mode: 'index',
                            intersect: false,
                        }},
                        plugins: {{
                            ...sourceChart.config.options.plugins,
                            legend: {{ 
                                display: true, 
                                labels: {{ color: '#c9d1d9', font: {{ size: 14 }} }} 
                            }},
                            tooltip: {{
                                enabled: true,
                                backgroundColor: 'rgba(22, 27, 34, 0.95)',
                                titleColor: '#58a6ff',
                                titleFont: {{ size: 16, weight: 'bold' }},
                                bodyColor: '#c9d1d9',
                                bodyFont: {{ size: 14 }},
                                borderColor: '#58a6ff',
                                borderWidth: 1,
                                padding: 12,
                                boxPadding: 5,
                                cornerRadius: 8,
                                callbacks: tooltipCallbacks
                            }}
                        }}
                    }}
                }});
            }}

            function closeModal() {{
                document.getElementById('chartModal').style.display = 'none';
                if (modalChart) modalChart.destroy();
            }}

            // Cerrar al clickar fuera
            window.onclick = function(event) {{
                const modal = document.getElementById('chartModal');
                if (event.target == modal) closeModal();
            }}

            pairs.forEach(p => {{
                let o = document.createElement('option'); o.value = p; o.innerHTML = p; selector.appendChild(o);
            }});

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
                        plugins: {{ 
                            legend: {{ display: false }},
                            tooltip: {{ callbacks: {{ label: (c) => c.parsed.y.toFixed(4) + " " + unit }} }}
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
                
                chartStrength = createChart('chartStrength', 'line', 'Fuerza Corr', ['Lun', 'Mar', 'Mie', 'Jue', 'Vie'], dataStrength[f], '#ffcc00', '');
                
                const hist = scatterData[f];
                const ctxS = document.getElementById('chartScatter').getContext('2d');
                chartScatter = new Chart(ctxS, {{
                    type: 'line',
                    data: {{
                        labels: hist.labels,
                        datasets: [{{
                            label: 'Retorno %',
                            data: hist.data,
                            backgroundColor: hist.data.map(v => v >= 0 ? '#3fb950' : '#f85149'),
                            borderColor: 'transparent',
                            pointRadius: 4,
                            pointHoverRadius: 6,
                            showLine: false
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }},
                            tooltip: {{
                                callbacks: {{
                                    label: (c) => "Fecha: " + c.label + " | Cierre: " + c.parsed.y + "%"
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{ grid: {{ color: '#30363d', alpha: 0.1 }}, ticks: {{ color: '#8b949e', font: {{ size: 9 }}, maxTicksLimit: 12 }} }},
                            y: {{ title: {{ display: true, text: '% al Cierre', color: '#8b949e' }}, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }}
                        }}
                    }}
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
                            borderWidth: 1,
                            pointRadius: 0,
                            fill: false
                        }}, {{
                            label: 'Kernel Reg.',
                            data: kh.kernel,
                            borderColor: '#ffcc00',
                            borderWidth: 2,
                            pointRadius: 0,
                            fill: false,
                            borderDash: [5, 5]
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }},
                            tooltip: {{
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
            }}

            function updateStats() {{
                const p = selector.value;
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
            }}

            function updateCharts() {{
                const p = selector.value;
                updatePriceChart();
                updateStats();
                chartDom.data.datasets[0].data = Object.values(dataDom[p]); chartDom.update();
                chartDow.data.datasets[0].data = Object.values(dataDow[p]); chartDow.update();
                chartPips.data.datasets[0].data = Object.values(dataPips[p]); chartPips.update();
                chartPipsDow.data.datasets[0].data = Object.values(dataPipsDow[p]); chartPipsDow.update();
                chartStrength.data.datasets[0].data = dataStrength[p]; chartStrength.update();
                
                const newHist = scatterData[p];
                chartScatter.data.labels = newHist.labels;
                chartScatter.data.datasets[0].data = newHist.data;
                chartScatter.data.datasets[0].backgroundColor = newHist.data.map(v => v >= 0 ? '#3fb950' : '#f85149');
                chartScatter.update();

                const newKh = kernelHist[p];
                chartKernelHist.data.labels = newKh.labels;
                chartKernelHist.data.datasets[0].data = newKh.price;
                chartKernelHist.data.datasets[1].data = newKh.kernel;
                chartKernelHist.update();

                chartSeasonality.data.datasets[0].data = seasonalData[p];
                chartSeasonality.data.datasets[0].backgroundColor = seasonalData[p].map(v => v >= 0 ? 'rgba(63, 185, 80, 0.6)' : 'rgba(248, 81, 73, 0.6)');
                chartSeasonality.update();
            }}

            window.onload = initAll;
        </script>
    </body>
    </html>
    """

    with open("forex_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    print("\nDashboard interactivo generado: forex_dashboard.html")

if __name__ == "__main__":
    analyze_forex_to_html()
