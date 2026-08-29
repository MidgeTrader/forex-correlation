import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# Configuración de estilo premium
plt.style.use('dark_background')
sns.set_theme(style="dark", palette="viridis")
plt.rcParams['figure.facecolor'] = '#121212'
plt.rcParams['axes.facecolor'] = '#1e1e1e'
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['text.color'] = '#e0e0e0'
plt.rcParams['axes.labelcolor'] = '#b0b0b0'
plt.rcParams['xtick.color'] = '#888888'
plt.rcParams['ytick.color'] = '#888888'

def analyze_forex_correlations():
    # Crear carpeta de resultados en la raíz del proyecto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lista de símbolos proporcionada por el usuario
    raw_symbols = [
        "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "USDCAD", 
        "GBPUSD", "EURUSD", "AUDUSD", "NZDUSD", "EURGBP", "USDCHF", "EURCAD", 
        "SEKJPY", "SGDJPY", "EURAUD", "GBPCAD", "EURNZD", "CADCHF", "GBPCHF", 
        "AUDCAD", "AUDCHF", "CHFJPY", "NZDCHF", "ZARJPY", "EURNOK", "EURCHF", 
        "CHFSEK", "GBPNOK", "NZDNOK"
    ]
    
    # Formatear para Yahoo Finance
    symbols = [f"{s}=X" for s in raw_symbols]
    
    print(f"Descargando datos para {len(symbols)} pares de Forex...")
    
    # Descargar datos
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=2)
    
    try:
        # Usamos group_by='ticker' o simplemente seleccionamos el nivel Adj Close
        raw_data = yf.download(symbols, start=start_date, end=end_date, progress=False)
        
        if 'Adj Close' in raw_data.columns:
            data = raw_data['Adj Close']
        else:
            # Si solo hay un nivel de columna (raro con múltiples tickers pero posible si fallan)
            data = raw_data
            
        if data.empty:
            print("No se encontraron datos para los símbolos proporcionados.")
            return
            
    except Exception as e:
        print(f"Error al descargar datos: {e}")
        return

    print(f"Estructura de columnas descargadas: {data.columns[:5]}...")
    
    # Renombrar columnas para quitar el =X (ajustado para manejar si son tuplas o strings)
    new_cols = []
    for col in data.columns:
        if isinstance(col, tuple):
            name = col[-1] # Tomar el último elemento si es tupla
        else:
            name = col
        new_cols.append(name.replace('=X', ''))
    
    data.columns = new_cols
    
    # Eliminar posibles duplicados en columnas
    if data.columns.duplicated().any():
        print(f"Advertencia: Se detectaron columnas duplicadas: {data.columns[data.columns.duplicated()].unique()}")
        data = data.loc[:, ~data.columns.duplicated()]
    
    # Calcular retornos diarios (% de variación)
    returns = data.pct_change()
    
    print(f"Días de datos descargados: {len(data)}")
    print(f"Días con datos completos para TODOS los pares: {returns.dropna().shape[0]}")
    
    # 1. Matriz de Correlación Global
    # corr() por defecto usa pairwise deletion para manejar NaNs
    corr_matrix = returns.corr()
    
    if corr_matrix.isna().all().all():
        print("CRÍTICO: La matriz de correlación está vacía. Verificando datos...")
        print(returns.head())
        # Intentar rellenar huecos si es necesario
        returns = returns.ffill().dropna(how='all')
        corr_matrix = returns.corr()

    # Si aún falla, dar mensaje claro
    if corr_matrix.isna().all().all():
        print("Incapaz de calcular correlaciones. Verifique la conexión o los símbolos.")
        return

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                annot_kws={"size": 8},
                cbar_kws={'label': 'Coeficiente de Correlación'})
    plt.title("Matriz de Correlación de Forex (Retornos Diarios)", fontsize=18, pad=20)
    plt.savefig(os.path.join(output_dir, "correlacion_global.png"), dpi=300, bbox_inches='tight')

    # 2. Análisis por Día de la Semana
    returns['DayOfWeek'] = returns.index.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    avg_corrs = []
    for day in day_order:
        day_returns = returns[returns['DayOfWeek'] == day].drop(columns=['DayOfWeek'], errors='ignore')
        # Calculamos la correlación media de todos los pares entre sí para ese día de la semana
        day_corr_matrix = day_returns.corr()
        # Promedio del triángulo superior (sin la diagonal)
        matrix_values = day_corr_matrix.values[np.triu_indices_from(day_corr_matrix, k=1)]
        avg_corrs.append(np.nanmean(matrix_values))
    
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x=day_order, y=avg_corrs, palette="magma", hue=day_order, legend=False)
    plt.title("Fuerza de Correlación Promedio por Día de la Semana", fontsize=15)
    plt.ylabel("Correlación Inter-Pares Media")
    plt.axhline(np.nanmean(avg_corrs), color='cyan', linestyle='--', label=f'Media Global: {np.nanmean(avg_corrs):.3f}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "correlacion_por_dia_semana.png"), dpi=300)

    # 3. Análisis por Día del Mes
    returns['DayOfMonth'] = returns.index.day
    dom_corrs = []
    days_of_month = sorted(returns['DayOfMonth'].unique())
    
    for day in days_of_month:
        day_returns = returns[returns['DayOfMonth'] == day].drop(columns=['DayOfWeek', 'DayOfMonth'], errors='ignore')
        if len(day_returns) > 3:
            day_corr_matrix = day_returns.corr()
            matrix_values = day_corr_matrix.values[np.triu_indices_from(day_corr_matrix, k=1)]
            dom_corrs.append(np.nanmean(matrix_values))
        else:
            dom_corrs.append(np.nan)
            
    plt.figure(figsize=(14, 6))
    plt.plot(days_of_month, dom_corrs, marker='o', linestyle='-', color='#00d4ff', linewidth=2)
    plt.fill_between(days_of_month, dom_corrs, alpha=0.2, color='#00d4ff')
    plt.title("Evolución de la Correlación por Día del Mes", fontsize=15)
    plt.xlabel("Día del Mes")
    plt.ylabel("Correlación Media")
    plt.xticks(days_of_month)
    plt.grid(True, alpha=0.1)
    plt.savefig(os.path.join(output_dir, "correlacion_por_dia_mes.png"), dpi=300)

    # 4. Volatilidad (% de Variación Diaria)
    volatility = returns.drop(columns=['DayOfWeek', 'DayOfMonth'], errors='ignore').std() * 100
    plt.figure(figsize=(14, 8))
    vol_sorted = volatility.sort_values()
    vol_sorted.plot(kind='barh', color=sns.color_palette("viridis", len(vol_sorted)))
    plt.title("Volatilidad Diaria Promedio (Desviación Típica %)", fontsize=15)
    plt.xlabel("Volatilidad (%)")
    plt.savefig(os.path.join(output_dir, "volatilidad_pares.png"), dpi=300)

    # 5. Métricas y Ratios
    print("\n" + "="*50)
    print("RESUMEN DE MÉTRICAS Y VARIACIÓN")
    print("="*50)
    
    vol_df = pd.DataFrame({
        'Par': vol_sorted.index,
        'Volatilidad (%)': vol_sorted.values
    }).sort_values('Volatilidad (%)', ascending=False)
    
    print("\nTop 5 Pares Más Volátiles (Mayor Variación %):")
    print(vol_df.head(5).to_string(index=False))
    
    stacked_corr = corr_matrix.stack()
    stacked_corr = stacked_corr[stacked_corr < 0.99]
    
    print("\nTop 5 Pares Con Mayor Correlación Positiva:")
    print(stacked_corr.sort_values(ascending=False).head(10)[::2])
    
    print("\nTop 5 Pares Con Mayor Correlación Negativa:")
    print(stacked_corr.sort_values(ascending=True).head(10)[::2])
    
    # Análisis de Días Extremos
    # Calculamos la correlación media para cada FECHA individual
    daily_returns_clean = returns.drop(columns=['DayOfWeek', 'DayOfMonth'], errors='ignore')
    # Esto es complejo para un solo día, pero podemos ver la dispersión o usar una ventana móvil
    # Para simplificar, reportamos los mejores/peores por día de semana y mes
    print(f"\nDía de la SEMANA con MÁXIMA correlación: {day_order[np.nanargmax(avg_corrs)]} ({np.nanmax(avg_corrs):.3f})")
    print(f"Día de la SEMANA con MÍNIMA correlación: {day_order[np.nanargmin(avg_corrs)]} ({np.nanmin(avg_corrs):.3f})")
    
    valid_dom = [d for d in dom_corrs if not np.isnan(d)]
    valid_days = [days_of_month[i] for i, d in enumerate(dom_corrs) if not np.isnan(d)]
    
    print(f"Día del MES con MÁXIMA correlación: Día {valid_days[np.argmax(valid_dom)]} ({np.max(valid_dom):.3f})")
    print(f"Día del MES con MÍNIMA correlación: Día {valid_days[np.argmin(valid_dom)]} ({np.min(valid_dom):.3f})")
    
    # Guardar CSV de resultados
    corr_matrix.to_csv(os.path.join(output_dir, "matriz_correlacion.csv"))
    vol_df.to_csv(os.path.join(output_dir, "volatilidad_reporte.csv"), index=False)
    print(f"\nReportes generados en '{output_dir}':")
    print("- matriz_correlacion.csv")
    print("- volatilidad_reporte.csv")
    print("- 4 Gráficos PNG")



if __name__ == "__main__":
    analyze_forex_correlations()
