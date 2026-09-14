# Quant Dashboard (Opciones + FX + Macro + Tesoro)

Dashboard de análisis de correlaciones FX y activos macro, generado como un
**HTML estático autocontenido**: sin servidor, sin dependencias privadas, un
solo archivo que se abre en cualquier navegador. Clonar, instalar, generar.

Toda la barra de categorías en una pantalla — desde los iron condors sobre el
S&P 500 hasta la curva del Tesoro de EE.UU., pasando por el oro y el crudo —
con las métricas de cada mercado contadas con su propio lenguaje.

## Lo que ves

Una barra de categorías navega entre secciones. Cada sección es un grid de
cards, y cada card habla el idioma de su activo:

| Categoría | Activos | Qué miran sus cards |
|---|---|---|
| **XSP** | Mini-SPX (el S&P 500 ÷ 10): 0 DTE, 1 DTE y semanal | volatilidad realizada y VIX con su percentil, y una card por vencimiento con IV ATM, movimiento esperado 1σ derivado del *straddle*, **ATR(14)** con su banda y bandas 1σ/2σ con su distancia OTM; rangos históricos en % de día, semana y mes |
| **Metales** | Oro, Plata, Cobre, Platino, Paladio | Sharpe, Sortino, Beta, R², Alpha, probabilidades bootstrap, drawdown, momentum |
| **Energía** | Crudo WTI, Brent, Gas natural, Gasolina, Gasóleo | ídem Metales |
| **Índices** | S&P 500, Nasdaq 100, DAX, Nikkei 225, DXY | ídem Metales, con SPY de benchmark |
| **Agro** | Trigo, Maíz, Soja, Café, Cacao | ídem Metales |
| **Crypto** | BTC, ETH, BNB, XRP, SOL | benchmark **Nasdaq Crypto Index** (NCI), 365 días de negociación |
| **FX** | EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/GBP, EUR/JPY, GBP/JPY | forma de la distribución (skewness, kurtosis, tail ratio, VaR, CVaR), carry, Real Yield Differential, **régimen GMM**, probabilidades 12m |
| **Bonos** | Curva del Tesoro EE.UU.: 3M, 2Y, 5Y, 10Y, 30Y | una card por tramo con **precio teórico derivado del yield**, retornos reales del bono, y métricas de mesa: carry, roll-down, duration, DV01 |
| **Gráficas** | 15 pares + macro | dashboard interactivo (Chart.js/Plotly): correlaciones, volatilidad, **superficie de volatilidad** |

### La sección XSP, en detalle

Es la pestaña de trabajo diario: iron condors sobre el Mini-SPX en ciclo diario
y semanal, con una card por vencimiento (0 DTE, 1 DTE y semanal). Como el
subyacente es un índice, todo va en **% del precio** —la unidad con la que se
comparan entre sí las bandas y los rangos históricos— y los puntos se enseñan al
lado porque son los que se teclean en la orden.

Tres decisiones que sostienen las cifras:

- El movimiento esperado (1σ) **no** sale de la IV cruda de Yahoo, que es
  ruidosa en strikes ilíctos, sino del *straddle* ATM: `σ = straddle / 0.7979`.
- Los rangos históricos se normalizan por el **cierre del día anterior al
  periodo** —el movimiento contado desde donde se entra, con el gap dentro— y
  nunca por el precio de hoy: así una media de 5 años no se sesga por el nivel
  de precio.
- El **ATR(14)** se lee en el último periodo cerrado, con una excepción: el
  1 DTE incluye la sesión de hoy, porque esa posición vive el día entero y un
  día volátil tiene que notarse en el informe mientras se forma. La card lo
  declara en su fila de fecha (`↳ hoy` frente a `↳ cierre`).

### La sección de Bonos, en detalle

Los tramos se descargan del Tesoro vía FRED (`DGS3MO`, `DGS2`, `DGS5`,
`DGS10`, `DGS30`). Como la serie es el *yield* (no el precio), cada card deriva
internamente un **precio teórico** valorando un bono hipotético con cupón fijo
y vencimiento constante (convención *constant-maturity*). Sobre ese precio se
calculan los retornos reales — una subida de yield se ve como pérdida, que es
lo que le pasa al que tiene el bono — y las métricas de mesa que mira un rates
desk: carry, roll-down, duration y DV01 por $1M.

## Cómo generar el dashboard

Requisitos: Python 3.11+ (recomendado 3.12–3.14) y conexión a internet para
los datos (Yahoo Finance, FRED opcional).

```bash
# Opción 1 — setup en un paso (venv + dependencias + generar)
bash scripts/setup.sh

# Opción 2 — manual
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/generar_dashboard.py

# Forzar la re-descarga de todos los datos
.venv/bin/python scripts/generar_dashboard.py --refresh
```

El resultado es `forex_dashboard.html`, un archivo único que puedes abrir en
cualquier navegador o copiar a otra máquina. En Linux, `scripts/launch.sh`
genera (si falta) y abre el dashboard.

Las cards se auto-refrescan: si la caché en `data/` queda por detrás del
último día hábil, se re-descarga sola al regenerar — no hace falta `--refresh`
salvo para forzar una descarga completa.

## API de FRED (opcional)

El Real Yield Differential usa inflación de FRED, y los tramos del Tesoro
salen de las series DGS. Configura tu key en una variable de entorno
`FRED_API_KEY` o en un `.env`. Sin key, esas filas se muestran como N/A y el
carry usa los últimos tipos de reserva verificados. El `.env` nunca se sube.

## Fuentes

- Precios (XSP, VIX y macro) y pares FX: Yahoo Finance (`yfinance`)
- Tipos de interés e inflación: FRED (`fredapi`)
- Benchmark crypto (NCI): API pública de nasdaq.com
- Curva del Tesoro: series DGS de FRED (los *Daily Treasury Par Yield Curve
  Rates* del Departamento del Tesoro de EE.UU.)

## Vista previa

![Dashboard de correlaciones FX + macro](docs/screenshots/dashboard.png)
