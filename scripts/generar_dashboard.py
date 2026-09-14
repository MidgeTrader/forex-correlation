"""Ensambla el dashboard final: barra de categorías + FX + paneles macro.

Enfoque A del diseño: el dashboard FX existente se genera intacto con
``forex_to_html.analyze_forex_to_html()`` y se envuelve como la *sección FX*
dentro del HTML final. Los activos macro (metales, energía, índices, agro) se
analizan con las métricas de panel (via ``macro_motores``) y se muestran como
paneles en las otras cuatro secciones. La barra de categorías cambia la sección
visible con ``showCategory(cat)`` — sin recargar y sin tocar el JS del FX.

Uso:
    .venv/bin/python scripts/generar_dashboard.py            # genera forex_dashboard.html
    .venv/bin/python scripts/generar_dashboard.py --refresh  # re-descarga todo
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
ROOT = _SCRIPTS.parent

sys.path.insert(0, str(_SCRIPTS))

from forex_to_html import analyze_forex_to_html  # noqa: E402
from macro_catalogo import CATEGORIAS, CATEGORIAS_ORDER, MACRO_ASSETS, activos_de_categoria  # noqa: E402
from macro_motores import analizar_activo  # noqa: E402
from macro_panel_html import panel_activo  # noqa: E402
from bonos_motores import analizar_curva  # noqa: E402
from bonos_panel_html import panel_curva  # noqa: E402
from xsp_motores import analizar_xsp  # noqa: E402
from xsp_panel_html import panel_xsp  # noqa: E402

# CSS adicional del dashboard (tema FX, misma paleta).
_CSS_MACRO = """
    /* --- Integración macro: barra de categorías y paneles --- */
    .cat-nav { display:flex; gap:8px; justify-content:center; flex-wrap:wrap;
               background:#1c2128; border:1px solid #30363d; border-radius:8px;
               padding:10px; margin-bottom:15px; }
    .cat-btn { background:#21262d; color:#8b949e; border:1px solid #30363d;
               padding:8px 18px; border-radius:6px; cursor:pointer; font-size:0.9em;
               transition:0.2s; }
    .cat-btn:hover { border-color:#58a6ff; color:#c9d1d9; }
    .cat-btn.active { background:#238636; color:white; border-color:#2ea043; }
    .cat-section { display:none; }
    .cat-section.active { display:block; }
    .cat-header { color:#a5d6ff; border-left:3px solid #58a6ff; padding-left:8px;
                  font-size:1em; margin:0 0 12px 0; }
    .macro-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(340px,1fr));
                  gap:12px; align-items:stretch; }
    .macro-card { background:#161b22; border:1px solid #30363d; border-radius:8px;
                  padding:14px; display:flex; flex-direction:column; }
    /* Card que ocupa dos columnas del grid (la del día del mes, con 31 filas).
       Por debajo del ancho de dos columnas mínimas (2x340 + 12 de gap + 20 de
       padding del body) el span se saldría del contenedor, así que vuelve a una
       sola columna: mejor estrecha que desbordada. */
    .macro-card--ancha { grid-column: span 2; }
    @media (max-width: 740px) { .macro-card--ancha { grid-column: span 1; } }
    .macro-card h3 { color:#a5d6ff; margin:0 0 2px 0; font-size:1em; }
    .macro-card .ticker { color:#8b949e; font-weight:normal; font-size:0.8em; }
    .macro-card .macro-precio { font-size:1.15em; font-weight:600; color:#c9d1d9; }
    .macro-card .ret1d { font-size:0.8em; font-weight:600; }
    .macro-card .sub { color:#8b949e; font-size:0.8em; margin:2px 0 10px 0; }
    /* Las imágenes reservan su espacio (aspect-ratio 16/5 = figsize 6.4x2.0)
       antes de decodificarse: sin layout shift, las tarjetas de abajo no se
       montan sobre las de arriba mientras cargan, y todas miden lo mismo. */
    .macro-card img { display:block; width:100%; height:auto; aspect-ratio:16/5;
                      object-fit:contain; margin-top:6px; }
    .macro-table { width:100%; border-collapse:collapse; font-size:0.85em;
                   margin-bottom:10px; table-layout:fixed; }
    .macro-table td { padding:3px 6px; border-bottom:1px solid #21262d; }
    .macro-table td:last-child { text-align:right; font-weight:600; color:#c9d1d9; }
    /* --- Pestaña Gráficas: todos los gráficos al mismo tamaño y contenidos ---
       initAll() crea los chart.js/plotly con la sección oculta (la activa al
       cargar es FX) y quedan con tamaño desproporcionado; este CSS los
       contiene dentro de su tarjeta y showCategory() los redimensiona al abrir. */
    #cat-graficas .card { overflow: hidden; }
    #cat-graficas .card .js-plotly-plot,
    #cat-graficas .card canvas { max-width: 100% !important; max-height: 100% !important; }
"""

_JS_CATEGORIAS = """
    function showCategory(cat) {
        document.querySelectorAll('.cat-section').forEach(function (s) { s.classList.remove('active'); });
        document.querySelectorAll('.cat-btn').forEach(function (b) { b.classList.remove('active'); });
        var sec = document.getElementById('cat-' + cat);
        if (sec) sec.classList.add('active');
        var btn = document.getElementById('btn-' + cat);
        if (btn) btn.classList.add('active');
        // Al abrir "Gráficas", re-dimensiona sus chart.js/plotly: initAll() los
        // crea con la sección oculta y quedan con tamaño desproporcionado.
        if (cat === 'graficas' && window._ajustarGraficasFX) {
            setTimeout(function () { window._ajustarGraficasFX(); }, 60);
        }
    }
"""

# Orden de las categorías en la barra: XSP primero (es la pestaña de trabajo
# diario: los iron condors), luego macro + FX (tarjetas) y al final Gráficas
# (dashboard interactivo FX, renombrado de la antigua pestaña "FX").
_NAV_ORDER = ["xsp"] + CATEGORIAS_ORDER + ["graficas"]

# Etiquetas de la barra: las categorías de activos + las secciones propias.
_ETIQUETAS_NAV = {**CATEGORIAS, "graficas": "Gráficas", "xsp": "XSP"}


def _seccion_bonos(refresh: bool) -> str:
    """Sección de la Curva del Tesoro de EE.UU. (una card con los 5 tramos).

    Los bonos no son activos con retorno/vol: se muestran como curva (nivel y
    cambios en pb) desde FRED/DGS, no con ``macro_motores``.

    Args:
        refresh: True fuerza la re-descarga (ignora la caché en disco).

    Returns:
        HTML de la sección: cabecera de categoría + grid con la card de curva.
    """
    print("  Curva del Tesoro (FRED DGS)...", end="", flush=True)
    df = analizar_curva(refresh=refresh)
    if df is not None:
        print("ok")
    else:
        print("sin datos")
    return (
        '<h2 class="cat-header">Bonos</h2>\n'
        '<div class="macro-grid">\n' + panel_curva(df) + "\n</div>"
    )


def _seccion_xsp(refresh: bool) -> str:
    """Sección de XSP: volatilidad, bandas del condor y rangos por periodo.

    XSP no es un activo macro más (no sale del catálogo): se muestra con su
    propio motor y sus propias cards, porque lo que se mira aquí son las bandas
    de un iron condor, no el retorno del activo.

    Args:
        refresh: True fuerza la re-descarga de los precios (ignora la caché).

    Returns:
        HTML de la sección: cabecera de categoría + grid con las 4 cards.
    """
    print("  XSP (volatilidad + condor)...", end="", flush=True)
    analisis = analizar_xsp(refresh=refresh)
    print("ok" if analisis.precio is not None else "sin datos")
    return (
        '<h2 class="cat-header">XSP · Iron Condors</h2>\n'
        '<div class="macro-grid">\n' + panel_xsp(analisis) + "\n</div>"
    )


def _seccion_macro(categoria: str, refresh: bool) -> str:
    """Paneles de los activos de una categoría macro.

    Args:
        categoria: Clave de categoría (``metales``, ...).
        refresh: True fuerza re-descarga (ignora caché).

    Returns:
        HTML de la sección: cabecera de categoría + grid de tarjetas.
    """
    cards = []
    for key in activos_de_categoria(categoria):
        print(f"  {key} ({MACRO_ASSETS[key]['ticker_yf']})...", end="", flush=True)
        out = analizar_activo(key, refresh=refresh)
        cards.append(panel_activo(key, out["data"], out["results"]))

        if out["results"]:
            print("ok")
        else:
            print(f"sin datos: {out['data'].error}")

    return (
        f'<h2 class="cat-header">{CATEGORIAS[categoria]}</h2>\n'
        f'<div class="macro-grid">\n{"".join(cards)}\n</div>'
    )


def _macro_tickers() -> dict:
    """Dict {nombre_limpio: ticker_yf} de los activos macro, para el panel FX.

    Los pares FX (categoría ``fx``) se excluyen: ya son columnas nativas del
    dashboard de gráficas y, si se inyectaran, se duplicarían.
    """
    return {
        key: spec["ticker_yf"]
        for key, spec in MACRO_ASSETS.items()
        if spec["categoria"] != "fx"
    }


def _macro_labels() -> dict:
    """Dict {nombre_limpio: texto} para el selector del FX (p.ej. ``Oro (GC=F)``)."""
    return {
        key: f'{spec["label"]} ({spec["ticker_yf"]})'
        for key, spec in MACRO_ASSETS.items()
        if spec["categoria"] != "fx"
    }


def _macro_categorias() -> dict:
    """Dict {nombre_limpio: categoria} para el filtro del selector del FX.

    Igual que ``_macro_tickers``/``_macro_labels``, los pares FX (categoría
    ``fx``) se excluyen: en el dashboard son columnas nativas y el filtro los
    considera "fx" por defecto.
    """
    return {
        key: spec["categoria"]
        for key, spec in MACRO_ASSETS.items()
        if spec["categoria"] != "fx"
    }


def _nav() -> str:
    """Barra de categorías con la pestaña FX (tarjetas) activa por defecto."""
    botones = []
    for cat in _NAV_ORDER:
        etiqueta = _ETIQUETAS_NAV[cat]
        activo = " active" if cat == "fx" else ""
        botones.append(
            f'<button class="cat-btn{activo}" id="btn-{cat}" '
            f'onclick="showCategory(\'{cat}\')">{etiqueta}</button>'
        )
    return f'<nav class="cat-nav">\n{"".join(botones)}\n</nav>'


def generar_dashboard(refresh: bool = False) -> str:
    """Construye el HTML final con las 5 categorías.

    Args:
        refresh: True fuerza la re-descarga de los datos macro.

    Returns:
        HTML completo del dashboard.
    """
    print("Generando sección FX...")
    fx_html = analyze_forex_to_html(
        macro_tickers=_macro_tickers(),
        macro_labels=_macro_labels(),
        macro_categorias=_macro_categorias(),
    )

    if fx_html is None:
        raise RuntimeError(
            "La sección FX no se pudo generar (fallo al descargar datos de Yahoo). "
            "Revisa la conexión a internet o la caché en data/."
        )

    head_fx = re.search(r"<head>(.*?)</head>", fx_html, re.S).group(1)
    body_fx = re.search(r"<body>(.*?)</body>", fx_html, re.S).group(1)

    # El header del FX ("Quant Dashboard · Análisis Pro") pasa a ser el
    # encabezado global del dashboard, siempre visible bajo la barra de
    # categorías (hoy vive dentro de la sección FX y desaparece al cambiar
    # de pestaña). Se quita del cuerpo FX para no duplicarlo.
    header_global = re.search(r"<header>.*?</header>", body_fx, re.S).group(0)
    body_fx = body_fx.replace(header_global, "")

    print("Analizando activos macro y pares FX...")
    secciones_macro = []
    for cat in CATEGORIAS_ORDER:
        print(f"[{CATEGORIAS[cat]}]")
        # La pestaña FX (tarjetas de los 7 mayores) es la visible por defecto.
        cls = " active" if cat == "fx" else ""
        seccion = _seccion_bonos(refresh) if cat == "bonos" else _seccion_macro(cat, refresh)
        secciones_macro.append(
            f'<section id="cat-{cat}" class="cat-section{cls}">\n{seccion}\n</section>'
        )

    # Sección XSP (iron condors): va la primera en el HTML para que el botón
    # del nav (que ya la precede) y la sección sigan el mismo orden.
    seccion_xsp = (
        '<section id="cat-xsp" class="cat-section">\n'
        + _seccion_xsp(refresh)
        + "\n</section>"
    )

    secciones = seccion_xsp + "\n" + "\n".join(secciones_macro)
    # Sección "Gráficas": el dashboard interactivo FX (renombrado de la antigua
    # pestaña FX). Queda detrás de la pestaña FX de tarjetas.
    seccion_graficas = '<section id="cat-graficas" class="cat-section">\n' + body_fx + "\n</section>"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
{head_fx}
<style>
{_CSS_MACRO}
</style>
</head>
<body>
<div class="container">
{header_global}
{_nav()}
{secciones}
{seccion_graficas}
</div>
<script>
{_JS_CATEGORIAS}
</script>
</body>
</html>
"""
    return html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera el dashboard integrado (FX + macro).")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-descarga los datos macro (ignora la caché en data/).")
    args = parser.parse_args()

    html = generar_dashboard(refresh=args.refresh)
    html_path = ROOT / "forex_dashboard.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"\nDashboard generado: {html_path}")
