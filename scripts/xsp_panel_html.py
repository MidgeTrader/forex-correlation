"""Cards HTML de la pestaña XSP.

Una card por asunto: volatilidad, **una por vencimiento** (0 DTE, 1 DTE y
semanal van separados, que son los ciclos que se operan), día, semana y mes
para los rangos históricos en %, y una card ancha aparte con los 31 días del
mes, que no cabe en el ancho de una.

El tema visual es el del dashboard (fondo ``#161b22``, acentos
``#58a6ff``/``#3fb950``/``#f85149``) y reutiliza los formateadores y las filas
de ``macro_panel_html`` para no duplicar formato.

Unidades — cada bloque llega en una escala distinta y hay que respetarla:
  - ``hv``, ``iv`` y ``sigma_pct``: fracciones (×100 para %).
  - ``vix``: ya es un % anualizado en puntos (16.84 = 16.84 %), no se toca.
  - ``vix_percentil``: 0-100.
  - ``media_pct``/``mediana_pct``/``ultimo_pct`` de los periodos: YA son %.
  - ``prima_vol``: ratio sin unidad.
Cualquier dato ausente se pinta como ``N/A`` honesto — nunca se inventa, y el
``0`` no se usa como sustituto de «no hay dato».
"""

from __future__ import annotations

from typing import Optional

from macro_panel_html import (  # formateadores y filas del tema FX
    _color_retorno,
    _es_na,
    _fila,
    _GRIS,
    fmt_num,
    fmt_pct_frac,
    fmt_precio,
)

_VERDE = "#3fb950"
_AZUL = "#58a6ff"
_NARANJA = "#ff9f43"
_MORADO = "#a371f7"

# Días del mes por bloque en la card del día del mes: los 31 van repartidos en
# dos bloques (1-16 y 17-31) para que la card no salga del doble de alta que las
# demás, ya que ocupa el ancho de dos.
_DIAS_POR_BLOQUE = 16

# Identificadores de card sin acentos (los ``id`` de HTML, ASCII por limpieza).
_SLUG = {"Día": "dia", "Semana": "semana", "Mes": "mes"}
_SLUG_VENC = {"0 DTE": "0dte", "1 DTE": "1dte", "Semanal": "semanal"}


def _fmt_pct(valor: Optional[float], decimales: int = 2) -> str:
    """Valor que YA está en % (1.23 → ``1.23%``), o N/A."""
    if _es_na(valor):
        return "N/A"
    return f"{valor:.{decimales}f}%"


def _color_percentil(pct: Optional[float]) -> str:
    """Color del percentil de volatilidad: verde si está caro, gris si barato.

    Un percentil alto significa que el mercado paga más volatilidad de lo normal
    en su propia historia, que es cuando vender un condor está mejor pagado.

    Args:
        pct: Percentil 0-100, o None.

    Returns:
        El color hex (sin ``style``) para el valor.
    """
    if _es_na(pct):
        return _GRIS
    return _VERDE if pct >= 50 else _GRIS


def _fila_vencimiento(etiqueta: str, texto: str, color: str = "") -> str:
    """Fila de una sub-tabla de vencimiento, con color opcional."""
    estilo = f' style="color:{color}"' if color else ""
    return f'<tr><td style="color:{_GRIS}">{etiqueta}</td><td><span{estilo}>{texto}</span></td></tr>'


def _filas_vencimiento(v) -> list[str]:
    """Filas de un vencimiento: IV, movimiento esperado, bandas σ con su OTM, ATR y banda del ATR.

    Las alas NO se pintan: en XSP el strike va de $1 en $1, así que el strike
    real más próximo a una banda es esa misma banda redondeada y salían dos
    filas con los mismos números. Se siguen calculando, eso sí, porque el OTM
    se mide sobre ellas: son los strikes que de verdad se teclean en la orden.

    Los importes van en dólares, no en puntos: el movimiento se calcula en puntos
    del índice, pero se pinta con «$» porque es la unidad con la que se lee una
    orden.

    Args:
        v: ``xsp_motores.Vencimiento``.

    Returns:
        Lista de filas ``<tr>`` de ese vencimiento (la card las envuelve).
    """
    # La fecha y el DTE NO van aquí: ya están en la cabecera de la card.
    filas = [
        _fila_vencimiento(
            "IV ATM",
            fmt_pct_frac(v.iv) + (f" (strike {v.strike_atm:,.0f})" if v.strike_atm else ""),
            _AZUL,
        ),
        _fila_vencimiento(
            "Mov. esperado 1σ",
            f'{_fmt_pct(v.sigma_pct * 100 if not _es_na(v.sigma_pct) else None)}'
            f' ({f"${v.sigma_puntos:,.2f}" if not _es_na(v.sigma_puntos) else "N/A"})',
        ),
    ]

    # Cada nivel σ con SU banda y su distancia OTM, que es lo que decide si el
    # condor compensa. Las dos bandas van en color propio —morado la de 1σ,
    # naranja la de 2σ—: en gris el número de la banda lejana se perdía entre las
    # filas y no se distinguía de un vistazo de la banda de dentro.
    niveles = (("1σ", v.banda1, v.ala1, _MORADO), ("2σ", v.banda2, v.ala2, _NARANJA))
    for nombre, banda, ala, color in niveles:
        if banda is None:
            filas.append(_fila_vencimiento(f"Banda {nombre}", "N/A"))
            continue
        filas.append(_fila_vencimiento(
            f"Banda {nombre}", f"{banda[0]:,.0f} – {banda[1]:,.0f}", color,
        ))
        if ala is None or ala[0] is None or ala[1] is None or not v.strike_atm:
            continue
        pct_put = (ala[0] - v.strike_atm) / v.strike_atm * 100
        pct_call = (ala[1] - v.strike_atm) / v.strike_atm * 100
        filas.append(
            f'<tr><td style="color:{_GRIS}; padding-left:14px;">↳ OTM</td>'
            f'<td><span style="color:{_GRIS}">{pct_put:+.2f}% / {pct_call:+.2f}%</span></td></tr>'
        )

    # El ATR va justo encima de su banda, al final de la lista: primero todas las
    # bandas σ juntas (lo que el mercado PAGA) y luego el bloque del ATR con la
    # banda que dibuja su cifra (lo que el mercado HACE). Se etiqueta con el
    # periodo del que sale (diario o semanal) y con la sesión de la que sale la
    # cifra, porque eso cambia según el ciclo: el 1 DTE lleva la sesión de hoy
    # dentro (en curso o ya cerrada) y el 0 DTE y el semanal salen de un cierre
    # anterior. La fila lo dice —«↳ hoy» frente a «↳ cierre»— para que nadie
    # compare dos ATR creyendo que se leyeron en el mismo momento.
    if v.atr is not None:
        filas.append(_fila_vencimiento(
            v.atr.etiqueta,
            f'${v.atr.valor:,.2f}'
            f' ({_fmt_pct(v.atr.pct * 100 if not _es_na(v.atr.pct) else None)})'
            if not _es_na(v.atr.valor) else "N/A",
        ))
        if v.atr.desde:
            origen = "↳ hoy" if v.atr.incluye_hoy else "↳ cierre"
            filas.append(
                f'<tr><td style="color:{_GRIS}; padding-left:14px;">{origen}</td>'
                f'<td><span style="color:{_GRIS}">{v.atr.desde}</span></td></tr>'
            )

    # La banda del ATR cierra la lista: mismo spot que las bandas σ, pero movido
    # por lo que el precio ha recorrido de verdad. Sin OTM: no es un strike que
    # se teclee, es hasta dónde ha llegado el precio.
    if v.banda_atr is not None:
        filas.append(_fila_vencimiento(
            "Banda ATR", f"{v.banda_atr[0]:,.0f} – {v.banda_atr[1]:,.0f}", _AZUL,
        ))

    return filas


def _card_volatilidad(a) -> str:
    """Card de volatilidad: realizada a varias ventanas, VIX y prima."""
    hv_filas = [
        _fila(f"HV {ventana}D", f'<span style="color:#c9d1d9">{fmt_pct_frac(valor)}</span>')
        for ventana, valor in a.hv.items()
    ]

    prima = a.prima_vol
    prima_txt = fmt_num(prima)
    prima_nota = ""
    if not _es_na(prima):
        prima_nota = " (vender pagado)" if prima >= 1 else " (vender barato)"

    filas = [
        _fila("Precio XSP", f'<span style="color:{_AZUL}">{fmt_precio(a.precio)}</span>'),
        _fila("Momento", f'<span style="color:{_GRIS}">{a.fecha_precio or "N/A"}</span>'),
        _fila("VIX", "N/A" if _es_na(a.vix) else f"{a.vix:.2f}%"),
        _fila(
            "Percentil VIX (5a)",
            f'<span style="color:{_color_percentil(a.vix_percentil)}">'
            f'{"N/A" if _es_na(a.vix_percentil) else f"{a.vix_percentil:.1f}%"}</span>',
        ),
        _fila(
            "Prima VIX / HV30",
            f'<span{_color_retorno(None if _es_na(prima) else prima - 1)}>{prima_txt}{prima_nota}</span>',
        ),
    ] + hv_filas

    notas = "".join(f'<div class="sub">· {n}</div>' for n in a.notas)

    return f"""
    <div class="macro-card" id="xsp-vol">
        <div class="macro-head">
            <h3>Volatilidad <span class="ticker">(XSP)</span></h3>
            <div class="sub">Vol realizada 5a · VIX y prima de volatilidad</div>
        </div>
        <table class="macro-table">
            {"".join(filas)}
        </table>
        <div style="margin-top:6px;">{notas}</div>
    </div>
    """


def _card_vencimiento(v) -> str:
    """Card de un vencimiento del condor: el diario o el semanal, cada uno aparte.

    Args:
        v: ``xsp_motores.Vencimiento``.

    Returns:
        HTML de la card con su IV, su movimiento esperado y sus alas.
    """
    return f"""
    <div class="macro-card" id="xsp-{_SLUG_VENC.get(v.etiqueta, "vencimiento")}">
        <div class="macro-head">
            <h3>{v.etiqueta} <span class="ticker">(XSP)</span></h3>
            <div class="sub">Expira {v.fecha} · {v.dte} DTE</div>
        </div>
        <table class="macro-table">
            {"".join(_filas_vencimiento(v))}
        </table>
    </div>
    """


def _texto_cierre(p) -> str:
    """Sufijo « · último cerrado <fecha>» para el subtítulo de una card.

    Se dice de qué cierre sale la cifra porque los rangos se miden contra el
    cierre ANTERIOR al periodo: sin la fecha, una card que llega hasta el
    viernes pasado se leería como si llegara a hoy.
    """
    return f" · último cerrado {p.ultimo_fin}" if p.ultimo_fin else ""


def _card_periodo(p, desglose: str = "", sub: str = "") -> str:
    """Card de un periodo (día, semana o mes) con su rango medio en %.

    Args:
        p: ``xsp_motores.EstadisticaPeriodo``.
        desglose: Tabla HTML ya montada (por día de la semana o por día del
            mes), o "" si la card no lleva desglose.
        sub: Texto de la cabecera (qué ventana de historia cubre).

    Returns:
        HTML de la card.
    """
    cierre = _texto_cierre(p)
    filas = [
        _fila("Rango medio", f'<span style="color:{_VERDE}">{_fmt_pct(p.media_pct)}</span>'),
        _fila("Mediana", f'<span style="color:#c9d1d9">{_fmt_pct(p.mediana_pct)}</span>'),
        _fila("Último", _fmt_pct(p.ultimo_pct)),
        _fila("Periodos", f'<span style="color:{_GRIS}">{p.n}</span>'),
    ]

    bloque_desglose = ""
    if desglose:
        bloque_desglose = (
            f'<div class="sub" style="color:{_AZUL}; font-weight:600; margin:6px 0 4px 0;">'
            f"Desglose</div>{desglose}"
        )

    return f"""
    <div class="macro-card" id="xsp-{_SLUG.get(p.etiqueta, "periodo")}">
        <div class="macro-head">
            <h3>{p.etiqueta} <span class="ticker">(XSP)</span></h3>
            <div class="sub">{sub}{cierre}</div>
        </div>
        <table class="macro-table">
            {"".join(filas)}
        </table>
        {bloque_desglose}
    </div>
    """


def _cabecera_desglose(columnas: tuple[str, ...]) -> str:
    """Fila de cabecera del desglose: la columna de etiquetas va vacía.

    El ``font-weight:normal`` de las celdas es necesario, no decorativo: la hoja
    de estilo pone en negrita la última columna de ``.macro-table`` —correcto en
    la tabla principal, donde es el valor— y aquí la última columna es una
    etiqueta más.
    """
    celdas = "".join(
        f'<td style="text-align:right; color:{_GRIS}; font-weight:normal">{col}</td>'
        for col in columnas
    )
    return f"<tr><td></td>{celdas}</tr>"


def _fila_desglose(etiqueta: str, valores: tuple[str, ...], color: str) -> str:
    """Fila del desglose: etiqueta a la izquierda y los valores a la derecha."""
    celdas = "".join(
        f'<td style="text-align:right; color:{color}">{valor}</td>' for valor in valores
    )
    return f'<tr><td style="color:{_GRIS}">{etiqueta}</td>{celdas}</tr>'


def _tabla_desglose(cabecera: str, filas: list[str]) -> str:
    """Tabla propia para el desglose.

    Va en su propia ``<table>`` y no dentro de la de la card porque necesita sus
    columnas: ``.macro-table`` usa ``table-layout: fixed`` y reparte el ancho a
    partes iguales, así que metida en la tabla de dos columnas los valores
    quedarían a media anchura en vez de pegados al borde derecho.
    """
    return f'<table class="macro-table">{cabecera}{"".join(filas)}</table>'


def _desglose_dow(a) -> str:
    """Tabla del desglose por día de la semana: arriba, abajo y total.

    Se marca en naranja el día de más recorrido total, que es el que más aprieta
    un condor.
    """
    if not a.por_dia_semana:
        return ""
    mayor = max(total for _, _, _, total in a.por_dia_semana)
    filas = []
    for nombre, arriba, abajo, total in a.por_dia_semana:
        # Las dos mitades se redondean primero y el total se pinta como su SUMA,
        # no como el redondeo de la media del total. Los dos caminos se separan
        # en 0,01 pp (0,577 + 0,515 son 1,092, pero 0,58 + 0,52 son 1,10), y una
        # fila cuyas columnas no cuadran parece un error de cálculo aunque no lo
        # sea: lo que se ve en la card tiene que sumar. El desvío frente a la
        # media medida nunca pasa de 0,01 pp.
        mitades = (round(arriba, 2), round(abajo, 2))
        filas.append(_fila_desglose(
            nombre,
            (_fmt_pct(mitades[0]), _fmt_pct(mitades[1]), _fmt_pct(mitades[0] + mitades[1])),
            _NARANJA if total == mayor else "#c9d1d9",
        ))
    return _tabla_desglose(_cabecera_desglose(("arriba", "abajo", "total")), filas)


def _bloque_dias(dias: list[tuple[int, float]], mayor: float) -> str:
    """Tabla de un bloque de días del mes, con su cabecera.

    Args:
        dias: Pares ``(día del mes, rango medio %)`` de ese bloque.
        mayor: Rango medio más alto del mes entero, que va en naranja.

    Returns:
        HTML de la tabla del bloque.
    """
    filas = [
        _fila_desglose(
            f"Día {dia}", (_fmt_pct(valor),),
            _NARANJA if valor == mayor else "#c9d1d9",
        )
        for dia, valor in dias
    ]
    return _tabla_desglose(_cabecera_desglose(("Rango medio",)), filas)


def _card_dia_mes(a, cierre: str = "") -> str:
    """Card ancha con el rango medio de cada día del mes, del 1 al 31.

    Va aparte de la card «Mes» y ocupa dos columnas del grid: son 31 filas y en
    el ancho de una sola card la lista saldría del doble de alta que las demás.
    Los días van en orden natural (1, 2, 3…) y **no** ordenados por rango, que
    es como se enseñaban antes los cinco primeros: puestos por tamaño parecían
    «los peores días del mes» cuando lo que hay es la dispersión que sale por
    azar al partir 31 cestas. En orden natural se ve de un vistazo si algún día
    se sale de la fila o si están todos a la par.

    Args:
        a: ``xsp_motores.AnalisisXSP``.
        cierre: Sufijo de fecha ya montado (``_texto_cierre``) para el subtítulo.

    Returns:
        HTML de la card, o "" si no hay desglose por día del mes.
    """
    if not a.por_dia_mes:
        return ""
    por_dia = sorted(a.por_dia_mes, key=lambda par: par[0])
    mayor = max(valor for _, valor in a.por_dia_mes)
    bloques = [
        _bloque_dias(por_dia[i:i + _DIAS_POR_BLOQUE], mayor)
        for i in range(0, len(por_dia), _DIAS_POR_BLOQUE)
    ]
    columnas = "".join(
        f'<div style="flex:1 1 0; min-width:0;">{bloque}</div>' for bloque in bloques
    )
    return f"""
    <div class="macro-card macro-card--ancha" id="xsp-dia-mes">
        <div class="macro-head">
            <h3>Día del mes <span class="ticker">(XSP)</span></h3>
            <div class="sub">Rango medio desde el cierre previo · 5 años{cierre}</div>
        </div>
        <div style="display:flex; gap:14px; align-items:flex-start;">{columnas}</div>
    </div>
    """


def panel_xsp(a) -> str:
    """Grid con las cards de la pestaña XSP.

    Args:
        a: ``xsp_motores.AnalisisXSP``.

    Returns:
        HTML del grid (o una card honesta de sin-datos si no hay precio).
    """
    if a.precio is None:
        notas = "".join(f'<div class="sub">· {n}</div>' for n in a.notas)
        return (
            '<div class="macro-card" id="xsp-sin-datos">'
            "<h3>XSP</h3>"
            f'<div class="sub">Sin datos disponibles</div>{notas}'
            "</div>"
        )

    por_etiqueta = {p.etiqueta: p for p in a.periodos}
    dia = por_etiqueta.get("Día")
    semana = por_etiqueta.get("Semana")
    mes = por_etiqueta.get("Mes")

    # Primero el contexto (volatilidad) y luego una card por vencimiento (0 DTE,
    # 1 DTE y semanal), que son los ciclos que se operan.
    cards = [_card_volatilidad(a)]
    cards += [_card_vencimiento(v) for v in a.vencimientos]
    if not a.vencimientos:
        cards.append(
            '<div class="macro-card" id="xsp-vencimientos">'
            "<h3>Vencimientos</h3>"
            '<div class="sub">Cadena de opciones no disponible.</div></div>'
        )
    if dia:
        cards.append(_card_periodo(
            dia, _desglose_dow(a), "Rango desde el cierre previo · 5 años",
        ))
    if semana:
        cards.append(_card_periodo(semana, "", "Rango desde el cierre previo · 5 años"))
    if mes:
        cards.append(_card_periodo(mes, "", "Rango desde el cierre previo · 5 años"))
        # El desglose del mes vive en su propia card (ancha, al final): dentro de
        # la del mes solo cabían cinco días y enseñar cinco de 31 es enseñar la
        # cabeza de una lista, que se lee como un ranking aunque no lo sea.
        cards.append(_card_dia_mes(a, _texto_cierre(mes)))

    return "".join(cards)
