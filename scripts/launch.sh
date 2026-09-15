#!/bin/bash
# Quant Dashboard — Launcher Linux
# Abre el dashboard (FX + macro) en el navegador, regenerándolo antes si el HTML
# no existe o si es de un día anterior. Regenerar es lo que hace que al abrir la
# aplicación las cards de los condores enseñen la sesión de HOY —el 1 DTE vive el
# día entero y su ATR incluye el día en curso—, así que no basta con abrir el
# HTML que quedó de la última vez.
set -euo pipefail

# Ruta derivada de la propia ubicación del script (portable: funciona aunque
# el repo se clone en cualquier directorio).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"
VENV_PY="$PROJ/.venv/bin/python"
HTML="$PROJ/forex_dashboard.html"

# La generación tarda alrededor de minuto y medio, y el lanzador se ejecuta sin
# terminal (Terminal=false en el .desktop): sin aviso, ese rato con el icono
# pulsado y nada en pantalla parece un programa que no arranca. Las llamadas a
# notify-send llevan `|| true` a propósito: quedarse sin notificaciones es una
# molestia, no poder abrir el dashboard sería otra cosa.
if [ ! -f "$HTML" ] || [ "$(date -r "$HTML" +%F)" != "$(date +%F)" ]; then
  notify-send -a "Quant Dashboard" "Actualizando datos…" \
    "Descargando precios y recalculando (1-2 min)." || true
  cd "$PROJ"
  if "$VENV_PY" "$SCRIPT_DIR/generar_dashboard.py"; then
    notify-send -a "Quant Dashboard" "Dashboard actualizado" \
      "Abriendo con los datos de hoy." || true
  else
    # Fallo seguro: el HTML anterior sigue en disco (se escribe al final del
    # proceso), así que se abre ese antes que dejar al usuario sin nada.
    notify-send -u critical -a "Quant Dashboard" "No se pudo actualizar" \
      "Se abre el dashboard anterior." || true
  fi
fi

setsid xdg-open "$HTML" >/dev/null 2>&1 &
