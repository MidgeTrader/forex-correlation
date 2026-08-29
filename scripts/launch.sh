#!/bin/bash
# Quant Dashboard — Launcher Linux
# Abre el dashboard (FX + macro) en el navegador. Si el HTML aún no existe
# (primera vez, o se copió el proyecto a otro equipo), lo genera primero con
# el venv del proyecto. Idempotente: si ya está, solo abre el navegador.
set -euo pipefail

# Ruta derivada de la propia ubicación del script (portable: funciona aunque
# el repo se clone en cualquier directorio).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"
VENV_PY="$PROJ/.venv/bin/python"
HTML="$PROJ/forex_dashboard.html"

# Primera vez: generar el dashboard si no existe todavía.
if [ ! -f "$HTML" ]; then
  cd "$PROJ"
  "$VENV_PY" "$SCRIPT_DIR/generar_dashboard.py"
fi

setsid xdg-open "$HTML" >/dev/null 2>&1 &
