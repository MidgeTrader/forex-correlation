#!/bin/bash
# Quant Dashboard — setup de un solo paso.
# Crea el venv, instala las dependencias pineadas y genera el dashboard.
# Uso:  bash scripts/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"

echo "[1/3] Creando venv en $PROJ/.venv ..."
python3 -m venv "$PROJ/.venv"

echo "[2/3] Instalando dependencias desde requirements.txt ..."
"$PROJ/.venv/bin/pip" install --upgrade pip
"$PROJ/.venv/bin/pip" install -r "$PROJ/requirements.txt"

echo "[3/3] Generando el dashboard ..."
"$PROJ/.venv/bin/python" "$SCRIPT_DIR/generar_dashboard.py"

echo "Listo: $PROJ/forex_dashboard.html"
