#!/usr/bin/env bash
# Cerberus Manual Start Script
# Use for development and testing — production uses systemd
set -euo pipefail

CERBERUS_DIR="/opt/cerberus"
VENV_DIR="${CERBERUS_DIR}/venv"

if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "WARNING: No virtual environment found at ${VENV_DIR}"
    echo "Using system Python"
fi

export PYTHONUNBUFFERED=1
export CERBERUS_ENV="${CERBERUS_ENV:-development}"
export CERBERUS_CONFIG="${CERBERUS_CONFIG:-${CERBERUS_DIR}/config/cerberus.yaml}"

echo "=========================================="
echo " Starting Cerberus — ${CERBERUS_ENV}"
echo " Config: ${CERBERUS_CONFIG}"
echo "=========================================="

cd "${CERBERUS_DIR}"
python -m cerberus.core.brain "$@"
```

Commit:
```
git add .
git commit -m "Infrastructure: requirements.txt, systemd service, install script, start script"
git push