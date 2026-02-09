#!/usr/bin/env bash
# Cerberus Installation Script
# Run on Raspberry Pi 5 after cloning the repo to /opt/cerberus
set -euo pipefail

CERBERUS_DIR="/opt/cerberus"
VENV_DIR="${CERBERUS_DIR}/venv"
SERVICE_USER="cerberus"

echo "=========================================="
echo " Cerberus Installer — Operation Ground Truth"
echo "=========================================="

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Run as root (sudo ./install.sh)"
    exit 1
fi

echo "[1/8] System packages..."
apt-get update
apt-get install -y \
    python3-dev python3-venv python3-pip \
    libopencv-dev libatlas-base-dev libjpeg-dev libpng-dev \
    gpsd gpsd-clients mosquitto mosquitto-clients \
    i2c-tools libgpiod-dev \
    git curl wget

echo "[2/8] Enabling interfaces..."
raspi-config nonint do_i2c 0
raspi-config nonint do_spi 0
raspi-config nonint do_serial_hw 0
raspi-config nonint do_camera 0

echo "[3/8] Creating service user..."
if ! id "${SERVICE_USER}" &>/dev/null; then
    useradd --system --home-dir "${CERBERUS_DIR}" --shell /usr/sbin/nologin "${SERVICE_USER}"
fi
usermod -aG gpio,i2c,spi,video,audio,dialout "${SERVICE_USER}"

echo "[4/8] Setting directory permissions..."
mkdir -p "${CERBERUS_DIR}/data"
mkdir -p "${CERBERUS_DIR}/data/weed_detections"
mkdir -p "${CERBERUS_DIR}/data/surveillance"
mkdir -p "${CERBERUS_DIR}/data/pest_detections"
mkdir -p "${CERBERUS_DIR}/data/bird_photos"
mkdir -p "${CERBERUS_DIR}/data/microclimate"
mkdir -p "${CERBERUS_DIR}/data/audio/predator"
mkdir -p "${CERBERUS_DIR}/models"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${CERBERUS_DIR}"

echo "[5/8] Creating virtual environment..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[6/8] Installing Python dependencies..."
pip install --upgrade pip
pip install -r "${CERBERUS_DIR}/requirements.txt"
pip install \
    gpiozero lgpio \
    picamera2 \
    tflite-runtime \
    gpsd-py3 \
    adafruit-circuitpython-bme680 \
    adafruit-circuitpython-scd4x \
    adafruit-circuitpython-sht4x \
    smbus2 \
    birdnetlib

deactivate

echo "[7/8] Installing systemd service..."
cp "${CERBERUS_DIR}/cerberus.service" /etc/systemd/system/cerberus.service
systemctl daemon-reload
systemctl enable cerberus.service

echo "[8/8] Configuring gpsd..."
if [ -f /etc/default/gpsd ]; then
    sed -i 's|DEVICES=""|DEVICES="/dev/ttyAMA0"|' /etc/default/gpsd
    sed -i 's|GPSD_OPTIONS=""|GPSD_OPTIONS="-n"|' /etc/default/gpsd
    systemctl enable gpsd
    systemctl restart gpsd
fi

echo ""
echo "=========================================="
echo " Cerberus installed successfully"
echo "=========================================="
echo " Start:   sudo systemctl start cerberus"
echo " Status:  sudo systemctl status cerberus"
echo " Logs:    journalctl -u cerberus -f"
echo " Config:  ${CERBERUS_DIR}/config/cerberus.yaml"
echo "=========================================="