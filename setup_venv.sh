#!/bin/bash
# Tạo venv Python 3.9 + cài toàn bộ thư viện cho bot AICoverGen. Chạy: bash setup_venv.sh
set -e
cd "$(dirname "$(readlink -f "$0")")"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "[1/6] Cài gói hệ thống (ffmpeg, sox, opus, python3.9)..."
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq ffmpeg sox libsox-fmt-all libopus0 software-properties-common curl
if ! command -v python3.9 >/dev/null; then
  $SUDO add-apt-repository -y ppa:deadsnakes/ppa
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq python3.9 python3.9-venv python3.9-dev
fi

echo "[2/6] Tạo .venv (Python 3.9)..."
[ -x .venv/bin/python ] || python3.9 -m venv .venv
. .venv/bin/activate
pip install -q -U pip uv "setuptools<81" wheel

echo "[3/6] Cài requirements (uv, nhanh hơn pip)..."
uv pip install -q -r requirements.txt

echo "[4/6] Sửa torchaudio (tránh bản ROCm)..."
uv pip install -q --reinstall torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

echo "[5/6] Cài thư viện CUDA 12 cho onnxruntime-gpu..."
uv pip install -q nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
  nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-nvjitlink-cu12 nvidia-cudnn-cu12

echo "[6/6] Tải model tách vocal / hubert / rmvpe..."
python src/download_models.py

echo "Xong! Chạy bot Discord bằng: bash run_bot.sh | bot Telegram bằng: bash run_telegram_bot.sh"
