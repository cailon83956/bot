#!/bin/bash
# Chạy bot Telegram AICoverGen (không có Web UI).
# Dùng được ở bất kỳ thư mục nào: bash run_telegram_bot.sh
cd "$(dirname "$(readlink -f "$0")")" || exit 1
ROOT="$(pwd)"

# Nếu có venv .venv thì dùng nó
if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
  SP="$(ls -d "$ROOT"/.venv/lib/python3*/site-packages 2>/dev/null | head -n1)"
else
  PY="${PYTHON:-python3}"
  SP="$($PY -c 'import site;print(site.getsitepackages()[0])' 2>/dev/null)"
fi

# onnxruntime-gpu cần thư viện CUDA 12 (nếu cài qua pip nvidia-*-cu12) - tránh âm thầm rơi về CPU
if [ -n "$SP" ] && [ -d "$SP/nvidia" ]; then
  for d in "$SP"/nvidia/*/lib; do export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"; done
fi

# Tự khởi động lại nếu bot sập (nghỉ 10s)
while true; do
  "$PY" src/telegram_bot.py "$@"
  code=$?
  [ $code -eq 0 ] && break
  echo "[run_telegram_bot] Bot thoát với mã $code, khởi động lại sau 10s..."
  sleep 10
done
