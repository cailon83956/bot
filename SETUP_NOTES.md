# AICoverGen - Setup Notes (Ubuntu 24.04 / Python 3.9 / NVIDIA T4)

Ghi lại các vấn đề đã gặp và cách khắc phục khi cài AICoverGen trên Ubuntu 24.04.
Repo gốc yêu cầu Python 3.9 (xem `cog.yaml` và README).

## Môi trường đã dùng

- Ubuntu 24.04.4 LTS, x86_64
- Python 3.9.25 (từ deadsnakes PPA)
- NVIDIA T4 15GB, driver 580.82.07, CUDA 13.0 (system)
- torch 2.0.1+cu118

## Cài đặt

```bash
apt-get install -y python3.9 python3.9-venv python3.9-dev ffmpeg sox libsox-fmt-all
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download_models.py
```

Có thể dùng `uv` thay cho `pip` để cài nhanh hơn nhiều:
`uv pip install -r requirements.txt`

## Lỗi đã gặp và cách sửa

### 1. torchaudio bị resolve nhầm bản ROCm

**Triệu chứng:**
```
OSError: libtorch_hip.so: cannot open shared object file: No such file or directory
```

**Nguyên nhân:** `requirements.txt` dùng
`--find-links https://download.pytorch.org/whl/torch_stable.html`.
Trang index cũ này nay chứa cả bản ROCm (GPU AMD), và trình resolve có thể chọn
`torchaudio==2.0.2+rocm5.4.2` thay vì bản CUDA.

**Sửa:** ép bản cu118 khớp torch
```bash
pip install --reinstall torchaudio==2.0.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

### 2. Thiếu setuptools / pkg_resources

**Triệu chứng:** nhiều module fail với
`ModuleNotFoundError: No module named 'pkg_resources'`
(gradio, librosa, pyworld, torchcrepe...)

**Nguyên nhân:** `uv venv` không cài `setuptools` mặc định.

**Sửa:**
```bash
pip install "setuptools<81" wheel
```
Giữ dưới 81 vì setuptools 81+ đã bắt đầu bỏ `pkg_resources`, mà gradio 3.39.0
vẫn cần nó.

### 3. onnxruntime rơi về CPU một cách âm thầm (lỗi nghiêm trọng nhất)

**Triệu chứng:** không có lỗi nào hiện ra, WebUI vẫn chạy bình thường, nhưng
tách vocal chậm hơn ~10-20 lần. Chỉ phát hiện khi kiểm tra thủ công:
```python
import onnxruntime as ort
s = ort.InferenceSession(model, providers=['CUDAExecutionProvider'])
print(s.get_providers())
# -> ['CPUExecutionProvider']   <- fallback CPU
```

**Nguyên nhân:** `onnxruntime-gpu 1.19.2` yêu cầu **CUDA 12.x + cuDNN 9**.
Máy chỉ có CUDA 11.8 (bundled trong torch) và CUDA 13.0 (system), nên
thiếu `libcublasLt.so.12`. Log báo:
```
Failed to load library libonnxruntime_providers_cuda.so with error:
libcublasLt.so.12: cannot open shared object file
Require cuDNN 9.* and CUDA 12.*
```

**Sửa:** cài thư viện CUDA 12 vào venv và trỏ `LD_LIBRARY_PATH`:
```bash
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
            nvidia-cufft-cu12 nvidia-curand-cu12 \
            nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-nvjitlink-cu12

export LD_LIBRARY_PATH=".venv/lib/python3.9/site-packages/nvidia/cublas/lib:\
.v/LIB/python3.9/site-packages/nvidia/cuda_runtime/lib:\
.v/LIB/python3.9/site-packages/nvidia/cufft/lib:\
.v/LIB/python3.9/site-packages/nvidia/curand/lib:\
.v/LIB/python3.9/site-packages/nvidia/cusparse/lib:\
.v/LIB/python3.9/site-packages/nvidia/cusolver/lib:\
.v/LIB/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
```
(giữ đúng đường dẫn `.venv/lib/python3.9/site-packages/nvidia/...`)

Script `run_webui.sh` trong thư mục gốc đã gói sẵn phần này.
**Phải dùng script đó, không gọi `python src/webui.py` trực tiếp.**

### 4. Thiếu binary sox

**Triệu chứng:** `SoX could not be found!` (chỉ cảnh báo, chưa chặn ngay)

**Sửa:** `apt-get install -y sox libsox-fmt-all`

### 5. Package `lib==4.0.0` trong requirements.txt

Đây là tên package rất chung trên PyPI (nghi vấn dependency-confusion / package
rác). Repo không có code nào `import lib` nên cài vô hại, nhưng nên xoá khỏi
`requirements.txt` để tránh rủi ro.

## Chạy

```bash
bash run_webui.sh --listen --listen-host 0.0.0.0 --listen-port 7860
```

WebUI mặc định ở http://127.0.0.1:7860

## Ghi chú

- Cần **RVC voice model** (`.pth` + `.index`) đặt trong `rvc_models/<tên>/`
  để chạy được bước đổi giọng. Script `src/download_models.py` chỉ tải model
  tách vocal + hubert, không tải RVC model.
- Nếu chạy trong Docker, cần `--gpus all` để dùng được GPU.
