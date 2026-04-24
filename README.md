# hello-remove-masking-image

Gradio 데모. 브러시로 칠한 영역을 3개 모델로 비교 제거합니다.

- **AS-IS** (가볍고 빠름, CPU/ONNX): LaMa, MI-GAN
- **TO-BE** (고품질 diffusion, GPU 권장): PowerPaint v2-1

또한 PowerPaint의 5가지 태스크(object-remove / text-guided / shape-guided / context-aware / outpainting)를 자유롭게 실험할 수 있는 playground 탭 포함.

## Models

| Model | HF Repo | 크기 | License | 비고 |
|---|---|---|---|---|
| LaMa | [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) | ~208 MB | Apache 2.0 | ONNX · 입력 512×512 고정 |
| MI-GAN | [andraniksargsyan/migan](https://huggingface.co/andraniksargsyan/migan) | ~29.5 MB | MIT | ONNX · 임의 해상도, crop→paste 내장 |
| PowerPaint v2-1 | [Sanster/PowerPaint_v2](https://huggingface.co/Sanster/PowerPaint_v2) + [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | ~5 GB | Apache 2.0 (code) / CreativeML OpenRAIL-M (weights) | SD 1.5 + BrushNet · 5개 태스크 지원 · GPU 권장 |

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate

# 기본 데모 (LaMa + MI-GAN, GPU 우선)
pip install gradio==5.49.1 onnxruntime-gpu numpy pillow huggingface_hub opencv-python

# PowerPaint 추가 (iopaint는 오래된 deps 핀이 있어 --no-deps 필요)
pip install --no-deps iopaint==1.6.0
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  'torch>=2.8,<2.11' torchvision \
  'diffusers>=0.32,<0.36' 'transformers>=4.39,<5' tokenizers \
  'accelerate>=1.0' safetensors 'peft>=0.17' \
  controlnet-aux==0.0.3 timm einops omegaconf \
  'antlr4-python3-runtime==4.9.3' scikit-image scipy \
  loguru yacs easydict piexif==1.1.3 \
  python-socketio==5.7.2 typer-config==1.4.0 \
  psutil ftfy sentencepiece requests standard-imghdr

# ONNX 가중치 다운로드 (~237 MB)
python download_model.py

# 실행 (PowerPaint 가중치 ~5 GB는 첫 호출 시 HF 캐시에 자동 다운로드)
python app.py
```

웹 UI는 로컬에서 `http://localhost:7860`.
프록시 배포 환경에서는 `https://dev-aplus-api.altools.co.kr/ai-demo/eraser/`.

### PowerPaint 비활성화 (LaMa + MI-GAN만 쓰고 싶을 때)
```bash
ENABLE_POWERPAINT=0 python app.py
```

### SD 1.5 base 교체
기본값은 `stable-diffusion-v1-5/stable-diffusion-v1-5` (runwayml 원본은 2024-07에 삭제됨).
다른 SD 1.5 base로 바꾸려면:
```bash
export POWERPAINT_SD15_BASE=<hf-repo-id>
```

### 경로 오버라이드
```bash
export LAMA_MODEL_PATH=/path/to/lama_fp32.onnx
export MIGAN_MODEL_PATH=/path/to/migan_pipeline_v2.onnx
python app.py
```

## Usage

### Tab 1 — Comparison (AS-IS vs TO-BE)
1. **비교용 이미지 파일 선택**에서 이미지를 고른 뒤 **선택한 파일을 에디터에 불러오기** 클릭.
2. 에디터에서 흰 브러시로 제거할 영역 칠하기.
3. **Run all 3 models** 클릭.
4. 오른쪽에 LaMa / MI-GAN / PowerPaint 결과가 세로로 표시되고 각 ms 출력.

### Tab 2 — PowerPaint Playground
1. **이미지 파일 선택**에서 이미지를 고른 뒤 **선택한 파일 불러오기** 클릭.
2. 확장할 방향(top/right/bottom/left)과 Prompt / Steps / CFG / Seed 조절.
3. **Outpaint** 클릭.

브라우저 파일 입력은 Gradio의 기본 upload-progress 경로를 우회하도록 구현되어 있습니다.
프록시 환경에서 `upload_progress?upload_id=undefined`가 발생하지 않도록 이미지 파일은 브라우저에서 base64로 읽어 서버 함수에 전달합니다.

## GPU

`app.py`가 자동으로 우선순위: **CUDA > MPS (Apple Silicon) > CPU**.
- NVIDIA: onnxruntime-gpu + CUDA toolkit 필요. torch도 CUDA 빌드 필수.
- Apple Silicon: torch MPS backend 자동 사용 (iopaint 내부에서 float32).
- CPU: 동작하지만 PowerPaint는 이미지당 수 분 단위.

## Docker

현재 저장소의 Docker 설정은 **NVIDIA GPU 우선** 기준으로 맞춰져 있습니다.

```bash
docker compose up --build -d
docker exec hello-eraser nvidia-smi
docker exec hello-eraser python -c "import torch, onnxruntime as ort; print(torch.cuda.is_available(), torch.version.cuda); print(ort.get_available_providers())"
```

정상이라면 아래 조건이 만족되어야 합니다.
- `torch.cuda.is_available()` 가 `True`
- `torch.version.cuda` 가 `12.8`
- ONNX Runtime provider 목록에 `CUDAExecutionProvider` 가 포함

참고:
- `docker-compose.yml` 에 `gpus: all` 을 넣어 일반 `docker compose up` 에서도 GPU가 안정적으로 전달되게 했습니다.
- `deploy.resources...devices` 만으로는 Compose 환경에서 GPU 예약이 무시될 수 있습니다.

## Files

```
hello-remove-masking-image/
├── app.py                 # Gradio UI (Tab1: 비교, Tab2: PowerPaint playground)
├── lama_model.py          # LaMa ONNX 래퍼
├── migan_model.py         # MI-GAN ONNX 래퍼
├── powerpaint_model.py    # PowerPaint v2-1 래퍼 (iopaint PowerPaintV2 직접 활용)
├── download_model.py      # LaMa/MI-GAN HF → ./models
├── requirements.txt
└── models/                # gitignored
    ├── lama_fp32.onnx
    └── migan_pipeline_v2.onnx
```

## License

- 데모 코드: 자유 사용.
- 모델 가중치: 각 모델 라이선스 (Apache 2.0 / MIT / CreativeML OpenRAIL-M) 준수.
