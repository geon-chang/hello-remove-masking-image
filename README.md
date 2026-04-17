# hello-remove-masking-image

Gradio 데모로 **LaMa** vs **MI-GAN** 오브젝트 제거(인페인팅) 성능을 브러시 마스크로 나란히 비교합니다.

## Models

| Model | HF Repo | File | Size | License | 비고 |
|---|---|---|---|---|---|
| LaMa | [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) | `lama_fp32.onnx` | ~208 MB | Apache 2.0 | 입력 512×512 고정 |
| MI-GAN | [andraniksargsyan/migan](https://huggingface.co/andraniksargsyan/migan) | `migan_pipeline_v2.onnx` | ~29.5 MB | MIT | 임의 해상도, crop→paste 내장 |

두 모델 모두 상업적 이용 가능.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_model.py      # ./models 아래에 두 onnx 다운로드 (~237 MB)
python app.py                 # http://localhost:7860
```

### 수동 다운로드 (옵션)
```bash
mkdir -p models
curl -L -o models/migan_pipeline_v2.onnx \
  https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx
curl -L -o models/lama_fp32.onnx \
  https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx
```

### 기존 weights 재사용
```bash
export LAMA_MODEL_PATH=/path/to/lama_fp32.onnx
export MIGAN_MODEL_PATH=/path/to/migan_pipeline_v2.onnx
python app.py
```

## Usage

1. 이미지 업로드 (혹은 클립보드 붙여넣기).
2. 흰 브러시로 제거할 영역을 칠하기 (드래그).
3. **Run** 클릭 → LaMa와 MI-GAN 결과가 오른쪽에 나란히 표시되고 추론 시간(ms) 확인.
4. `MI-GAN: invert mask` 토글로 마스크 극성 검증.

## 비교 관전 포인트

- **속도**: MI-GAN이 LaMa 대비 몇 배 빠른지 (2~3회 평균).
- **품질**: 작은 오브젝트, 큰 오브젝트, 여러 영역 동시 제거 케이스.
- **해상도**: 고해상도 이미지에서 MI-GAN의 crop-paste 이음새가 보이는지 vs LaMa의 전역 512 다운샘플 복원이 디테일을 잃는지.
- **마스크 경계**: MI-GAN은 내부 Gaussian blending, LaMa는 마스크 원형 그대로.

## Execution Providers

`onnxruntime`이 자동으로 다음 순서로 시도합니다: CUDA → CoreML (macOS) → CPU.
실행 시 콘솔 로그에 선택된 provider가 출력됩니다.

## Files

```
hello-remove-masking-image/
├── app.py                 # Gradio UI (LaMa + MI-GAN 병렬 추론)
├── lama_model.py          # LaMa ONNX 래퍼 (float32, 512x512)
├── migan_model.py         # MI-GAN ONNX 파이프라인 래퍼 (uint8, 임의 해상도)
├── download_model.py      # HF Hub → ./models
├── requirements.txt
├── .gitignore
└── models/                # gitignored
    ├── lama_fp32.onnx
    └── migan_pipeline_v2.onnx
```

## License

- 코드: 데모 목적, 자유 사용.
- 모델 가중치: 각 모델 라이선스 (Apache 2.0 / MIT) 준수.
