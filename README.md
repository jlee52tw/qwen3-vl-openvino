# Qwen3-VL OpenVINO

Visual-language assistant using **Qwen3-VL-8B-Instruct** with OpenVINO acceleration.  
Supports image captioning, visual Q&A, and video understanding.

Based on the [OpenVINO Notebooks qwen3-vl](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/qwen3-vl) example.

## Features

- **Image Captioning** — Describe any image in natural language
- **Visual Q&A** — Ask questions about image contents
- **Video Understanding** — Analyze and describe video content
- **INT4 Quantization** — 5.71 GB model (from ~17 GB FP16) for efficient inference
- **GPU Accelerated** — Runs on Intel integrated GPU via OpenVINO

## Model

| Property | Value |
|---|---|
| Model | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Architecture | Qwen3-VL (Vision-Language, 8B params) |
| Weight Format | INT4 (group-size 128, ratio 0.8) |
| Model Size | 5.71 GB |
| Framework | OpenVINO 2026.0 + optimum-intel 1.27.0.dev0 |

## GPU Benchmark Results (Intel Core Ultra iGPU)

### Image Captioning (demo.jpeg 2048×1365, 100 tokens)

| Metric | Value |
|---|---|
| Median Generate Time | 8.777s |
| Mean Generate Time | 8.578s |
| Median Tokens/sec | 11.4 |
| Input Tokens | 964 |
| Output Tokens | 100 |

### Visual Q&A (demo.jpeg 2048×1365, 50 tokens)

| Metric | Value |
|---|---|
| Median Generate Time | 5.639s |
| Mean Generate Time | 5.466s |
| Median Tokens/sec | 8.9 |
| Input Tokens | 966 |
| Output Tokens | 50 |

### Video Understanding (demo_video.mp4 640×480 3s, 100 tokens)

| Metric | Value |
|---|---|
| Median Generate Time | 7.167s |
| Mean Generate Time | 7.426s |
| Median Tokens/sec | 13.9 |
| Input Tokens | 632 |
| Output Tokens | 100 |

## Setup

```powershell
# Create and activate venv
python -m venv qwen3-vl-venv
& .\qwen3-vl-venv\Scripts\Activate.ps1

# Install dependencies
pip install "torch==2.8" "torchvision==0.23.0" --extra-index-url https://download.pytorch.org/whl/cpu
pip install "qwen-vl-utils" "nncf" "openvino>=2025.4" "openvino-tokenizers"
pip install git+https://github.com/huggingface/optimum-intel.git
```

## Usage

### Convert Model (one-time)

```bash
python qwen3_vl.py --convert-only
```

Converts `Qwen/Qwen3-VL-8B-Instruct` to INT4 OpenVINO IR at `C:\working\models\Qwen3-VL-8B-Instruct\INT4`.

### Image Captioning

```bash
python qwen3_vl.py --task caption --image photo.jpg --device GPU --skip-conversion
```

### Visual Q&A

```bash
python qwen3_vl.py --task vqa --image photo.jpg --question "What color is the car?" --device GPU --skip-conversion
```

### Video Understanding

```bash
python qwen3_vl.py --task video --video clip.mp4 --question "What happens in this video?" --device GPU --skip-conversion
```

### Benchmark

```bash
python qwen3_vl.py --task caption --device GPU --skip-conversion --benchmark 5 --max-tokens 100
```

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--task` | `caption` | Task: `caption`, `vqa`, `video` |
| `--image` | demo.jpeg | Input image path |
| `--video` | — | Input video path (required for video task) |
| `--question` | auto | Question text |
| `--device` | `GPU` | Inference device: `GPU`, `CPU`, `AUTO` |
| `--model-id` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID |
| `--model-dir` | `C:\working\models\Qwen3-VL-8B-Instruct\INT4` | OpenVINO model directory |
| `--weight-format` | `int4` | Weight format: `fp16`, `int8`, `int4` |
| `--max-tokens` | `200` | Max output tokens |
| `--convert-only` | — | Only convert, don't infer |
| `--skip-conversion` | — | Skip conversion step |
| `--benchmark N` | — | Run N benchmark iterations |
| `--no-stream` | — | Disable streaming output |

## Sample Outputs

**Caption** (demo.jpeg — woman with dog on beach):
> This is a heartwarming and serene photograph capturing a tender moment between a young woman and her golden Labrador Retriever on a sandy beach at sunset.

**VQA** — "How many people and animals are in this photo?":
> There is **one person** in the photo: a woman sitting on the sand. There is **one animal** in the photo: a yellow Labrador Retriever.

**Video** — "What do you see in this video?":
> A woman sits on a sandy beach at sunset, smiling as she interacts with a golden retriever. The dog, wearing a harness, sits upright and reaches its paw toward the woman, who holds a small treat in her hand.

## Dependencies

- Python 3.12
- torch 2.8 (CPU)
- torchvision 0.23.0
- openvino 2026.0
- optimum-intel 1.27.0.dev0
- transformers 4.57.6
- qwen-vl-utils 0.0.14
- nncf 3.0.0

## License

Apache 2.0 (follows OpenVINO Notebooks licensing)
