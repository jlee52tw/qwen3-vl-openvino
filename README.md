# Qwen3-VL OpenVINO

Visual-language assistant using **Qwen3-VL** (2B / 4B / 8B) with OpenVINO acceleration.  
Supports image captioning, visual Q&A, video understanding, and multi-model comparison.

Based on the [OpenVINO Notebooks qwen3-vl](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/qwen3-vl) example.

## Features

- **Image Captioning** — Describe any image in natural language
- **Visual Q&A** — Ask questions about image contents
- **Video Understanding** — Analyze and describe video content
- **OCR** — Read and transcribe text from images
- **Multi-Model Comparison** — Compare 2B, 4B, 8B models side by side
- **INT4 Quantization** — Efficient inference with minimal quality loss
- **GPU Accelerated** — Runs on Intel integrated GPU via OpenVINO

## Supported Models

| Model | Parameters | INT4 Size | Load Time | Avg Tok/s (GPU) |
|---|---|---|---|---|
| [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 2B | 3.60 GB | 5.4s | **19.9** |
| [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | 4B | 6.48 GB | 7.0s | **14.2** |
| [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 8B | 5.71 GB | 5.8s | **13.1** |

All models: INT4 weight format (group-size 128, ratio 0.8) · OpenVINO 2026.0 · optimum-intel 1.27.0.dev0

## Multi-Model Comparison (Intel Core Ultra iGPU, 300 max tokens)

### Performance Summary

| Model | Avg Tok/s | Avg Time(s) | Avg Tokens | Load Time | Disk Size |
|---|---|---|---|---|---|
| **2B** | **19.9** | 6.98 | 176 | 5.4s | 3.60 GB |
| **4B** | 14.2 | 10.53 | 191 | 7.0s | 6.48 GB |
| **8B** | 13.1–13.4 | 22.4–22.9 | 300 | 5.8s | 5.71 GB |

> **Note:** 8B tested individually per image/video (GPU memory limits prevent sequential multi-inference in one session).

### Detailed Results — Image Captioning

| Model | File | Tokens | Time(s) | Tok/s |
|---|---|---|---|---|
| 2B | demo.jpeg | 300 | 10.40 | 28.8 |
| 4B | demo.jpeg | 300 | 15.32 | 19.6 |
| 8B | demo.jpeg | 300 | 22.40 | 13.4 |
| 2B | 吃牛肉麵照片.png | 300 | 10.70 | 28.0 |
| 4B | 吃牛肉麵照片.png | 300 | 15.54 | 19.3 |
| 8B | 吃牛肉麵照片.png | 300 | 22.88 | 13.1 |

### Detailed Results — Video Understanding

| Model | File | Tokens | Time(s) | Tok/s |
|---|---|---|---|---|
| 2B | demo_video.mp4 | 135 | 4.97 | 27.2 |
| 4B | demo_video.mp4 | 300 | 14.25 | 21.1 |
| 8B | demo_video.mp4 | 122 | 9.55 | 12.8 |
| 2B | 牛肉麵影片.mp4 | 187 | 7.98 | 23.4 |
| 4B | 牛肉麵影片.mp4 | 203 | 11.60 | 17.5 |
| 2B | 計程車繁忙都市街口短影片.mp4 | 300 | 10.89 | 27.6 |
| 4B | 計程車繁忙都市街口短影片.mp4 | 224 | 12.33 | 18.2 |

### Key Findings

1. **2B is fastest** — ~2× throughput vs 8B (19.9 vs 13.1 tok/s average), great for latency-sensitive use cases
2. **4B is the sweet spot** — Better output quality than 2B with reasonable speed (14.2 tok/s)
3. **8B has highest quality** — Most detailed and structured descriptions, but slowest and hits GPU memory limits with sequential inference
4. **All models handle OCR** — Correctly identify "no text" in images without text; larger models give more nuanced OCR
5. **Video understanding works across all sizes** — 2B gives concise summaries; 4B and 8B provide step-by-step narratives
6. **GPU memory constraint** — 8B requires fresh model load per inference on Intel iGPU; 2B and 4B can run many inferences sequentially

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
# Convert default 8B model
python qwen3_vl.py --convert-only

# Convert specific model size
python qwen3_vl.py --model-size 2b --convert-only
python qwen3_vl.py --model-size 4b --convert-only
python qwen3_vl.py --model-size 8b --convert-only
```

Converts to INT4 OpenVINO IR at `C:\working\models\Qwen3-VL-{size}-Instruct\INT4`.

### Image Captioning

```bash
python qwen3_vl.py --task caption --image photo.jpg --device GPU --skip-conversion
python qwen3_vl.py --model-size 2b --task caption --image photo.jpg --device GPU --skip-conversion
```

### Visual Q&A

```bash
python qwen3_vl.py --task vqa --image photo.jpg --question "What color is the car?" --device GPU --skip-conversion
```

### Video Understanding

```bash
python qwen3_vl.py --task video --video clip.mp4 --question "What happens in this video?" --device GPU --skip-conversion
```

### Multi-Model Comparison

```bash
# Compare all available models
python qwen3_vl.py --compare --device GPU --max-tokens 300

# Compare specific sizes
python qwen3_vl.py --compare --compare-sizes 2b,4b --device GPU --max-tokens 300
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
| `--model-size` | — | Model size shortcut: `2b`, `4b`, `8b` |
| `--model-id` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID |
| `--model-dir` | `C:\working\models\Qwen3-VL-8B-Instruct\INT4` | OpenVINO model directory |
| `--weight-format` | `int4` | Weight format: `fp16`, `int8`, `int4` |
| `--max-tokens` | `200` | Max output tokens |
| `--convert-only` | — | Only convert, don't infer |
| `--skip-conversion` | — | Skip conversion step |
| `--benchmark N` | — | Run N benchmark iterations |
| `--no-stream` | — | Disable streaming output |
| `--compare` | — | Compare all available models |
| `--compare-sizes` | all | Comma-separated sizes to compare, e.g. `2b,4b` |

## Sample Outputs

### demo.jpeg — Woman with Dog on Beach

![demo.jpeg](demo.jpeg)

**Caption:**
> This is a heartwarming and serene photograph capturing a tender moment between a young woman and her golden Labrador Retriever on a sandy beach at sunset.

**VQA** — "How many people and animals are in this photo?":
> There is **one person** in the photo: a woman sitting on the sand. There is **one animal** in the photo: a yellow Labrador Retriever.

---

### 吃牛肉麵照片.png — Beef Noodle Soup

![吃牛肉麵照片.png](吃牛肉麵照片.png)

**Caption:**
> This is a vibrant, candid photograph capturing a joyful moment of a young man enjoying a meal in a bustling, traditional Asian noodle shop. He is actively eating, holding chopsticks in his right hand, lifting a large, steaming portion of noodles and a piece of beef towards his mouth. The soup contains thick, yellow noodles, tender-looking beef slices, chopped green onions, and a garnish of fresh cilantro and bok choy.

**VQA** — "What food is this person eating? What ingredients can you identify?":
> The person is eating a bowl of **beef noodle soup**. Identifiable ingredients: thick yellow wheat noodles, large tender chunks of braised beef, rich dark broth, bok choy, fresh cilantro, chopped green onions, red chili sauce, and pickled greens.

---

### Video Samples

**demo_video.mp4** — "What do you see in this video?":
> A woman sits on a sandy beach at sunset, smiling as she interacts with a golden retriever. The dog, wearing a harness, sits upright and reaches its paw toward the woman, who holds a small treat in her hand.

**計程車繁忙都市街口短影片.mp4** — "Describe in detail what is happening in this video." (busy city intersection):
> The video captures the bustling energy of Times Square in New York City. The scene opens with a yellow taxi cab prominently positioned in the foreground, its iconic color contrasting with the towering skyscrapers and bright billboards that dominate the background. As the camera pans to the right, the focus shifts to the dense crowd of pedestrians crossing the street. People of all ages and backgrounds are seen walking with purpose. In the background, a red double-decker bus stands out against the backdrop of towering glass buildings.

**牛肉麵影片.mp4** — "Describe in detail what is happening in this video." (eating ramen):
> A young woman is enjoying a steaming bowl of ramen in a warmly lit, cozy restaurant. She holds the large white bowl with one hand, using chopsticks to lift a generous portion of noodles and slices of tender beef toward her mouth. The ramen is rich and aromatic, with visible toppings including chopped green onions, slices of beef, and a flavorful broth that steams gently. The background is softly blurred with warm, glowing lights creating a bokeh effect.

## Multilingual Support (繁體中文)

Qwen3-VL natively supports multilingual output including **Traditional Chinese**. Simply ask your question in Chinese or add an instruction like `請用繁體中文回答`.

### Example — Traditional Chinese Response

```bash
python qwen3_vl.py --task caption --image "吃牛肉麵照片.png" --question "請用繁體中文描述這張圖片。" --device GPU --skip-conversion
```

**Output:**
> 這張圖片捕捉了一位年輕男子在熱鬧的中式麵館裡享用一碗熱騰騰牛肉麵的歡樂瞬間。他身穿深藍色T恤，正用筷子夾起一大口麵條與牛肉，同時用湯匙舀著湯頭，臉上洋溢著開心滿足的笑容，眼睛因笑容而眯起，顯得極度享受這頓美食。他面前的木桌上擺著一碗湯頭濃郁、配料豐富的牛肉麵，碗裡有麵條、大塊牛肉、翠綠的香菜與青菜，熱氣騰騰，令人食指大動。背景是典型的台灣或東南亞街頭小吃店或老字號麵館，木製桌椅、熱鬧的用餐客人、廚房的煙霧，以及牆上貼著的菜單或海報，營造出濃厚的庶民美食氛圍。

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
