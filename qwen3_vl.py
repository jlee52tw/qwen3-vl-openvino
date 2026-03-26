#!/usr/bin/env python3
"""
Qwen3-VL OpenVINO Standalone Project
=====================================
Visual-language assistant using Qwen3-VL with OpenVINO acceleration.
Supports image captioning, VQA, and video understanding.

Based on: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/qwen3-vl

Usage examples:
    # Convert model (INT4)
    python qwen3_vl.py --convert-only

    # Image captioning
    python qwen3_vl.py --task caption --image demo.jpeg --device GPU

    # Visual Q&A
    python qwen3_vl.py --task vqa --image demo.jpeg --question "What is the woman doing?" --device GPU

    # Video understanding
    python qwen3_vl.py --task video --video video.mp4 --question "Describe what happens" --device GPU

    # Benchmark mode
    python qwen3_vl.py --task caption --image demo.jpeg --device GPU --benchmark 5
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "2b": {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "model_dir": r"C:\working\models\Qwen3-VL-2B-Instruct\INT4",
    },
    "4b": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "model_dir": r"C:\working\models\Qwen3-VL-4B-Instruct\INT4",
    },
    "8b": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_dir": r"C:\working\models\Qwen3-VL-8B-Instruct\INT4",
    },
}

DEFAULT_MODEL_SIZE = "8b"
DEFAULT_MODEL_ID = MODEL_CONFIGS[DEFAULT_MODEL_SIZE]["model_id"]
DEFAULT_MODEL_DIR = MODEL_CONFIGS[DEFAULT_MODEL_SIZE]["model_dir"]
DEFAULT_DEVICE = "GPU"
DEFAULT_MAX_TOKENS = 200

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

DEMO_IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
DEMO_IMAGE_PATH = Path("demo.jpeg")


# ─── Conversion ──────────────────────────────────────────────────────────────
def convert_model(model_id: str, output_dir: str, weight_format: str = "int4",
                  group_size: int = 128, ratio: float = 0.8) -> None:
    """Convert HuggingFace model to OpenVINO IR using optimum-cli."""
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.glob("*.xml")):
        print(f"[INFO] Model already exists at {output_dir}, skipping conversion.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Find optimum-cli executable – try venv Scripts, then next to sys.executable
    optimum_cli = None
    candidates = [
        Path(os.environ.get("VIRTUAL_ENV", "")) / "Scripts" / "optimum-cli.exe",
        Path(sys.executable).parent / "optimum-cli.exe",
        Path(sys.executable).parent / "optimum-cli",
    ]
    for c in candidates:
        if c.exists():
            optimum_cli = c
            break

    if optimum_cli:
        cmd = [str(optimum_cli)]
    else:
        # Fallback: invoke via python -m
        cmd = [sys.executable, "-m", "optimum.exporters.openvino"]

    cmd += [
        "export", "openvino",
        "--model", model_id,
        "--task", "image-text-to-text",
        "--weight-format", weight_format,
        "--group-size", str(group_size),
        "--ratio", str(ratio),
        str(output_dir),
    ]

    print(f"[INFO] Converting model: {model_id}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Weight format: {weight_format}, group-size: {group_size}, ratio: {ratio}")
    print(f"[INFO] Command: {' '.join(cmd)}")
    print()

    start = time.time()
    result = subprocess.run(cmd, env=os.environ.copy())
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"[ERROR] Conversion failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n[INFO] Conversion completed in {elapsed:.1f}s")

    # Print model directory size
    total_size = sum(f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file())
    print(f"[INFO] Model directory size: {total_size / (1024**3):.2f} GB")


# ─── Model Loading ───────────────────────────────────────────────────────────
def load_model(model_dir: str, device: str):
    """Load the OpenVINO model and processor."""
    from optimum.intel.openvino import OVModelForVisualCausalLM
    from transformers import AutoProcessor

    print(f"[INFO] Loading model from {model_dir} on {device}...")
    start = time.time()
    model = OVModelForVisualCausalLM.from_pretrained(model_dir, device=device)
    load_time = time.time() - start
    print(f"[INFO] Model loaded in {load_time:.1f}s")

    processor = AutoProcessor.from_pretrained(
        model_dir, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )

    return model, processor, load_time


# ─── Demo Image ──────────────────────────────────────────────────────────────
def ensure_demo_image() -> Path:
    """Download demo image if not present."""
    if not DEMO_IMAGE_PATH.exists():
        import requests
        print(f"[INFO] Downloading demo image from {DEMO_IMAGE_URL}")
        img = Image.open(requests.get(DEMO_IMAGE_URL, stream=True).raw)
        img.save(DEMO_IMAGE_PATH)
        print(f"[INFO] Saved to {DEMO_IMAGE_PATH}")
    return DEMO_IMAGE_PATH


# ─── Inference ────────────────────────────────────────────────────────────────
def run_image_inference(model, processor, image_path: str, question: str,
                        max_tokens: int = DEFAULT_MAX_TOKENS,
                        stream: bool = True) -> dict:
    """Run inference on a single image. Returns dict with output text and timing."""
    from transformers import TextStreamer

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    # Tokenize
    t0 = time.time()
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    preprocess_time = time.time() - t0
    input_tokens = inputs["input_ids"].shape[-1]

    # Generate
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

    t1 = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, streamer=streamer)
    generate_time = time.time() - t1

    output_tokens = generated_ids.shape[-1] - input_tokens

    # Decode if not streamed
    output_text = processor.tokenizer.decode(
        generated_ids[0][input_tokens:], skip_special_tokens=True
    )

    # Compute metrics
    first_token_time = generate_time / output_tokens if output_tokens > 0 else 0
    tokens_per_sec = output_tokens / generate_time if generate_time > 0 else 0

    result = {
        "image": str(image_path),
        "image_size": f"{w}x{h}",
        "question": question,
        "answer": output_text.strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "preprocess_time_s": round(preprocess_time, 3),
        "generate_time_s": round(generate_time, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "total_time_s": round(preprocess_time + generate_time, 3),
    }

    return result


def run_video_inference(model, processor, video_path: str, question: str,
                        max_tokens: int = DEFAULT_MAX_TOKENS,
                        stream: bool = True) -> dict:
    """Run inference on a video file. Returns dict with output text and timing."""
    from transformers import TextStreamer

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "max_pixels": MAX_PIXELS,
                    "min_pixels": MIN_PIXELS,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    # Use qwen_vl_utils to process video frames
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    t0 = time.time()
    inputs = processor(
        text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    preprocess_time = time.time() - t0
    input_tokens = inputs["input_ids"].shape[-1]

    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

    t1 = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, streamer=streamer)
    generate_time = time.time() - t1

    output_tokens = generated_ids.shape[-1] - input_tokens

    output_text = processor.tokenizer.decode(
        generated_ids[0][input_tokens:], skip_special_tokens=True
    )

    tokens_per_sec = output_tokens / generate_time if generate_time > 0 else 0

    result = {
        "video": str(video_path),
        "question": question,
        "answer": output_text.strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "preprocess_time_s": round(preprocess_time, 3),
        "generate_time_s": round(generate_time, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "total_time_s": round(preprocess_time + generate_time, 3),
    }

    return result


# ─── Benchmark ────────────────────────────────────────────────────────────────
def run_benchmark(model, processor, args, iterations: int = 5) -> None:
    """Run multiple iterations and report statistics."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {iterations} iterations")
    print(f"  Task: {args.task}, Device: {args.device}")
    if args.task in ("caption", "vqa"):
        print(f"  Image: {args.image}")
    elif args.task == "video":
        print(f"  Video: {args.video}")
    print(f"{'='*60}\n")

    results = []
    for i in range(iterations):
        print(f"--- Iteration {i+1}/{iterations} ---")

        if args.task in ("caption", "vqa"):
            question = args.question or ("Describe this image." if args.task == "caption" else "What do you see?")
            r = run_image_inference(model, processor, args.image, question,
                                   max_tokens=args.max_tokens, stream=False)
        elif args.task == "video":
            question = args.question or "Describe what happens in this video."
            r = run_video_inference(model, processor, args.video, question,
                                   max_tokens=args.max_tokens, stream=False)
        else:
            print(f"[ERROR] Unknown task: {args.task}")
            return

        print(f"  Generate: {r['generate_time_s']:.3f}s | "
              f"Tokens: {r['output_tokens']} | "
              f"Speed: {r['tokens_per_sec']:.1f} tok/s")
        results.append(r)

    # Statistics
    gen_times = [r["generate_time_s"] for r in results]
    tok_speeds = [r["tokens_per_sec"] for r in results]
    total_times = [r["total_time_s"] for r in results]
    out_tokens = [r["output_tokens"] for r in results]

    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS ({iterations} iterations)")
    print(f"{'='*60}")
    print(f"  Generate time  : median={np.median(gen_times):.3f}s, "
          f"mean={np.mean(gen_times):.3f}s, std={np.std(gen_times):.3f}s")
    print(f"  Total time     : median={np.median(total_times):.3f}s, "
          f"mean={np.mean(total_times):.3f}s")
    print(f"  Tokens/sec     : median={np.median(tok_speeds):.1f}, "
          f"mean={np.mean(tok_speeds):.1f}")
    print(f"  Output tokens  : median={np.median(out_tokens):.0f}, "
          f"mean={np.mean(out_tokens):.0f}")
    print(f"  Min/Max gen    : {min(gen_times):.3f}s / {max(gen_times):.3f}s")
    print(f"{'='*60}")

    # Print last answer as sample
    print(f"\n  Sample answer: {results[-1]['answer'][:200]}...")


# ─── GPU Memory ───────────────────────────────────────────────────────────────
def get_gpu_memory_mb() -> float:
    """Get Intel GPU shared/dedicated memory usage in MB via openvino properties."""
    try:
        import openvino as ov
        core = ov.Core()
        props = core.get_property("GPU", "GPU_MEMORY_STATISTICS")
        # props is a dict-like with device memory usage
        total = sum(v for v in props.values() if isinstance(v, (int, float)))
        return total / (1024 * 1024)
    except Exception:
        return -1.0


# ─── Compare Mode ─────────────────────────────────────────────────────────────
def run_comparison(args):
    """Compare all available models on the same image/video inputs."""
    import gc

    # Determine which models to compare
    requested_sizes = None
    if hasattr(args, 'compare_sizes') and args.compare_sizes:
        requested_sizes = [s.strip().lower() for s in args.compare_sizes.split(',')]

    sizes_to_test = []
    for size, cfg in MODEL_CONFIGS.items():
        if requested_sizes and size not in requested_sizes:
            continue
        model_path = Path(cfg["model_dir"])
        if model_path.exists() and any(model_path.glob("*.xml")):
            sizes_to_test.append(size)
        else:
            print(f"[WARN] Model {size.upper()} not found at {cfg['model_dir']}, skipping.")

    if not sizes_to_test:
        print("[ERROR] No converted models found. Run conversion first.")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"  MODEL COMPARISON: {', '.join(s.upper() for s in sizes_to_test)}")
    print(f"  Device: {args.device} | Max tokens: {args.max_tokens}")
    print(f"{'='*80}")

    # Define test cases
    test_cases = []

    # Image tests
    image_files = list(Path(".").glob("*.jpeg")) + list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.png"))
    for img in image_files:
        test_cases.append({
            "type": "image",
            "path": str(img),
            "questions": [
                ("caption", "Describe this image in detail."),
                ("ocr", "Read and transcribe all visible text in this image. If there is no text, say 'No text found'."),
            ]
        })

    # Video tests
    video_files = list(Path(".").glob("*.mp4"))
    for vid in video_files:
        test_cases.append({
            "type": "video",
            "path": str(vid),
            "questions": [
                ("video_understanding", "Describe what happens in this video step by step."),
            ]
        })

    if not test_cases:
        print("[ERROR] No test files (.jpg/.jpeg/.png/.mp4) found in current directory.")
        sys.exit(1)

    print(f"\n  Test files: {len(test_cases)}")
    for tc in test_cases:
        print(f"    [{tc['type']}] {tc['path']}")

    all_results = {}  # {model_size: [{...}, ...]}

    for size in sizes_to_test:
        cfg = MODEL_CONFIGS[size]
        all_results[size] = []

        print(f"\n{'#'*80}")
        print(f"  Loading model: Qwen3-VL-{size.upper()}-Instruct (INT4)")
        print(f"{'#'*80}")

        # Get model dir size
        model_path = Path(cfg["model_dir"])
        model_size_gb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / (1024**3)
        print(f"  Model size on disk: {model_size_gb:.2f} GB")

        model, processor, load_time = load_model(cfg["model_dir"], args.device)
        oom_abort = False

        for tc in test_cases:
            for task_name, question in tc["questions"]:
                print(f"\n  --- {size.upper()} | {task_name} | {tc['path']} ---")
                print(f"  Q: {question}")

                try:
                    if tc["type"] == "image":
                        result = run_image_inference(
                            model, processor, tc["path"], question,
                            max_tokens=args.max_tokens, stream=False
                        )
                    else:
                        result = run_video_inference(
                            model, processor, tc["path"], question,
                            max_tokens=args.max_tokens, stream=False
                        )

                    result["model_size"] = size
                    result["model_size_gb"] = round(model_size_gb, 2)
                    result["load_time_s"] = round(load_time, 1)
                    result["task_name"] = task_name
                    all_results[size].append(result)

                    answer_preview = result["answer"][:300]
                    print(f"  A: {answer_preview}")
                    print(f"  >> {result['output_tokens']} tokens | "
                          f"{result['generate_time_s']:.2f}s | "
                          f"{result['tokens_per_sec']:.1f} tok/s")

                except Exception as e:
                    err_str = str(e)
                    print(f"  [ERROR] {err_str[:200]}")
                    all_results[size].append({
                        "model_size": size, "task_name": task_name,
                        "error": err_str, "path": tc["path"],
                    })
                    # If GPU OOM, abort remaining tests for this model
                    if "CL_OUT_OF_RESOURCES" in err_str or "out of memory" in err_str.lower():
                        print(f"  [WARN] GPU OOM detected — skipping remaining tests for {size.upper()}")
                        oom_abort = True
                        break
            if oom_abort:
                break

        # Free model memory
        try:
            del model, processor
            gc.collect()
        except Exception:
            pass
        print(f"\n  [INFO] Model {size.upper()} unloaded.")

    # ── Print comparison table ──
    print(f"\n\n{'='*100}")
    print(f"  COMPARISON RESULTS")
    print(f"{'='*100}")

    # Header
    print(f"\n{'Model':<8} {'Task':<22} {'File':<35} {'Tokens':<8} {'Time(s)':<10} {'Tok/s':<8} {'Load(s)':<8} {'Size(GB)':<9}")
    print("-" * 108)

    for size in sizes_to_test:
        for r in all_results[size]:
            if "error" in r:
                fname = Path(r.get("path", "?")).name
                print(f"{size.upper():<8} {r['task_name']:<22} {fname:<35} {'ERROR':<8}")
                continue
            fname = Path(r.get("image", r.get("video", "?"))).name
            print(f"{size.upper():<8} {r['task_name']:<22} {fname:<35} "
                  f"{r['output_tokens']:<8} {r['generate_time_s']:<10.2f} "
                  f"{r['tokens_per_sec']:<8.1f} {r['load_time_s']:<8.1f} "
                  f"{r['model_size_gb']:<9.2f}")

    # Per-model summary
    print(f"\n{'='*100}")
    print(f"  PER-MODEL SUMMARY")
    print(f"{'='*100}")
    print(f"\n{'Model':<8} {'Avg Tok/s':<12} {'Avg Time(s)':<14} {'Avg Tokens':<12} {'Load(s)':<10} {'Size(GB)':<10}")
    print("-" * 66)

    for size in sizes_to_test:
        valid = [r for r in all_results[size] if "error" not in r]
        if not valid:
            print(f"{size.upper():<8} {'N/A'}")
            continue
        avg_tps = np.mean([r["tokens_per_sec"] for r in valid])
        avg_time = np.mean([r["generate_time_s"] for r in valid])
        avg_tokens = np.mean([r["output_tokens"] for r in valid])
        load_t = valid[0]["load_time_s"]
        disk_gb = valid[0]["model_size_gb"]
        print(f"{size.upper():<8} {avg_tps:<12.1f} {avg_time:<14.2f} {avg_tokens:<12.0f} {load_t:<10.1f} {disk_gb:<10.2f}")

    print(f"\n{'='*100}")

    # Print answers side by side for each test case
    print(f"\n\n{'='*100}")
    print(f"  OUTPUT QUALITY COMPARISON")
    print(f"{'='*100}")

    for tc in test_cases:
        for task_name, question in tc["questions"]:
            fname = Path(tc["path"]).name
            print(f"\n{'─'*100}")
            print(f"  [{task_name}] {fname}")
            print(f"  Q: {question}")
            print(f"{'─'*100}")

            for size in sizes_to_test:
                matching = [r for r in all_results[size]
                           if r.get("task_name") == task_name
                           and "error" not in r
                           and Path(r.get("image", r.get("video", ""))).name == fname]
                if matching:
                    r = matching[0]
                    print(f"\n  [{size.upper()}] ({r['output_tokens']} tokens, {r['tokens_per_sec']:.1f} tok/s)")
                    print(f"  {r['answer'][:500]}")
                else:
                    print(f"\n  [{size.upper()}] SKIPPED/ERROR")

    print(f"\n{'='*100}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL OpenVINO — Visual Language Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", choices=["caption", "vqa", "video"],
                        default="caption", help="Task type (default: caption)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image (uses demo image if not specified)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input video file")
    parser.add_argument("--question", type=str, default=None,
                        help="Question for VQA/video task")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help=f"Inference device (default: {DEFAULT_DEVICE})")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"OpenVINO model directory (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--weight-format", type=str, default="int4",
                        choices=["fp16", "int8", "int4"],
                        help="Weight compression format (default: int4)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max output tokens (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert model, don't run inference")
    parser.add_argument("--skip-conversion", action="store_true",
                        help="Skip model conversion step")
    parser.add_argument("--benchmark", type=int, default=0, metavar="N",
                        help="Run N benchmark iterations")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming output")
    parser.add_argument("--model-size", type=str, default=None,
                        choices=["2b", "4b", "8b"],
                        help="Model size shortcut (overrides --model-id and --model-dir)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all available models (2B, 4B, 8B) on the same input")
    parser.add_argument("--compare-sizes", type=str, default=None,
                        help="Comma-separated model sizes to compare, e.g. '2b,4b' (default: all)")

    args = parser.parse_args()

    # ── Resolve model-size shortcut ──
    if args.model_size:
        cfg = MODEL_CONFIGS[args.model_size]
        args.model_id = cfg["model_id"]
        args.model_dir = cfg["model_dir"]

    # ── Compare mode ──
    if args.compare:
        run_comparison(args)
        return

    # ── Step 1: Convert model ──
    if not args.skip_conversion:
        convert_model(
            args.model_id, args.model_dir,
            weight_format=args.weight_format,
        )

    if args.convert_only:
        print("[INFO] Conversion complete. Exiting.")
        return

    # ── Step 2: Load model ──
    model, processor, load_time = load_model(args.model_dir, args.device)

    # ── Step 3: Prepare input ──
    if args.task in ("caption", "vqa"):
        if args.image is None:
            args.image = str(ensure_demo_image())
            print(f"[INFO] Using demo image: {args.image}")

        if not Path(args.image).exists():
            print(f"[ERROR] Image not found: {args.image}")
            sys.exit(1)

        if args.question is None:
            if args.task == "caption":
                args.question = "Describe this image."
            else:
                args.question = "What do you see in this image?"

    elif args.task == "video":
        if args.video is None:
            print("[ERROR] --video is required for video task")
            sys.exit(1)
        if not Path(args.video).exists():
            print(f"[ERROR] Video not found: {args.video}")
            sys.exit(1)
        if args.question is None:
            args.question = "Describe what happens in this video."

    # ── Step 4: Run inference or benchmark ──
    if args.benchmark > 0:
        run_benchmark(model, processor, args, iterations=args.benchmark)
    else:
        print(f"\n[Task: {args.task}]")
        if args.task in ("caption", "vqa"):
            print(f"[Image: {args.image}]")
            print(f"[Question: {args.question}]")
            print(f"[Device: {args.device}]")
            print()
            result = run_image_inference(
                model, processor, args.image, args.question,
                max_tokens=args.max_tokens, stream=not args.no_stream,
            )
        elif args.task == "video":
            print(f"[Video: {args.video}]")
            print(f"[Question: {args.question}]")
            print(f"[Device: {args.device}]")
            print()
            result = run_video_inference(
                model, processor, args.video, args.question,
                max_tokens=args.max_tokens, stream=not args.no_stream,
            )

        # Print summary
        print(f"\n{'─'*50}")
        print(f"  Input tokens  : {result['input_tokens']}")
        print(f"  Output tokens : {result['output_tokens']}")
        print(f"  Preprocess    : {result['preprocess_time_s']:.3f}s")
        print(f"  Generate      : {result['generate_time_s']:.3f}s")
        print(f"  Total         : {result['total_time_s']:.3f}s")
        print(f"  Tokens/sec    : {result['tokens_per_sec']:.1f}")
        print(f"{'─'*50}")


if __name__ == "__main__":
    main()
