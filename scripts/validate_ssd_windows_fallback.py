from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import LlamaConfig


def ensure_env(root: Path) -> None:
    os.environ.setdefault("SSD_HF_CACHE", str(root / "tmp" / "hf-cache"))
    os.environ.setdefault("SSD_DATASET_DIR", str(root / "tmp" / "datasets"))


def ensure_upstream_on_path(root: Path) -> None:
    upstream = root / "upstream-ssd"
    if str(upstream) not in os.sys.path:
        os.sys.path.insert(0, str(upstream))


def build_tiny_checkpoint(root: Path, model_dir: Path) -> Path:
    ensure_upstream_on_path(root)
    from ssd.models.llama3 import LlamaForCausalLM

    model_dir.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=500000.0,
        torch_dtype="float32",
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    config.save_pretrained(model_dir)
    from safetensors.torch import save_file

    save_file(model.state_dict(), str(model_dir / "model.safetensors"))
    return model_dir


def run_validation(root: Path, model_dir: Path) -> dict:
    ensure_upstream_on_path(root)
    from ssd.compat import format_runtime_summary
    from ssd.config import Config
    from ssd.engine.model_runner import ModelRunner
    from ssd.engine.sequence import Sequence

    config = Config(
        model=str(model_dir),
        max_num_seqs=1,
        max_model_len=16,
        max_num_batched_tokens=16,
        num_gpus=1,
        enforce_eager=True,
        verbose=False,
    )
    runner = ModelRunner(config, rank=0, event=[], is_draft=False, num_tp_gpus=1)
    seqs = [Sequence([1, 2, 3])]
    token_ids = runner.run(seqs, is_prefill=True)
    return {
        "runtime": format_runtime_summary(),
        "device": str(runner.device),
        "cuda_backend": runner.cuda_backend,
        "returned_type": type(token_ids).__name__,
        "returned_tokens": token_ids,
        "kv_cache_shape": list(runner.kv_cache.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the experimental Windows SSD fallback path.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    ensure_env(root)
    model_dir = args.model_dir.resolve() if args.model_dir else root / "tmp" / "tiny-llama"
    build_tiny_checkpoint(root, model_dir)
    result = run_validation(root, model_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
