from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path


DEFAULT_MODELS_DIR = Path(r"C:\Users\AMD\.lmstudio\models")
DEFAULT_MODEL_DATA = Path(r"C:\Users\AMD\.lmstudio\.internal\model-data.json")
DEFAULT_OUTPUT = Path("artifacts/model_inventory.json")

PARAM_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*B(?!\w)", re.IGNORECASE)


def is_auxiliary_model(path: Path) -> bool:
    name = path.name.lower()
    text = path.as_posix().lower()
    return (
        name.startswith("mmproj")
        or "embedding" in text
        or "minilm" in text
    )


def is_multimodal_model(path: Path) -> bool:
    text = path.as_posix().lower()
    return "-vl-" in text or ".6v-" in text or "vision" in text


def infer_param_billions(path: Path) -> float | None:
    match = PARAM_RE.search(path.as_posix())
    return float(match.group(1)) if match else None


def read_last_loaded_map(model_data_path: Path) -> dict[str, int]:
    if not model_data_path.exists():
        return {}
    payload = json.loads(model_data_path.read_text(encoding="utf-8"))
    entries = payload.get("json", [])
    loaded: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        key, value = entry
        if not isinstance(value, dict):
            continue
        ts = value.get("lastLoadedTimestamp")
        if isinstance(ts, int):
            loaded[str(key).replace("/", "\\")] = ts
    return loaded


def score_model(path: Path, param_b: float | None, size_gib: float) -> tuple[int, str]:
    if is_auxiliary_model(path):
        return (0, "auxiliary or embedding artifact; skip for text-generation benchmarks")
    if is_multimodal_model(path):
        return (1, "multimodal model; keep separate from text-only baseline runs")
    if size_gib > 20:
        return (0, "too large for practical Windows baseline work on 32 GB RAM")
    if param_b is not None and param_b <= 8 and size_gib <= 8:
        return (3, "recommended benchmark candidate")
    if size_gib <= 12:
        return (2, "reasonable if memory pressure stays low")
    return (1, "possible but likely slower or tighter on memory")


def discover_models(models_dir: Path, model_data_path: Path) -> list[dict[str, object]]:
    last_loaded_map = read_last_loaded_map(model_data_path)
    models: list[dict[str, object]] = []
    for file_path in sorted(models_dir.rglob("*.gguf")):
        rel = file_path.relative_to(models_dir)
        size_gib = file_path.stat().st_size / (1024 ** 3)
        param_b = infer_param_billions(rel)
        score, note = score_model(rel, param_b, size_gib)
        rel_key = rel.as_posix().replace("/", "\\")
        last_loaded = last_loaded_map.get(rel_key)
        last_loaded_iso = None
        if last_loaded:
            last_loaded_iso = dt.datetime.fromtimestamp(last_loaded / 1000, dt.timezone.utc).isoformat()
        models.append(
            {
                "path": str(rel).replace("/", "\\"),
                "size_gib": round(size_gib, 2),
                "params_b": param_b,
                "recommendation_score": score,
                "note": note,
                "last_loaded_utc": last_loaded_iso,
            }
        )
    models.sort(key=lambda item: (-int(item["recommendation_score"]), float(item["size_gib"])))
    return models


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect LM Studio GGUF models and rank local benchmark candidates.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model-data", type=Path, default=DEFAULT_MODEL_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    models = discover_models(args.models_dir, args.model_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(models, indent=2), encoding="utf-8")

    print(f"Discovered {len(models)} GGUF models in {args.models_dir}")
    print(f"Wrote inventory to {args.output}")
    print()
    print("Top candidates:")
    for item in models[:8]:
        print(
            f"- {item['path']} | {item['size_gib']} GiB"
            f" | params={item['params_b']}"
            f" | score={item['recommendation_score']}"
            f" | {item['note']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
