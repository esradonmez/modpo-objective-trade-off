"""Score objective-study generations with ArmoRM-Llama3-8B-v0.1."""

import argparse
import json
from pathlib import Path

import torch
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DOCUMENTED_ATTRIBUTES = {
    0: "helpsteer-helpfulness",
    1: "helpsteer-correctness",
    2: "helpsteer-coherence",
    3: "helpsteer-complexity",
    4: "helpsteer-verbosity",
    5: "ultrafeedback-overall_score",
    6: "ultrafeedback-instruction_following",
    7: "ultrafeedback-truthfulness",
    8: "ultrafeedback-honesty",
    9: "ultrafeedback-helpfulness",
    10: "beavertails-is_safe",
    11: "prometheus-score",
    12: "argilla-overall_quality",
    13: "argilla-judge_lm",
    14: "code-complexity",
    15: "code-style",
    16: "code-explanation",
    17: "code-instruction-following",
    18: "code-readability",
}
SCORED_ATTRIBUTES = {
    "helpsteer-helpfulness",
    "helpsteer-correctness",
    "helpsteer-coherence",
    "helpsteer-complexity",
    "helpsteer-verbosity",
    "ultrafeedback-instruction_following",
    "ultrafeedback-truthfulness",
    "ultrafeedback-honesty",
    "ultrafeedback-helpfulness",
}
GENERATION_FIELDS = {
    "source_row_index",
    "pair_id",
    "raw_prompt",
    "prompt",
    "response",
    "prompt_response",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def canonicalize(raw) -> dict[int, str]:
    if raw is None:
        return {}
    items = raw.items() if isinstance(raw, dict) else enumerate(raw)
    output = {}
    for key, value in items:
        try:
            output[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return dict(sorted(output.items()))


def resolve_attributes(model) -> dict[int, str]:
    candidates = []
    if hasattr(model, "score") and hasattr(model.score, "attributes"):
        candidates.append(canonicalize(model.score.attributes))
    if hasattr(model.config, "attributes"):
        candidates.append(canonicalize(model.config.attributes))
    if getattr(model.config, "id2label", None):
        candidates.append(canonicalize(model.config.id2label))
    candidates.append(DOCUMENTED_ATTRIBUTES)
    for candidate in candidates:
        if SCORED_ATTRIBUTES.issubset(set(candidate.values())):
            return candidate
    raise ValueError("ArmoRM metadata does not expose every required score head.")


def load_rows(input_dir: Path) -> list[dict]:
    paths = sorted(input_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No JSONL generation files found in {input_dir}.")
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object.")
                missing = sorted(GENERATION_FIELDS - set(row))
                if missing:
                    raise ValueError(f"{path}:{line_number} is missing fields: {', '.join(missing)}")
                source_row_index = row["source_row_index"]
                if isinstance(source_row_index, bool) or not isinstance(source_row_index, int) or source_row_index < 0:
                    raise ValueError(f"{path}:{line_number} has an invalid source_row_index.")
                if row["pair_id"] != f"validation:{source_row_index}":
                    raise ValueError(f"{path}:{line_number} has an inconsistent pair_id.")
                for field in ("raw_prompt", "prompt", "response", "prompt_response"):
                    if not isinstance(row[field], str):
                        raise ValueError(f"{path}:{line_number} field {field!r} is not text.")
                if not row["raw_prompt"] or not row["prompt"]:
                    raise ValueError(f"{path}:{line_number} has an empty prompt.")
                rows.append(row)
    pair_ids = [row["pair_id"] for row in rows]
    if not rows:
        raise ValueError("Generation input files contain no rows.")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Generation inputs contain duplicate pair_id values.")
    return rows


def load_evaluator_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Objective configuration must use schema_version 1.")
    evaluator = config.get("evaluator")
    if not isinstance(evaluator, dict):
        raise ValueError("Objective configuration is missing its evaluator.")
    return evaluator


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    evaluator = load_evaluator_config(args.config)
    rows = load_rows(args.input_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if evaluator.get("attn_implementation"):
        model_kwargs["attn_implementation"] = evaluator["attn_implementation"]
    model = AutoModelForSequenceClassification.from_pretrained(
        evaluator["name"],
        revision=evaluator["revision"],
        **model_kwargs,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        evaluator["name"],
        revision=evaluator["revision"],
        use_fast=True,
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("ArmoRM requires a tokenizer with apply_chat_template().")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id

    attributes = resolve_attributes(model)
    scored_indices = {name: index for index, name in attributes.items() if name in SCORED_ATTRIBUTES}
    max_index = max(scored_indices.values())
    results = []
    for offset in tqdm.tqdm(range(0, len(rows), args.batch_size)):
        batch = rows[offset : offset + args.batch_size]
        chats = [
            [
                {"role": "user", "content": row["raw_prompt"]},
                {"role": "assistant", "content": row["response"]},
            ]
            for row in batch
        ]
        texts = [
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            for chat in chats
        ]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(evaluator["max_length"]),
        ).to(device)
        with torch.inference_mode():
            rewards = model(**encoded).rewards.detach().cpu().float()
        if rewards.ndim != 2 or rewards.shape[0] != len(batch) or rewards.shape[1] <= max_index:
            raise ValueError(f"Unexpected ArmoRM reward tensor shape: {tuple(rewards.shape)}")
        for row_index, row in enumerate(batch):
            scored = dict(row)
            for name, index in scored_indices.items():
                scored[f"armorm_{name}"] = rewards[row_index, index].item()
            results.append(scored)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "scores_armorm.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    for name in sorted(SCORED_ATTRIBUTES):
        key = f"armorm_{name}"
        values = [row[key] for row in results]
        print(f"{name}: {sum(values) / len(values):.6f}")
    print(f"Saved {len(results)} scores to {output_path}")


if __name__ == "__main__":
    main()
