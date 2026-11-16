"""Generate responses on a configured fixed 400-prompt evaluation set."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch
import tqdm
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.configs import DATASET_CONFIGS
from src.utils import disable_progress_bar_non_local_main, print_local_main, set_seeds

disable_progress_bar_non_local_main()

DIGEST_CANONICALIZATION = (
    "utf8(value) prefixed by unsigned 8-byte big-endian byte length, concatenated in order"
)
PROMPT_TEMPLATE_ID = "modpo_human_assistant_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--decoding-config", type=Path, required=True)
    parser.add_argument("--decode-profile", required=True)
    parser.add_argument("--sft-model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def ordered_sha256(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def load_configuration(args: argparse.Namespace) -> tuple[dict, dict]:
    config = load_json(args.config)
    if config.get("schema_version") != 1:
        raise ValueError("Objective configuration must use schema_version 1.")
    condition = config.get("conditions", {}).get(args.condition)
    if not isinstance(condition, dict):
        raise ValueError(f"Unknown condition {args.condition!r}.")

    decoding = load_json(args.decoding_config)
    if decoding.get("schema_version") != 1:
        raise ValueError("Decoding configuration must use schema_version 1.")
    profile = decoding.get("profiles", {}).get(args.decode_profile)
    if not isinstance(profile, dict):
        raise ValueError(f"Unknown decode profile {args.decode_profile!r}.")
    if profile.get("do_sample") is not False:
        raise ValueError("The submitted decode profiles must use greedy decoding.")
    if args.decode_profile == "robustness_2048":
        allowed = config.get("generation", {}).get("robustness_conditions", [])
        if args.condition not in allowed:
            raise ValueError(f"The robustness decode profile is not configured for {args.condition!r}.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    return config, profile


def resolve_manifest_path(config: dict) -> Path:
    path = Path(config["generation"]["manifest"])
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[3] / path


def select_and_verify_prompts(config: dict, eval_dataset: Dataset) -> tuple[Dataset, list[int]]:
    manifest_path = resolve_manifest_path(config)
    manifest = load_json(manifest_path)
    required = {
        "manifest_schema_version",
        "dataset_name",
        "dataset_revision",
        "source_split",
        "expected_source_split_rows",
        "selection_rule",
        "expected_selected_rows",
        "expected_unique_raw_prompts",
        "prompt_template_id",
        "hash_algorithm",
        "digest_canonicalization",
        "ordered_raw_prompt_sha256",
        "source_row_identifier",
        "ordered_source_row_identifier_sha256",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Evaluation manifest is missing fields: {', '.join(missing)}")

    dataset = config["dataset"]
    expected_values = {
        "manifest_schema_version": 1,
        "dataset_name": dataset["evaluation_view"],
        "dataset_revision": dataset["revision"],
        "source_split": "validation",
        "expected_source_split_rows": len(eval_dataset),
        "selection_rule": "first_400_after_exact_raw_prompt_deduplication_keep_first",
        "expected_selected_rows": 400,
        "expected_unique_raw_prompts": 400,
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "hash_algorithm": "sha256",
        "digest_canonicalization": DIGEST_CANONICALIZATION,
        "source_row_identifier": (
            "zero-based row index in the pinned source split before deduplication, "
            "written as an unpadded base-10 integer"
        ),
    }
    for key, expected in expected_values.items():
        if manifest[key] != expected:
            raise ValueError(
                f"Evaluation manifest {key} mismatch: expected {expected!r}, got {manifest[key]!r}."
            )

    seen = set()
    keep_indices = []
    for source_row_index, raw_prompt in enumerate(eval_dataset["raw_prompt"]):
        if raw_prompt not in seen:
            seen.add(raw_prompt)
            keep_indices.append(source_row_index)
    selected_source_indices = keep_indices[: manifest["expected_selected_rows"]]
    if len(selected_source_indices) != manifest["expected_selected_rows"]:
        raise ValueError("The pinned dataset does not contain enough unique validation prompts.")
    selected = eval_dataset.select(selected_source_indices)
    raw_prompts = selected["raw_prompt"]
    if len(set(raw_prompts)) != manifest["expected_unique_raw_prompts"]:
        raise ValueError("Evaluation manifest unique-prompt count mismatch.")
    if ordered_sha256(raw_prompts) != manifest["ordered_raw_prompt_sha256"]:
        raise ValueError("Evaluation manifest raw-prompt digest mismatch.")
    if ordered_sha256(selected_source_indices) != manifest["ordered_source_row_identifier_sha256"]:
        raise ValueError("Evaluation manifest source-row digest mismatch.")
    return selected, selected_source_indices


def input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, NotImplementedError):
        return next(model.parameters()).device


def main() -> None:
    args = parse_args()
    config, profile = load_configuration(args)
    generation = config["generation"]
    dataset = config["dataset"]
    if generation["max_input_length"] + generation["max_new_tokens"] != generation["max_length"]:
        raise ValueError("Generation token limits must sum to max_length.")

    set_seeds(int(generation["seed"]))
    print_local_main("loading merged SFT model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(sft_model, args.adapter_path) if args.adapter_path else sft_model
    model.eval()
    device = input_device(model)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    evaluation_view = dataset["evaluation_view"]
    if evaluation_view not in DATASET_CONFIGS:
        raise ValueError(f"Unknown evaluation dataset view: {evaluation_view}")
    from datasets import disable_caching

    disable_caching()
    rdp = DATASET_CONFIGS[evaluation_view](
        revision=dataset["revision"],
        prompt_template=config["prompt_template"],
        num_proc=int(config["training"]["num_proc"]),
    )
    eval_dataset = rdp.get_sft_dataset(split="validation")
    eval_dataset, source_row_indices = select_and_verify_prompts(config, eval_dataset)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "00001-of-00001.jsonl"
    print_local_main(
        f"decode_profile={args.decode_profile} repetition_penalty={profile['repetition_penalty']} "
        f"no_repeat_ngram_size={profile['no_repeat_ngram_size']}"
    )

    results = []
    for offset in tqdm.tqdm(range(0, len(eval_dataset), args.batch_size)):
        batch = eval_dataset[offset : offset + args.batch_size]
        encoded = tokenizer(
            batch["prompt"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(generation["max_input_length"]),
        )
        padded_input_length = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                max_new_tokens=int(generation["max_new_tokens"]),
                do_sample=False,
                repetition_penalty=float(profile["repetition_penalty"]),
                no_repeat_ngram_size=int(profile["no_repeat_ngram_size"]),
            )
        full_samples = tokenizer.batch_decode(generated, skip_special_tokens=True)
        completions = tokenizer.batch_decode(generated[:, padded_input_length:], skip_special_tokens=True)
        for batch_index, full_sample in enumerate(full_samples):
            source_row_index = source_row_indices[offset + batch_index]
            results.append(
                {
                    "source_row_index": source_row_index,
                    "pair_id": f"validation:{source_row_index}",
                    "raw_prompt": batch["raw_prompt"][batch_index],
                    "prompt": batch["prompt"][batch_index],
                    "response": completions[batch_index],
                    "prompt_response": full_sample,
                }
            )

    Dataset.from_list(results).to_json(str(output_path))
    print(f"Saved {len(results)} generations to {output_path}")


if __name__ == "__main__":
    main()
