from dataclasses import dataclass, field
from typing import Optional
import os
import math
import hashlib
import json
import struct

import torch
import tyro
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import PeftModel

from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import (
    print_local_main, disable_progress_bar_non_local_main, set_seeds
)

disable_progress_bar_non_local_main()


DIGEST_CANONICALIZATION = (
    "utf8(value) prefixed by unsigned 8-byte big-endian byte length, concatenated in order"
)
PROMPT_TEMPLATES = {
    "beavertails_conversation_v1": "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:",
}


def _ordered_sha256(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _load_and_verify_eval_manifest(script_args, eval_dataset: Dataset):
    with open(script_args.eval_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError("The evaluation manifest must be a JSON object.")

    required = {
        "manifest_schema_version",
        "dataset_name",
        "dataset_revision",
        "source_split",
        "source_split_rule",
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

    expected_values = {
        "manifest_schema_version": 1,
        "dataset_name": script_args.dataset_name,
        "dataset_revision": script_args.dataset_revision,
        "source_split": "validation",
        "source_split_rule": "train_test_split(test_size=0.1, seed=0)['test']",
        "expected_source_split_rows": len(eval_dataset),
        "selection_rule": f"first_{script_args.eval_size}_rows",
        "expected_selected_rows": script_args.eval_size,
        "hash_algorithm": "sha256",
        "digest_canonicalization": DIGEST_CANONICALIZATION,
        "source_row_identifier": (
            "zero-based row index in the derived validation split before selection, "
            "written as an unpadded base-10 integer"
        ),
    }
    for key, actual in expected_values.items():
        if manifest[key] != actual:
            raise ValueError(
                f"Evaluation manifest {key} mismatch: expected {actual!r}, got {manifest[key]!r}."
            )

    prompt_template = PROMPT_TEMPLATES.get(manifest["prompt_template_id"])
    if prompt_template is None:
        raise ValueError(f"Unknown prompt_template_id: {manifest['prompt_template_id']!r}.")
    if script_args.prompt_template != prompt_template:
        raise ValueError(f"Prompt template does not match {manifest['prompt_template_id']!r}.")
    if script_args.eval_size > len(eval_dataset):
        raise ValueError(
            f"Requested {script_args.eval_size} rows from a validation split with {len(eval_dataset)} rows."
        )

    selected = eval_dataset.select(range(script_args.eval_size))
    raw_prompts = selected["raw_prompt"]
    source_row_indices = list(range(script_args.eval_size))
    unique_raw_prompts = len(set(raw_prompts))
    if unique_raw_prompts != manifest["expected_unique_raw_prompts"]:
        raise ValueError(
            "Evaluation manifest unique-prompt count mismatch: "
            f"expected {manifest['expected_unique_raw_prompts']}, got {unique_raw_prompts}."
        )
    prompt_digest = _ordered_sha256(raw_prompts)
    if prompt_digest != manifest["ordered_raw_prompt_sha256"]:
        raise ValueError(
            "Evaluation manifest raw-prompt digest mismatch: "
            f"expected {manifest['ordered_raw_prompt_sha256']}, got {prompt_digest}."
        )
    source_digest = _ordered_sha256(source_row_indices)
    if source_digest != manifest["ordered_source_row_identifier_sha256"]:
        raise ValueError(
            "Evaluation manifest source-row digest mismatch: "
            f"expected {manifest['ordered_source_row_identifier_sha256']}, got {source_digest}."
        )
    return selected, source_row_indices


def _input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, NotImplementedError):
        return next(model.parameters()).device


@dataclass
class ScriptArguments:

    sft_model_name: str = field(metadata={"help": "the sft model name"})
    sft_model_revision: Optional[str] = field(default=None, metadata={"help": "the sft model revision"})
    adapter_model_name: str = field(default=None, metadata={"help": "lora name"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})
    dataset_name: Optional[str] = field(default="PKU-Alignment/PKU-SafeRLHF-10K-safer", metadata={"help": "the dataset name"})
    dataset_revision: Optional[str] = field(default=None, metadata={"help": "the dataset revision"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "used cached dataset"})
    dataset_num_proc: Optional[int] = field(default=1, metadata={"help": "number of dataset map processes"})

    eval_manifest: Optional[str] = field(default=None, metadata={"help": "fixed evaluation manifest"})
    output_dir: Optional[str] = field(default=None, metadata={"help": "output path for generations"})
    eval_size: Optional[int] = field(default=700, metadata={"help": "number of prompts for generations"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    batch_size: Optional[int] = field(default=8)
    rank: Optional[int] = field(default=0)
    world_size: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=0)


if __name__ == "__main__":

    script_args = tyro.cli(ScriptArguments)
    set_seeds(script_args.seed)

    if script_args.sft_model_revision is None:
        raise ValueError("--sft_model_revision is required.")
    if script_args.dataset_revision is None:
        raise ValueError("--dataset_revision is required.")
    if script_args.eval_manifest is None:
        raise ValueError("--eval_manifest is required.")
    if script_args.output_dir is None:
        raise ValueError("--output_dir is required.")
    if script_args.eval_size is None or script_args.eval_size <= 0:
        raise ValueError("--eval_size must be positive.")
    if script_args.batch_size is None or script_args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if script_args.dataset_num_proc is None or script_args.dataset_num_proc <= 0:
        raise ValueError("--dataset_num_proc must be positive.")
    if script_args.world_size is None or script_args.world_size <= 0:
        raise ValueError("--world_size must be positive.")
    if script_args.rank is None or not 0 <= script_args.rank < script_args.world_size:
        raise ValueError("--rank must be in [0, world_size).")

    # base model
    print_local_main("loading model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name,
        revision=script_args.sft_model_revision,
        use_flash_attention_2=script_args.use_flash_attention_2, # flash attn
        torch_dtype=torch.bfloat16, # necessary for llama2, otherwise will be cast to float32
        device_map="auto",
    )
    if script_args.adapter_model_name:
        model = PeftModel.from_pretrained(sft_model, script_args.adapter_model_name)
    else:
        model = sft_model # sft
    model.eval()
    input_device = _input_device(model)

    # tokenizer: left padding for generation
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.sft_model_name,
        revision=script_args.sft_model_revision,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # dataset
    if not script_args.dataset_caching:
        from datasets import disable_caching
        disable_caching()
    rdp = DATASET_CONFIGS[script_args.dataset_name](
        prompt_template=script_args.prompt_template,
        num_proc=script_args.dataset_num_proc,
        revision=script_args.dataset_revision,
    )
    eval_dataset = rdp.get_sft_dataset(split="validation")
    eval_dataset, source_row_indices = _load_and_verify_eval_manifest(script_args, eval_dataset)

    split_size = math.ceil(len(eval_dataset) /script_args.world_size)
    shard_start = script_args.rank * split_size
    shard_end = min((script_args.rank + 1) * split_size, len(eval_dataset))
    eval_dataset = eval_dataset.select(range(shard_start, shard_end))
    source_row_indices = source_row_indices[shard_start:shard_end]
    os.makedirs(script_args.output_dir, exist_ok=True)
    output_path = os.path.join(
        script_args.output_dir,
        f"{str(script_args.rank+1).zfill(5)}-of-{str(script_args.world_size).zfill(5)}.jsonl"
    )

    results = []
    for idx in tqdm.tqdm(range(0, len(eval_dataset), script_args.batch_size)):
        batch = eval_dataset[idx: idx+script_args.batch_size]
        prompt_tokenized = tokenizer(
            batch["prompt"],
            return_tensors="pt",
            padding=True,
        )
        output_tokenized = model.generate(
            input_ids=prompt_tokenized["input_ids"].to(input_device),
            attention_mask=prompt_tokenized["attention_mask"].to(input_device),
            max_length=script_args.max_length,
        )
        output = tokenizer.batch_decode(output_tokenized, skip_special_tokens=True)
        for batch_index, sample in enumerate(output):
            source_row_index = source_row_indices[idx + batch_index]
            prompt = batch["prompt"][batch_index]
            if not sample.startswith(prompt):
                raise ValueError(
                    f"Decoded generation does not start with the formatted prompt at source row {source_row_index}."
                )
            results.append({
                "source_row_index": source_row_index,
                "pair_id": f"validation:{source_row_index}",
                "raw_prompt": batch["raw_prompt"][batch_index],
                "prompt": prompt,
                "response": sample[len(prompt):],
                "prompt_response": sample,
            })


    dataset = Dataset.from_list(results)
    dataset.to_json(output_path)
