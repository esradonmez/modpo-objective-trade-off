"""Generate responses after linear interpolation of two LoRA adapters.

For every adapter tensor, theta = (1 - alpha) * theta_lower + alpha * theta_upper.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import os

import torch
import tqdm
import tyro
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.modpo.beavertails.utils.gen import _input_device, _load_and_verify_eval_manifest
from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import disable_progress_bar_non_local_main, print_local_main, set_seeds

disable_progress_bar_non_local_main()


def _load_adapter_config(adapter_path: str) -> dict:
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing adapter configuration: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Adapter configuration must be a JSON object: {config_path}")
    return config


def _load_adapter_weights(adapter_path: str) -> dict[str, torch.Tensor]:
    bin_path = os.path.join(adapter_path, "adapter_model.bin")
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.isfile(bin_path):
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
    elif os.path.isfile(safetensors_path):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Install safetensors to load this adapter checkpoint.") from exc
        state = load_file(safetensors_path, device="cpu")
    else:
        raise FileNotFoundError(f"No adapter_model.bin or adapter_model.safetensors in {adapter_path}")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Adapter state must be a non-empty tensor dictionary: {adapter_path}")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"Adapter state contains a non-tensor value: {adapter_path}")
    return state


def _interpolate_adapter_weights(
    lower: dict[str, torch.Tensor],
    upper: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if set(lower) != set(upper):
        different_keys = sorted(set(lower) ^ set(upper))
        raise ValueError(f"Adapter state keys differ: {different_keys[:10]}")

    interpolated = {}
    for key, lower_tensor in lower.items():
        upper_tensor = upper[key]
        if lower_tensor.shape != upper_tensor.shape:
            raise ValueError(
                f"Adapter tensor shape differs for {key}: {tuple(lower_tensor.shape)} != {tuple(upper_tensor.shape)}"
            )
        if lower_tensor.dtype != upper_tensor.dtype:
            raise ValueError(
                f"Adapter tensor dtype differs for {key}: {lower_tensor.dtype} != {upper_tensor.dtype}"
            )
        if not torch.is_floating_point(lower_tensor):
            raise ValueError(f"Adapter tensor is not floating point: {key}")
        interpolated[key] = (1.0 - alpha) * lower_tensor + alpha * upper_tensor
    return interpolated


def _resolve_adapter_param_key(adapter_key: str, model_param_keys: set[str], adapter_name: str):
    if adapter_key in model_param_keys:
        return adapter_key

    for token in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
        suffix = f".{token}.weight"
        if adapter_key.endswith(suffix):
            candidate = adapter_key[:-len(".weight")] + f".{adapter_name}.weight"
            if candidate in model_param_keys:
                return candidate
            break
    return None


@dataclass
class ScriptArguments:
    sft_model_name: str = field(metadata={"help": "the sft base model name"})
    adapter_lower: str = field(metadata={"help": "lower-weight LoRA adapter checkpoint"})
    adapter_upper: str = field(metadata={"help": "upper-weight LoRA adapter checkpoint"})
    sft_model_revision: Optional[str] = field(default=None, metadata={"help": "the sft model revision"})
    alpha: float = field(default=0.5, metadata={"help": "0 selects lower; 1 selects upper"})

    use_flash_attention_2: Optional[bool] = field(default=False)
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE)
    dataset_name: Optional[str] = field(default="PKU-Alignment/PKU-SafeRLHF-10K-safer")
    dataset_revision: Optional[str] = field(default=None, metadata={"help": "the dataset revision"})
    dataset_caching: Optional[bool] = field(default=False)
    dataset_num_proc: Optional[int] = field(default=1)

    eval_manifest: Optional[str] = field(default=None, metadata={"help": "fixed evaluation manifest"})
    output_dir: Optional[str] = field(default=None)
    eval_size: Optional[int] = field(default=700)
    max_length: Optional[int] = field(default=1024)
    batch_size: Optional[int] = field(default=8)
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
    if not 0.0 <= script_args.alpha <= 1.0:
        raise ValueError(f"--alpha must be in [0, 1]. Got {script_args.alpha}.")
    if script_args.eval_size is None or script_args.eval_size <= 0:
        raise ValueError("--eval_size must be positive.")
    if script_args.batch_size is None or script_args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if script_args.dataset_num_proc is None or script_args.dataset_num_proc <= 0:
        raise ValueError("--dataset_num_proc must be positive.")

    lower_config = _load_adapter_config(script_args.adapter_lower)
    upper_config = _load_adapter_config(script_args.adapter_upper)
    if lower_config != upper_config:
        config_keys = sorted(set(lower_config) | set(upper_config))
        different_keys = [key for key in config_keys if lower_config.get(key) != upper_config.get(key)]
        raise ValueError(f"Adapter configurations differ: {different_keys}")

    lower_state = _load_adapter_weights(script_args.adapter_lower)
    upper_state = _load_adapter_weights(script_args.adapter_upper)
    interpolated_state = _interpolate_adapter_weights(lower_state, upper_state, script_args.alpha)

    print_local_main("loading model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name,
        revision=script_args.sft_model_revision,
        use_flash_attention_2=script_args.use_flash_attention_2,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(sft_model, script_args.adapter_lower)

    active_adapter = getattr(model, "active_adapter", "default")
    if isinstance(active_adapter, list):
        if len(active_adapter) != 1:
            raise ValueError(f"Expected one active adapter, got {active_adapter}.")
        active_adapter = active_adapter[0]
    model_parameters = dict(model.named_parameters())
    model_param_keys = set(model_parameters)
    resolved_keys = set()
    for adapter_key, interpolated_tensor in interpolated_state.items():
        model_key = _resolve_adapter_param_key(adapter_key, model_param_keys, active_adapter)
        if model_key is None:
            raise ValueError(f"Could not map adapter tensor to the loaded model: {adapter_key}")
        if model_key in resolved_keys:
            raise ValueError(f"More than one adapter tensor maps to model parameter: {model_key}")
        resolved_keys.add(model_key)
        model_tensor = model_parameters[model_key]
        if model_tensor.shape != interpolated_tensor.shape:
            raise ValueError(
                f"Loaded model tensor shape differs for {model_key}: "
                f"{tuple(model_tensor.shape)} != {tuple(interpolated_tensor.shape)}"
            )
        model_tensor.data.copy_(interpolated_tensor.to(model_tensor.device, model_tensor.dtype))
    if len(resolved_keys) != len(interpolated_state):
        raise ValueError(
            f"Replaced {len(resolved_keys)} of {len(interpolated_state)} adapter tensors."
        )
    model.eval()
    input_device = _input_device(model)

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.sft_model_name,
        revision=script_args.sft_model_revision,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    os.makedirs(script_args.output_dir, exist_ok=True)
    output_path = os.path.join(script_args.output_dir, "00001-of-00001.jsonl")
    results = []
    for idx in tqdm.tqdm(range(0, len(eval_dataset), script_args.batch_size)):
        batch = eval_dataset[idx : idx + script_args.batch_size]
        prompt_tokenized = tokenizer(batch["prompt"], return_tensors="pt", padding=True)
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

    Dataset.from_list(results).to_json(output_path)
