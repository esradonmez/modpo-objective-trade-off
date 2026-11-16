"""Train one configured HelpSteer or UltraFeedback MODPO model."""

import argparse
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.data.configs import DATASET_CONFIGS
from src.trainer.modpo_trainer import MODPOTrainer
from src.utils import (
    PeftAsPreTrained,
    disable_progress_bar_non_local_main,
    param_sharding_enabled,
    prepare_model_for_peft,
    print_local_main,
    set_seeds,
)
from src.utils.reward import ImplicitRewardWrapper, RewardWrapperList

disable_progress_bar_non_local_main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--w", type=float, required=True)
    parser.add_argument("--sft-model-path", required=True)
    parser.add_argument("--margin-adapter-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_run_config(path: Path, condition_name: str, w: float) -> tuple[dict, dict]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("Objective configuration must use schema_version 1.")
    conditions = config.get("conditions")
    if not isinstance(conditions, dict) or condition_name not in conditions:
        available = ", ".join(sorted(conditions or {}))
        raise ValueError(f"Unknown condition {condition_name!r}. Available conditions: {available}")
    condition = conditions[condition_name]
    weights = condition.get("weights")
    if not isinstance(weights, list) or not any(abs(w - float(value)) < 1e-12 for value in weights):
        raise ValueError(f"w={w} is not configured for condition {condition_name!r}: {weights}")
    if not (0.0 < w <= 1.0):
        raise ValueError("w must be in (0, 1].")
    if config.get("weight_semantics") != {"w": "helpfulness", "one_minus_w": "second_objective"}:
        raise ValueError("The objective configuration has unexpected weight semantics.")
    return config, condition


def main() -> None:
    args = parse_args()
    config, condition = load_run_config(args.config, args.condition, args.w)
    training = config["training"]
    dataset = config["dataset"]
    lora = training["lora"]

    if training.get("precision") != "bf16":
        raise ValueError("Objective training is configured for BF16.")
    if training.get("generate_during_eval") is not False:
        raise ValueError("Objective training must disable generation during trainer evaluation.")
    dataset_view = dataset["modpo_view"]
    if dataset_view not in DATASET_CONFIGS:
        raise ValueError(f"Unknown MODPO dataset view: {dataset_view}")

    set_seeds(int(training["seed"]))
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        seed=int(training["seed"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=training["lr_scheduler_type"],
        warmup_steps=float(training["warmup_steps"]),
        weight_decay=float(training["weight_decay"]),
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        run_name=f"{config['study']}_{args.condition}_w{args.w:g}",
        report_to=[],
        max_steps=int(training["modpo_max_steps"]),
        logging_steps=int(training["logging_steps"]),
        save_strategy="steps",
        save_steps=int(training["save_steps"]),
        eval_steps=int(training["eval_steps"]),
        eval_delay=0.25,
        evaluation_strategy="steps",
        save_total_limit=int(training["save_total_limit"]),
        load_best_model_at_end=bool(training["load_best_model_at_end"]),
    )
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    print_local_main("loading merged SFT model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch.bfloat16,
        **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
    )
    sft_model.config.update({"use_cache": False, "pad_token_id": sft_model.config.eos_token_id})
    model = prepare_model_for_peft(sft_model, peft_config=peft_config, args=training_args)
    model.load_adapter(args.margin_adapter_path, adapter_name="margin_reward")

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    from datasets import disable_caching

    disable_caching()
    rdp = DATASET_CONFIGS[dataset_view](
        revision=dataset["revision"],
        prompt_template=config["prompt_template"],
        num_proc=int(training["num_proc"]),
    )
    train_dataset = rdp.get_preference_dataset(split="train")
    eval_dataset = rdp.get_preference_dataset(split="validation")

    print_local_main(
        f"condition={args.condition} w={args.w:g} margin_beta={condition['margin_beta']} "
        f"margin_dataset_view={condition['margin_dataset_view']}"
    )
    trainer = MODPOTrainer(
        model=model,
        beta=float(training["beta"]),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=int(condition["max_length"]),
        num_proc=int(training["num_proc"]),
        generate_during_eval=False,
    )
    if Accelerator().is_local_main_process:
        trainer.model.print_trainable_parameters()
    trainer.set_wrapped_margin_reward_model_list(
        RewardWrapperList(
            [
                ImplicitRewardWrapper(
                    model=PeftAsPreTrained(trainer.model, "margin_reward"),
                    ref_model=PeftAsPreTrained(trainer.model),
                    tokenizer=tokenizer,
                    beta=float(condition["margin_beta"]),
                    prompt_template=config["prompt_template"],
                )
            ]
        ),
        w=(args.w, 1.0 - args.w),
        prepare=False,
    )
    trainer.train()

    save_name = "best_checkpoint" if training_args.load_best_model_at_end else "final_checkpoint"
    save_path = os.path.join(training_args.output_dir, save_name)
    trainer.model.save_pretrained(save_path)
    trainer.tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    main()
