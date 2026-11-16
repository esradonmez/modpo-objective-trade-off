from dataclasses import dataclass, field
from typing import Optional
import os
import math

import torch
import tyro
import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset

from src.utils import (
    disable_progress_bar_non_local_main
)
from scripts.modpo.beavertails.utils.score_model import LlamaForScore

disable_progress_bar_non_local_main()

REWARD_MODEL_NAME = "PKU-Alignment/beaver-7b-v1.0-reward"
COST_MODEL_NAME = "PKU-Alignment/beaver-7b-v1.0-cost"


def _input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, NotImplementedError):
        return next(model.parameters()).device


@dataclass
class ScriptArguments:

    input_dir: Optional[str] = field(default=None, metadata={"help": "output path for generations"})
    output_dir: Optional[str] = field(default=None, metadata={"help": "output path for generations"})
    reward_model_revision: Optional[str] = field(default=None)
    cost_model_revision: Optional[str] = field(default=None)


if __name__ == "__main__":

    script_args = tyro.cli(ScriptArguments)

    if script_args.input_dir is None:
        raise ValueError("--input_dir is required.")
    if script_args.output_dir is None:
        raise ValueError("--output_dir is required.")
    if script_args.reward_model_revision is None:
        raise ValueError("--reward_model_revision is required.")
    if script_args.cost_model_revision is None:
        raise ValueError("--cost_model_revision is required.")

    reward = LlamaForScore.from_pretrained(REWARD_MODEL_NAME, revision=script_args.reward_model_revision, torch_dtype=torch.bfloat16, device_map='auto')
    cost = LlamaForScore.from_pretrained(COST_MODEL_NAME, revision=script_args.cost_model_revision, torch_dtype=torch.bfloat16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME, revision=script_args.reward_model_revision)
    reward.eval()
    cost.eval()
    reward_device = _input_device(reward)
    cost_device = _input_device(cost)

    generation = load_dataset(script_args.input_dir, split="train")
    required_fields = {
        "source_row_index",
        "pair_id",
        "raw_prompt",
        "prompt",
        "response",
        "prompt_response",
    }
    missing = sorted(required_fields - set(generation.column_names))
    if missing:
        raise ValueError(f"Generation rows are missing fields: {', '.join(missing)}")

    results = []
    seen_pair_ids = set()
    with torch.no_grad():
        for row in tqdm.tqdm(generation):
            source_row_index = row["source_row_index"]
            pair_id = row["pair_id"]
            if (
                isinstance(source_row_index, bool)
                or not isinstance(source_row_index, int)
                or source_row_index < 0
            ):
                raise ValueError("source_row_index must be a non-negative integer.")
            if pair_id != f"validation:{source_row_index}":
                raise ValueError(f"pair_id does not match source_row_index: {pair_id!r}.")
            if pair_id in seen_pair_ids:
                raise ValueError(f"Duplicate pair_id: {pair_id!r}.")
            seen_pair_ids.add(pair_id)

            text_fields = ("raw_prompt", "prompt", "response", "prompt_response")
            if any(not isinstance(row[key], str) for key in text_fields):
                raise ValueError(f"Generation text fields must be strings for {pair_id!r}.")
            if row["prompt_response"] != row["prompt"] + row["response"]:
                raise ValueError(f"prompt_response does not equal prompt + response for {pair_id!r}.")

            input = tokenizer(row["prompt_response"], return_tensors="pt")
            reward_output = reward(input["input_ids"].to(reward_device), input["attention_mask"].to(reward_device))
            cost_output = cost(input["input_ids"].to(cost_device), input["attention_mask"].to(cost_device))
            results.append({
                **row,
                "reward": reward_output.end_scores.item(),
                "cost": cost_output.end_scores.item(),
            })

    if not results:
        raise ValueError("No generation rows were found.")
    os.makedirs(script_args.output_dir, exist_ok=True)

    # raw
    dataset = Dataset.from_list(results)
    dataset.to_json(os.path.join(script_args.output_dir, "raw.jsonl"))

    # mean
    rewards = [result["reward"] for result in results]
    costs = [result["cost"] for result in results]
    mean_reward = sum(rewards) / len(rewards)
    mean_cost = sum(costs) / len(costs)
    with open(os.path.join(script_args.output_dir, "mean.csv"), "w") as f:
        f.write("mean reward, mean cost\n")
        f.write(f"{mean_reward}, {mean_cost}\n")
