from dataclasses import dataclass
from functools import partial
from typing import Literal, Dict, Optional

from datasets import load_dataset

from .utils import RawDatasetPreprocessor
from src.utils import print_local_main

UF_FINE_DIMS = ("instruction_following", "honesty", "truthfulness", "helpfulness")
UF_DIMS = UF_FINE_DIMS + ("overall",)


def _to_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_for_dimension(completion: dict, dimension: str) -> Optional[float]:
    if dimension == "overall":
        return _to_float_or_none(completion.get("overall_score"))
    annotations = completion.get("annotations", {})
    annotation = annotations.get(dimension, {}) if isinstance(annotations, dict) else {}
    rating = annotation.get("Rating") if isinstance(annotation, dict) else None
    return _to_float_or_none(rating)


def _chosen_id_and_gap(score_0: Optional[float], score_1: Optional[float]):
    if score_0 is None or score_1 is None:
        return -1, 0.0
    difference = score_0 - score_1
    if difference > 0:
        return 0, difference
    if difference < 0:
        return 1, -difference
    return -1, 0.0


def ultrafeedback_transform_to_sft(batched_sample, prompt_template):
    new_batched_sample = {
        "raw_prompt": [],
        "prompt": [],
        "response": [],
    }
    for instruction, completions in zip(batched_sample["instruction"], batched_sample["completions"]):
        # breakpoint()
        for completion in completions:
            new_batched_sample["raw_prompt"].append(instruction)
            new_batched_sample["prompt"].append(prompt_template.format(raw_prompt=instruction))
            new_batched_sample["response"].append(completion['response'])
    return new_batched_sample


def ultrafeedback_transform_to_preference(batched_sample):
    new_batched_sample = {
        "prompt": [],
        "response_0": [],
        "response_1": [],
        **{f"{dimension}_chosen_id": [] for dimension in UF_DIMS},
        **{f"{dimension}_gap": [] for dimension in UF_DIMS},
    }
    for instruction, completions in zip(batched_sample["instruction"], batched_sample["completions"]):
        n_responses = len(completions)

        for j in range(n_responses):
            for k in range(j+1, n_responses):
                new_batched_sample["prompt"].append(instruction)
                new_batched_sample["response_0"].append(completions[j]['response'])
                new_batched_sample["response_1"].append(completions[k]['response'])
                for dimension in UF_DIMS:
                    chosen_id, gap = _chosen_id_and_gap(
                        _score_for_dimension(completions[j], dimension),
                        _score_for_dimension(completions[k], dimension),
                    )
                    new_batched_sample[f"{dimension}_chosen_id"].append(chosen_id)
                    new_batched_sample[f"{dimension}_gap"].append(gap)

    return new_batched_sample


@dataclass
class UltraFeedbackRDP(RawDatasetPreprocessor):
    path: Optional[str] = "OpenBMB/UltraFeedback"
    dimension: Optional[Literal["overall", "instruction_following", "honesty", "truthfulness", "helpfulness"]] = None
    min_gap: float = 1.0
    anchor_dimension: Optional[
        Literal["overall", "instruction_following", "honesty", "truthfulness", "helpfulness"]
    ] = None
    min_anchor_gap: float = 1.0
    require_disagreement_with_anchor: bool = False
    require_agreement_with_anchor: bool = False

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_dataset(self.path, revision=self.revision, split="train").train_test_split(test_size=0.1, seed=0)["train"]
        elif split == "validation":
            return load_dataset(self.path, revision=self.revision, split="train").train_test_split(test_size=0.1, seed=0)["test"]
        elif split == "test":
            raise NotImplementedError("test split not implemented for UltraFeedbackRDP")
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        chosen_id = example[f"{self.dimension}_chosen_id"]
        return {
            "raw_prompt": example["prompt"],
            "prompt":   self.prompt_template.format(raw_prompt=example["prompt"]),
            "chosen":   example[f"response_{chosen_id}"],
            "rejected": example[f"response_{1-chosen_id}"],
        }

    def get_preference_dataset(self, split):
        if not self.dimension:
            raise ValueError("preference dimension has to be specified")
        if self.min_gap < 0 or self.min_anchor_gap < 0:
            raise ValueError("preference score gaps must be non-negative")
        if self.require_disagreement_with_anchor and self.require_agreement_with_anchor:
            raise ValueError("anchor agreement and disagreement filters cannot both be enabled")
        if (self.require_disagreement_with_anchor or self.require_agreement_with_anchor) and not self.anchor_dimension:
            raise ValueError("anchor agreement and disagreement filters require anchor_dimension")
        dataset = self._get_raw_dataset(split)
        if self.sanity_check:
            dataset = dataset.select(range(min(len(dataset), 100)))
        print_local_main("mapping raw dataset to preference...")
        dataset = dataset.map(
            ultrafeedback_transform_to_preference,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
        )
        print_local_main("filtering preference...")
        dataset = dataset.filter(
            lambda x: x[f"{self.dimension}_chosen_id"] != -1
            and x[f"{self.dimension}_gap"] >= self.min_gap
        )
        if self.anchor_dimension:
            dataset = dataset.filter(
                lambda x: x[f"{self.anchor_dimension}_chosen_id"] != -1
                and x[f"{self.anchor_dimension}_gap"] >= self.min_anchor_gap
            )
            if self.require_disagreement_with_anchor:
                dataset = dataset.filter(
                    lambda x: x[f"{self.dimension}_chosen_id"] != x[f"{self.anchor_dimension}_chosen_id"]
                )
            elif self.require_agreement_with_anchor:
                dataset = dataset.filter(
                    lambda x: x[f"{self.dimension}_chosen_id"] == x[f"{self.anchor_dimension}_chosen_id"]
                )
        print_local_main("mapping dataset to standard format...")
        return dataset.map(self._dataset_to_preference_formatter, num_proc=self.num_proc, remove_columns=dataset.column_names)

    def get_sft_dataset(self, split, **kwargs):
        if self.dimension:
            return super().get_sft_dataset(split, **kwargs)
        dataset = self._get_raw_dataset(split)
        if self.sanity_check:
            dataset = dataset.select(range(min(len(dataset), 100)))
        print_local_main("mapping raw dataset to sft...")
        return dataset.map(
            partial(ultrafeedback_transform_to_sft, prompt_template=self.prompt_template),
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
        )


if __name__ == "__main__":
    num_proc = 4
    overall_dataset = UltraFeedbackRDP(dimension="overall", num_proc=num_proc).get_preference_dataset(split="train")
    sft_dataset     = UltraFeedbackRDP(num_proc=num_proc).get_sft_dataset(split="train")
    breakpoint()
