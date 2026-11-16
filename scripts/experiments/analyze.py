"""Compute the numerical checks for the objective trade-off and inference-time control studies."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from src.data.configs import DATASET_CONFIGS

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
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} is not numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} is not numeric.") from error
    if not math.isfinite(result):
        raise ValueError(f"{context} is not finite.")
    return result


def mean(values) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot compute a mean over no values.")
    return float(array.mean())


def pearson(left, right) -> float:
    left_array = np.asarray(list(left), dtype=float)
    right_array = np.asarray(list(right), dtype=float)
    if left_array.size != right_array.size or left_array.size < 2:
        raise ValueError("Pearson correlation requires aligned vectors with at least two values.")
    if left_array.std() == 0 or right_array.std() == 0:
        raise ValueError("Pearson correlation is undefined for a constant vector.")
    return float(np.corrcoef(left_array, right_array)[0, 1])


def has_repeated_4gram(text: str) -> bool:
    words = text.split()
    seen = set()
    for index in range(len(words) - 3):
        ngram = tuple(words[index : index + 4])
        if ngram in seen:
            return True
        seen.add(ngram)
    return False


def resolve_score_path(scores_root: Path, relative_path: str) -> Path:
    root = scores_root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Score path escapes --scores-root: {relative_path}") from error
    return path


def load_scored_rows(
    relative_path: str,
    scores_root: Path,
    expected_rows: int,
    expected_unique_prompts: int,
    score_keys: list[str],
    cache: dict[str, list[dict]],
    input_files: dict[str, dict],
) -> list[dict]:
    path = resolve_score_path(scores_root, relative_path)
    if relative_path not in cache:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{relative_path}:{line_number} is not a JSON object.")
                missing = sorted(GENERATION_FIELDS - set(row))
                if missing:
                    raise ValueError(f"{relative_path}:{line_number} is missing fields: {', '.join(missing)}")
                source_row_index = row["source_row_index"]
                if isinstance(source_row_index, bool) or not isinstance(source_row_index, int) or source_row_index < 0:
                    raise ValueError(f"{relative_path}:{line_number} has an invalid source_row_index.")
                if row["pair_id"] != f"validation:{source_row_index}":
                    raise ValueError(f"{relative_path}:{line_number} has an inconsistent pair_id.")
                for field in ("raw_prompt", "prompt", "response", "prompt_response"):
                    if not isinstance(row[field], str):
                        raise ValueError(f"{relative_path}:{line_number} field {field!r} is not text.")
                rows.append(row)
        if len(rows) != expected_rows:
            raise ValueError(f"{relative_path} has {len(rows)} rows; expected {expected_rows}.")
        pair_ids = [row["pair_id"] for row in rows]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError(f"{relative_path} contains duplicate pair_id values.")
        unique_prompts = len({row["raw_prompt"] for row in rows})
        if unique_prompts != expected_unique_prompts:
            raise ValueError(
                f"{relative_path} has {unique_prompts} unique prompts; expected {expected_unique_prompts}."
            )
        cache[relative_path] = rows
        input_files[relative_path] = {
            "sha256": file_sha256(path),
            "rows": len(rows),
            "unique_raw_prompts": unique_prompts,
        }
    rows = cache[relative_path]
    for score_key in score_keys:
        for row_index, row in enumerate(rows):
            if score_key not in row:
                raise ValueError(f"{relative_path} row {row_index} is missing score {score_key!r}.")
            number(row[score_key], f"{relative_path} row {row_index} score {score_key}")
    return rows


def score_summary(rows: list[dict], score_keys: list[str]) -> dict:
    return {
        "n_rows": len(rows),
        "mean_response_words": mean(len(row["response"].split()) for row in rows),
        "share_with_repeated_4gram": mean(float(has_repeated_4gram(row["response"])) for row in rows),
        "mean_scores": {key: mean(number(row[key], key) for row in rows) for key in score_keys},
    }


def bootstrap_mean_difference(
    differences: list[float], rng: np.random.Generator, bootstrap: dict, direction: float
) -> dict:
    values = np.asarray(differences, dtype=float)
    indices = rng.integers(0, values.size, size=(bootstrap["resamples"], values.size))
    sampled_means = values[indices].mean(axis=1)
    quantiles = bootstrap["interval_quantiles"]
    raw_low, raw_high = np.quantile(sampled_means, quantiles, method=bootstrap["quantile_method"])
    preferred_means = direction * sampled_means
    preferred_low, preferred_high = np.quantile(
        preferred_means, quantiles, method=bootstrap["quantile_method"]
    )
    return {
        "mean_raw_difference": float(values.mean()),
        "raw_difference_interval": [float(raw_low), float(raw_high)],
        "mean_preferred_direction_difference": float(direction * values.mean()),
        "preferred_direction_interval": [float(preferred_low), float(preferred_high)],
    }


def bootstrap_clustered_difference(
    clusters: dict[str, list[float]], rng: np.random.Generator, bootstrap: dict
) -> dict:
    ordered_prompts = sorted(clusters, key=lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    arrays = [np.asarray(clusters[prompt], dtype=float) for prompt in ordered_prompts]
    values = np.concatenate(arrays)
    cluster_sums = np.asarray([array.sum() for array in arrays], dtype=float)
    cluster_sizes = np.asarray([array.size for array in arrays], dtype=int)
    sampled_means = np.empty(bootstrap["resamples"], dtype=float)
    for start in range(0, bootstrap["resamples"], 1000):
        stop = min(start + 1000, bootstrap["resamples"])
        indices = rng.integers(0, len(arrays), size=(stop - start, len(arrays)))
        sampled_means[start:stop] = cluster_sums[indices].sum(axis=1) / cluster_sizes[indices].sum(axis=1)
    low, high = np.quantile(
        sampled_means,
        bootstrap["interval_quantiles"],
        method=bootstrap["quantile_method"],
    )
    return {
        "mean_raw_difference": float(values.mean()),
        "raw_difference_interval": [float(low), float(high)],
    }


def preference_choice(left, right) -> int:
    if left is None or right is None:
        return -1
    if left < right:
        return 1
    if left > right:
        return 0
    return -1


def helpsteer_raw_pairs(dataset, objectives: set[str]):
    batch_size = 1000
    for batch_start in range(0, len(dataset), batch_size):
        batch = dataset[batch_start : batch_start + batch_size]
        prompts = batch["prompt"]
        group_start = 0
        for index in range(len(prompts) + 1):
            if index < len(prompts) and (index == 0 or prompts[index] == prompts[index - 1]):
                continue
            for left_index, right_index in combinations(range(group_start, index), 2):
                scores = {
                    objective: (
                        number(batch[objective][left_index], f"HelpSteer {objective}"),
                        number(batch[objective][right_index], f"HelpSteer {objective}"),
                    )
                    for objective in objectives
                }
                yield (
                    prompts[left_index],
                    batch["response"][left_index],
                    batch["response"][right_index],
                    scores,
                )
            group_start = index


def ultrafeedback_score(completion: dict, objective: str):
    annotation = completion.get("annotations") or {}
    value = (annotation.get(objective) or {}).get("Rating")
    try:
        return number(value, f"UltraFeedback {objective}")
    except ValueError:
        return None


def ultrafeedback_raw_pairs(dataset, objectives: set[str]):
    for row in dataset:
        completions = row["completions"]
        for left_index, right_index in combinations(range(len(completions)), 2):
            left = completions[left_index]
            right = completions[right_index]
            scores = {
                objective: (
                    ultrafeedback_score(left, objective),
                    ultrafeedback_score(right, objective),
                )
                for objective in objectives
            }
            yield row["instruction"], left["response"], right["response"], scores


def condition_pair_rule(condition: dict, objective_config: dict) -> dict:
    dataset = objective_config["dataset"]
    common = {
        "revision": dataset["revision"],
        "prompt_template": objective_config["prompt_template"],
        "num_proc": int(objective_config["training"]["num_proc"]),
    }
    anchor = DATASET_CONFIGS[dataset["modpo_view"]](**common)
    margin = DATASET_CONFIGS[condition["margin_dataset_view"]](**common)
    anchor_dimension = getattr(margin, "anchor_dimension", None) or getattr(anchor, "dimension", None)
    if anchor_dimension != "helpfulness":
        raise ValueError(f"Expected helpfulness anchor, got {anchor_dimension!r}.")
    if getattr(margin, "anchor_dimension", None):
        anchor_gap = float(getattr(margin, "min_anchor_gap", 0.0))
    else:
        anchor_gap = float(getattr(anchor, "min_gap", 0.0))
    return {
        "target_dimension": getattr(margin, "dimension", None),
        "target_gap": float(getattr(margin, "min_gap", 0.0)),
        "anchor_gap": anchor_gap,
        "require_agreement": bool(getattr(margin, "require_agreement_with_anchor", False)),
        "require_disagreement": bool(getattr(margin, "require_disagreement_with_anchor", False)),
    }


def pair_is_selected(rule: dict, scores: dict) -> tuple[bool, bool]:
    target_left, target_right = scores[rule["target_dimension"]]
    anchor_left, anchor_right = scores["helpfulness"]
    target_choice = preference_choice(target_left, target_right)
    anchor_choice = preference_choice(anchor_left, anchor_right)
    target_gap = abs(target_left - target_right) if target_choice != -1 else 0.0
    anchor_gap = abs(anchor_left - anchor_right) if anchor_choice != -1 else 0.0
    if (
        target_choice == -1
        or anchor_choice == -1
        or target_gap < rule["target_gap"]
        or anchor_gap < rule["anchor_gap"]
    ):
        return False, False
    conflict = target_choice != anchor_choice
    if rule["require_agreement"] and conflict:
        return False, conflict
    if rule["require_disagreement"] and not conflict:
        return False, conflict
    return True, conflict


def raw_dataset_correlations(study: str, objective_config: dict, study_config: dict) -> list[dict]:
    dataset = objective_config["dataset"]
    objectives = study_config["target_score_keys"]
    values = {objective: ([], []) for objective in objectives}
    if study == "helpsteer":
        rows = []
        for split in ("train", "validation"):
            rows.extend(load_dataset(dataset["name"], revision=dataset["revision"], split=split))
        for row in rows:
            for objective in objectives:
                left = number(row["helpfulness"], "HelpSteer helpfulness")
                right = number(row[objective], f"HelpSteer {objective}")
                values[objective][0].append(left)
                values[objective][1].append(right)
    else:
        rows = load_dataset(dataset["name"], revision=dataset["revision"], split="train")
        for row in rows:
            for completion in row["completions"]:
                annotations = completion.get("annotations") or {}
                helpfulness = (annotations.get("helpfulness") or {}).get("Rating")
                for objective in objectives:
                    target = (annotations.get(objective) or {}).get("Rating")
                    try:
                        left = number(helpfulness, "UltraFeedback helpfulness")
                        right = number(target, f"UltraFeedback {objective}")
                    except ValueError:
                        continue
                    values[objective][0].append(left)
                    values[objective][1].append(right)

    directions = {}
    for condition in objective_config["conditions"].values():
        if condition["pair_selection"] == "all":
            directions.setdefault(condition["second_objective"], set()).add(
                condition["second_objective_score_direction"]
            )
    output = []
    for objective, (helpfulness, target) in values.items():
        expected = study_config["dataset_correlation_expected_rows"][objective]
        if len(helpfulness) != expected:
            raise ValueError(f"{study} {objective} has {len(helpfulness)} correlation rows; expected {expected}.")
        raw = pearson(helpfulness, target)
        for direction in sorted(directions[objective]):
            multiplier = -1.0 if direction == "lower" else 1.0
            output.append(
                {
                    "study": study,
                    "second_objective": objective,
                    "second_objective_score_direction": direction,
                    "n_responses": len(helpfulness),
                    "raw_pearson": raw,
                    "preferred_direction_pearson": multiplier * raw,
                }
            )
    return output


def pair_relationships(study: str, objective_config: dict) -> list[dict]:
    dataset = objective_config["dataset"]
    training = objective_config["training"]
    tokenizer = AutoTokenizer.from_pretrained(
        objective_config["base_model"]["name"],
        revision=objective_config["base_model"]["revision"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    conditions = objective_config["conditions"]
    rules = {
        name: condition_pair_rule(condition, objective_config)
        for name, condition in conditions.items()
    }
    objectives = {"helpfulness"} | {rule["target_dimension"] for rule in rules.values()}
    if None in objectives:
        raise ValueError(f"A {study} pair rule has no target dimension.")
    output = []
    token_lengths = {}
    for split in ("train", "validation"):
        raw = load_dataset(
            dataset["name"],
            revision=dataset["revision"],
            split="train" if study == "ultrafeedback" else split,
        )
        if study == "ultrafeedback":
            raw = raw.train_test_split(test_size=0.1, seed=int(training["split_seed"]))[
                "train" if split == "train" else "test"
            ]
            pairs = ultrafeedback_raw_pairs(raw, objectives)
        else:
            pairs = helpsteer_raw_pairs(raw, objectives)
        counts = {
            name: {"valid": 0, "retained": 0, "conflicts": 0}
            for name in conditions
        }
        for raw_prompt, left_response, right_response, scores in pairs:
            prompt = objective_config["prompt_template"].format(raw_prompt=raw_prompt)
            lengths = []
            for response in (left_response, right_response):
                text = prompt + response
                if text not in token_lengths:
                    token_lengths[text] = len(tokenizer(text)["input_ids"]) + 1
                lengths.append(token_lengths[text])
            for condition_name, condition in conditions.items():
                selected, conflict = pair_is_selected(rules[condition_name], scores)
                if not selected:
                    continue
                counts[condition_name]["valid"] += 1
                if max(lengths) <= int(condition["max_length"]):
                    counts[condition_name]["retained"] += 1
                    counts[condition_name]["conflicts"] += int(conflict)
        for condition_name, condition in conditions.items():
            valid = counts[condition_name]["valid"]
            retained = counts[condition_name]["retained"]
            raw_conflicts = counts[condition_name]["conflicts"]
            if valid == 0 or retained == 0:
                raise ValueError(f"No retained pairs for {condition_name}, {split}.")
            lower = condition["second_objective_score_direction"] == "lower"
            preferred_conflicts = retained - raw_conflicts if lower else raw_conflicts
            output.append(
                {
                    "study": study,
                    "condition": condition_name,
                    "split": split,
                    "pair_selection": condition["pair_selection"],
                    "max_length": condition["max_length"],
                    "valid_pairs_before_length_filter": valid,
                    "retained_pairs": retained,
                    "retained_share": retained / valid,
                    "raw_direction_non_conflict_pairs": retained - raw_conflicts,
                    "raw_direction_conflict_pairs": raw_conflicts,
                    "raw_direction_conflict_rate": raw_conflicts / retained,
                    "preferred_direction_non_conflict_pairs": retained - preferred_conflicts,
                    "preferred_direction_conflict_pairs": preferred_conflicts,
                    "preferred_direction_conflict_rate": preferred_conflicts / retained,
                }
            )
    return output


def sft_correlations(
    study: str,
    objective_config: dict,
    study_config: dict,
    scores_root: Path,
    cache: dict,
    input_files: dict,
) -> list[dict]:
    output = []
    seen = set()
    for condition_name, condition in objective_config["conditions"].items():
        if condition["pair_selection"] != "all":
            continue
        key = (
            condition["second_objective"],
            condition["second_objective_score_direction"],
            condition["max_length"],
        )
        if key in seen:
            continue
        seen.add(key)
        helpfulness_key = study_config["helpfulness_score_key"]
        target_key = study_config["target_score_keys"][condition["second_objective"]]
        relative_path = study_config["sft_score_paths"]["primary"][str(condition["max_length"])]
        rows = load_scored_rows(
            relative_path,
            scores_root,
            study_config["expected_rows"],
            study_config["expected_unique_raw_prompts"],
            [helpfulness_key, target_key],
            cache,
            input_files,
        )
        raw = pearson(
            (number(row[helpfulness_key], helpfulness_key) for row in rows),
            (number(row[target_key], target_key) for row in rows),
        )
        multiplier = -1.0 if condition["second_objective_score_direction"] == "lower" else 1.0
        output.append(
            {
                "study": study,
                "condition": condition_name,
                "n_responses": len(rows),
                "raw_pearson": raw,
                "preferred_direction_pearson": multiplier * raw,
                "score_file": relative_path,
            }
        )
    return output


def objective_scores(
    analysis_config: dict,
    objective_configs: dict[str, dict],
    scores_root: Path,
    cache: dict,
    input_files: dict,
) -> tuple[list[dict], list[dict]]:
    summaries = []
    comparisons = []
    bootstrap = analysis_config["bootstrap"]
    for study, study_config in analysis_config["objective_studies"].items():
        objective_config = objective_configs[study]
        robustness = set(objective_config["generation"]["robustness_conditions"])
        for condition_name, condition in objective_config["conditions"].items():
            profiles = ["primary"] + (["robustness_2048"] if condition_name in robustness else [])
            helpfulness_key = study_config["helpfulness_score_key"]
            target_key = study_config["target_score_keys"][condition["second_objective"]]
            score_keys = [helpfulness_key, target_key]
            for profile in profiles:
                sft_path = study_config["sft_score_paths"][profile][str(condition["max_length"])]
                sft_rows = load_scored_rows(
                    sft_path,
                    scores_root,
                    study_config["expected_rows"],
                    study_config["expected_unique_raw_prompts"],
                    score_keys,
                    cache,
                    input_files,
                )
                summaries.append(
                    {
                        "study": study,
                        "condition": condition_name,
                        "decode_profile": profile,
                        "model": "sft",
                        "score_file": sft_path,
                        **score_summary(sft_rows, score_keys),
                    }
                )
                for weight in condition["weights"]:
                    weight_text = f"{float(weight):.1f}"
                    model_path = study_config["score_path_pattern"].format(
                        profile=profile,
                        condition=condition_name,
                        weight=weight_text,
                    )
                    model_rows = load_scored_rows(
                        model_path,
                        scores_root,
                        study_config["expected_rows"],
                        study_config["expected_unique_raw_prompts"],
                        score_keys,
                        cache,
                        input_files,
                    )
                    summaries.append(
                        {
                            "study": study,
                            "condition": condition_name,
                            "decode_profile": profile,
                            "model": f"w{weight_text}",
                            "score_file": model_path,
                            **score_summary(model_rows, score_keys),
                        }
                    )

    configured = {
        f"{study}:{condition_name}"
        for study, objective_config in objective_configs.items()
        for condition_name in objective_config["conditions"]
    }
    comparison_order = analysis_config.get("objective_comparison_order")
    if not isinstance(comparison_order, list) or any(not isinstance(item, str) for item in comparison_order):
        raise ValueError("objective_comparison_order must be a list of study:condition strings.")
    if len(comparison_order) != len(set(comparison_order)) or set(comparison_order) != configured:
        raise ValueError("objective_comparison_order must contain every configured objective condition exactly once.")

    rng = np.random.default_rng(bootstrap["seed"])
    for comparison_key in comparison_order:
        study, condition_name = comparison_key.split(":", 1)
        study_config = analysis_config["objective_studies"][study]
        condition = objective_configs[study]["conditions"][condition_name]
        helpfulness_key = study_config["helpfulness_score_key"]
        target_key = study_config["target_score_keys"][condition["second_objective"]]
        score_keys = [helpfulness_key, target_key]
        sft_path = study_config["sft_score_paths"]["primary"][str(condition["max_length"])]
        sft_rows = load_scored_rows(
            sft_path,
            scores_root,
            study_config["expected_rows"],
            study_config["expected_unique_raw_prompts"],
            score_keys,
            cache,
            input_files,
        )
        sft_by_prompt = {row["raw_prompt"]: row for row in sft_rows}
        if len(sft_by_prompt) != len(sft_rows):
            raise ValueError(f"SFT rows are not unique by raw_prompt for {condition_name}, primary.")
        for weight in condition["weights"]:
            weight_text = f"{float(weight):.1f}"
            model_path = study_config["score_path_pattern"].format(
                profile="primary",
                condition=condition_name,
                weight=weight_text,
            )
            model_rows = load_scored_rows(
                model_path,
                scores_root,
                study_config["expected_rows"],
                study_config["expected_unique_raw_prompts"],
                score_keys,
                cache,
                input_files,
            )
            model_by_prompt = {row["raw_prompt"]: row for row in model_rows}
            if len(model_by_prompt) != len(model_rows) or set(model_by_prompt) != set(sft_by_prompt):
                raise ValueError(f"Prompt pairing failed for {condition_name}, primary, w{weight_text}.")
            prompts = sorted(sft_by_prompt)
            for metric, score_key in (("helpfulness", helpfulness_key), ("second_objective", target_key)):
                differences = []
                for prompt in prompts:
                    model_row = model_by_prompt[prompt]
                    sft_row = sft_by_prompt[prompt]
                    if model_row["pair_id"] != sft_row["pair_id"]:
                        raise ValueError(f"Pair ID mismatch for {condition_name}, prompt {prompt!r}.")
                    differences.append(number(model_row[score_key], score_key) - number(sft_row[score_key], score_key))
                direction = (
                    -1.0
                    if metric == "second_objective"
                    and condition["second_objective_score_direction"] == "lower"
                    else 1.0
                )
                comparisons.append(
                    {
                        "study": study,
                        "condition": condition_name,
                        "decode_profile": "primary",
                        "model": f"w{weight_text}",
                        "reference_model": "sft",
                        "comparison_order": "model_minus_sft",
                        "metric": metric,
                        "score_key": score_key,
                        "score_direction": (
                            condition["second_objective_score_direction"]
                            if metric == "second_objective"
                            else "higher"
                        ),
                        "n_paired_rows": len(prompts),
                        "n_clusters": len(prompts),
                        **bootstrap_mean_difference(differences, rng, bootstrap, direction),
                    }
                )
    return summaries, comparisons


def pku_scores(
    analysis_config: dict,
    scores_root: Path,
    cache: dict,
    input_files: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    config = analysis_config["pku"]
    score_keys = config["score_keys"]
    bootstrap = analysis_config["bootstrap"]
    rng = np.random.default_rng(bootstrap["seed"])
    roles_by_path = defaultdict(list)
    for target in config["targets"]:
        for role in ("direct", "fine_nearest", "coarse_nearest", "interpolated", "lower_neighbor", "upper_neighbor"):
            if role in target:
                roles_by_path[target[role]["path"]].append(f"target_{target['target_weight']:.1f}:{role}")
    summaries = []
    for path, roles in roles_by_path.items():
        rows = load_scored_rows(
            path,
            scores_root,
            config["expected_rows"],
            config["expected_unique_raw_prompts"],
            score_keys,
            cache,
            input_files,
        )
        summaries.append({"score_file": path, "roles": roles, **score_summary(rows, score_keys)})

    comparisons = []
    between_neighbors = []
    for target in config["targets"]:
        direct = load_scored_rows(
            target["direct"]["path"],
            scores_root,
            config["expected_rows"],
            config["expected_unique_raw_prompts"],
            score_keys,
            cache,
            input_files,
        )
        direct_by_id = {row["pair_id"]: row for row in direct}
        for method in ("fine_nearest", "coarse_nearest", "interpolated"):
            if method not in target:
                continue
            method_rows = load_scored_rows(
                target[method]["path"],
                scores_root,
                config["expected_rows"],
                config["expected_unique_raw_prompts"],
                score_keys,
                cache,
                input_files,
            )
            method_by_id = {row["pair_id"]: row for row in method_rows}
            if set(method_by_id) != set(direct_by_id):
                raise ValueError(f"PKU pair alignment failed for target {target['target_weight']}, {method}.")
            pair_ids = sorted(direct_by_id, key=lambda pair_id: direct_by_id[pair_id]["source_row_index"])
            for score_key in score_keys:
                clusters = defaultdict(list)
                for pair_id in pair_ids:
                    direct_row = direct_by_id[pair_id]
                    method_row = method_by_id[pair_id]
                    if method_row["raw_prompt"] != direct_row["raw_prompt"]:
                        raise ValueError(f"PKU prompt mismatch for target {target['target_weight']}, {method}.")
                    clusters[direct_row["raw_prompt"]].append(
                        number(method_row[score_key], score_key) - number(direct_row[score_key], score_key)
                    )
                if len(clusters) != config["expected_unique_raw_prompts"]:
                    raise ValueError(f"PKU cluster count mismatch for target {target['target_weight']}, {method}.")
                statistics = bootstrap_clustered_difference(clusters, rng, bootstrap)
                same_model = target[method]["path"] == target["direct"]["path"]
                comparisons.append(
                    {
                        "target_weight": target["target_weight"],
                        "method": method,
                        "method_definition": target[method],
                        "direct_definition": target["direct"],
                        "comparison_order": "method_minus_direct",
                        "score_key": score_key,
                        "n_paired_rows": len(pair_ids),
                        "n_clusters": len(clusters),
                        "same_model": same_model,
                        "include_in_aggregate": not same_model,
                        **statistics,
                        "absolute_mean_difference": abs(statistics["mean_raw_difference"]),
                    }
                )

        interpolated = load_scored_rows(
            target["interpolated"]["path"],
            scores_root,
            config["expected_rows"],
            config["expected_unique_raw_prompts"],
            score_keys,
            cache,
            input_files,
        )
        lower = load_scored_rows(
            target["lower_neighbor"]["path"],
            scores_root,
            config["expected_rows"],
            config["expected_unique_raw_prompts"],
            score_keys,
            cache,
            input_files,
        )
        upper = load_scored_rows(
            target["upper_neighbor"]["path"],
            scores_root,
            config["expected_rows"],
            config["expected_unique_raw_prompts"],
            score_keys,
            cache,
            input_files,
        )
        id_sets = [{row["pair_id"] for row in rows} for rows in (interpolated, lower, upper, direct)]
        if any(ids != id_sets[0] for ids in id_sets[1:]):
            raise ValueError(f"PKU neighbor alignment failed for target {target['target_weight']}.")
        for score_key in score_keys:
            interpolation_mean = mean(number(row[score_key], score_key) for row in interpolated)
            lower_mean = mean(number(row[score_key], score_key) for row in lower)
            upper_mean = mean(number(row[score_key], score_key) for row in upper)
            between_neighbors.append(
                {
                    "target_weight": target["target_weight"],
                    "score_key": score_key,
                    "interpolated_mean": interpolation_mean,
                    "lower_neighbor_mean": lower_mean,
                    "upper_neighbor_mean": upper_mean,
                    "interpolated_mean_between_neighbor_means": (
                        min(lower_mean, upper_mean) <= interpolation_mean <= max(lower_mean, upper_mean)
                    ),
                }
            )
    return summaries, comparisons, between_neighbors


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    if config.get("schema_version") != 1:
        raise ValueError("Analysis configuration must use schema_version 1.")
    if np.__version__ != config["bootstrap"]["numpy_version"]:
        raise RuntimeError(
            f"This analysis requires NumPy {config['bootstrap']['numpy_version']}; found {np.__version__}."
        )

    repo_root = Path(__file__).resolve().parents[2]
    objective_configs = {}
    config_files = {str(args.config): file_sha256(args.config)}
    for study, study_config in config["objective_studies"].items():
        path = repo_root / study_config["config"]
        objective_configs[study] = load_json(path)
        config_files[study_config["config"]] = file_sha256(path)

    cache = {}
    input_files = {}
    relationships = {"dataset_correlations": [], "pair_measurements": [], "sft_correlations": []}
    for study, study_config in config["objective_studies"].items():
        objective_config = objective_configs[study]
        relationships["dataset_correlations"].extend(
            raw_dataset_correlations(study, objective_config, study_config)
        )
        relationships["pair_measurements"].extend(pair_relationships(study, objective_config))
        relationships["sft_correlations"].extend(
            sft_correlations(
                study,
                objective_config,
                study_config,
                args.scores_root,
                cache,
                input_files,
            )
        )

    objective_summaries, objective_comparisons = objective_scores(
        config, objective_configs, args.scores_root, cache, input_files
    )
    pku_summaries, pku_comparisons, between_neighbors = pku_scores(
        config, args.scores_root, cache, input_files
    )
    result = {
        "metadata": {
            "configuration_files": config_files,
            "input_score_files": input_files,
            "bootstrap": config["bootstrap"],
            "score_keys": {
                "objective": {
                    study: {
                        "helpfulness": study_config["helpfulness_score_key"],
                        "second_objectives": study_config["target_score_keys"],
                    }
                    for study, study_config in config["objective_studies"].items()
                },
                "pku": config["pku"]["score_keys"],
            },
            "comparison_rules": {
                "objective": {
                    "condition_order": config["objective_comparison_order"],
                    "paired_by": "raw_prompt",
                    "difference": "model_minus_sft",
                },
                "pku": {
                    "paired_by": "pair_id",
                    "clustered_by": "raw_prompt",
                    "difference": "method_minus_direct",
                    "same_model_include_in_aggregate": False,
                },
            },
            "objective_resources": {
                study: {
                    "base_model": objective_configs[study]["base_model"],
                    "dataset": objective_configs[study]["dataset"],
                    "evaluator": objective_configs[study]["evaluator"],
                }
                for study in objective_configs
            },
            "pku_resources": {
                key: config["pku"][key]
                for key in ("dataset", "base_model", "reward_model", "cost_model")
            },
        },
        "relationships": relationships,
        "scores": {
            "objective_summaries": objective_summaries,
            "objective_model_minus_sft": objective_comparisons,
            "pku_summaries": pku_summaries,
            "pku_method_minus_direct": pku_comparisons,
            "pku_interpolated_between_neighbors": between_neighbors,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
