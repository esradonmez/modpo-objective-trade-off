# Which Objectives Need a Dial?

Code for the paper [Which Objectives Need a Dial? Predicting Objective Conflict and Covering Trade-offs in Steerable Pluralistic Alignment](https://aclanthology.org/2026.pandora-1.10/) by David Tsoi and Esra Dönmez, published at the PANDORA 2026 workshop.

The paper presents two studies of Multi-Objective Direct Preference Optimization (MODPO) as a way to build steerable models that balance competing objectives:

- **Objective trade-off study.** Tests whether measurements taken before training predict if MODPO improves two objectives together or trades one off against the other. Covers seven objective pairs from HelpSteer and UltraFeedback, scored with ArmoRM.
- **Inference-time control study.** Compares ways of covering objective weights that were not trained directly: selecting the nearest trained model, interpolating trained adapters, and training at the target weight. Uses the PKU-SafeRLHF helpfulness and harmlessness objectives, scored with the Beaver reward and cost models.

The code extends the [official MODPO repository](https://github.com/ZHZisZZ/modpo) at commit `2e2df20f73b35b09f88ae1fb7d8e19880a6bc3fc`.

## Repository layout

| Path | Contents |
| --- | --- |
| `configs/objective_pairs/` | Objective pairs, training conditions, and decoding settings for HelpSteer and UltraFeedback |
| `configs/eval_manifests/` | Fixed evaluation prompt sets |
| `configs/analysis.json` | Model roles and target objective weights used by the analysis |
| `scripts/experiments/` | Training, generation, ArmoRM scoring, and analysis scripts added for these studies |
| `scripts/examples/`, `scripts/modpo/` | SFT, DPO, and MODPO scripts from the MODPO repository, with additions such as adapter interpolation |
| `src/` | Data loaders, trainers, and utilities |

## Setup

Create the Python 3.10 environment used for training, generation, Beaver scoring, and analysis:

```bash
conda create -n modpo python=3.10
conda activate modpo
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Create a separate environment for ArmoRM scoring:

```bash
conda create -n modpo-armorm python=3.10
conda activate modpo-armorm
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-armorm.txt
```

The objective trade-off study uses [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf), a gated model. Request access, then log in from the training environment:

```bash
conda activate modpo
huggingface-cli login
```

Run all commands below from the repository root. Outputs are written to `outputs/`.

## Objective trade-off study

The settings used for HelpSteer and UltraFeedback are in `configs/objective_pairs/helpsteer.json` and `configs/objective_pairs/ultrafeedback.json`.
The commands below reproduce the HelpSteer helpfulness and correctness objective pair with a 2048 token training length limit and objective weight `w=0.1`.

Train the SFT adapter:

```bash
PYTHONPATH=. accelerate launch scripts/examples/sft/sft.py \
  --base_model_name meta-llama/Llama-2-7b-hf \
  --base_model_revision 01c7f73d771dfac7d292323805ebc428287df4f9 \
  --dataset_name nvidia/HelpSteer-pairwise-helpfulness \
  --dataset_revision 3ca5d59c1bc1080af195b4254e7407db60b6f450 \
  --generate-during-eval False \
  --max_length 2048 \
  --num_proc 1 \
  --training_args.output_dir outputs/reproduction-training/objective/helpsteer/sft_2048 \
  --training_args.seed 303 \
  --training_args.max_steps 1000 \
  --training_args.per_device_train_batch_size 1 \
  --training_args.per_device_eval_batch_size 1 \
  --training_args.gradient_accumulation_steps 8 \
  --training_args.gradient_checkpointing \
  --training_args.bf16 \
  --training_args.no-fp16 \
  --training_args.save_steps 200 \
  --training_args.eval_steps 200 \
  --training_args.save_total_limit 5
```

Merge the SFT adapter into the base model to create the SFT reference model:

```bash
PYTHONPATH=. python src/tools/merge_peft_adapter.py \
  --adapter_model_name outputs/reproduction-training/objective/helpsteer/sft_2048/best_checkpoint \
  --base_model_name meta-llama/Llama-2-7b-hf \
  --base_model_revision 01c7f73d771dfac7d292323805ebc428287df4f9 \
  --dtype bf16 \
  --output_name outputs/reproduction-training/objective/helpsteer/sft_2048/merged_checkpoint
```

Train the correctness margin reward model:

```bash
PYTHONPATH=. accelerate launch scripts/examples/dpo/dpo.py \
  --sft_model_name outputs/reproduction-training/objective/helpsteer/sft_2048/merged_checkpoint \
  --dataset_name nvidia/HelpSteer-pairwise-correctness \
  --dataset_revision 3ca5d59c1bc1080af195b4254e7407db60b6f450 \
  --beta 0.1 \
  --generate-during-eval False \
  --max_length 2048 \
  --num_proc 1 \
  --training_args.output_dir outputs/reproduction-training/objective/helpsteer/correctness_2048/margin \
  --training_args.seed 303 \
  --training_args.max_steps 1000 \
  --training_args.per_device_train_batch_size 1 \
  --training_args.per_device_eval_batch_size 1 \
  --training_args.gradient_accumulation_steps 8 \
  --training_args.gradient_checkpointing \
  --training_args.bf16 \
  --training_args.no-fp16 \
  --training_args.save_steps 200 \
  --training_args.eval_steps 200 \
  --training_args.save_total_limit 5
```

Train the MODPO model:

```bash
PYTHONPATH=. accelerate launch scripts/experiments/objective_pairs/train_modpo.py \
  --config configs/objective_pairs/helpsteer.json \
  --condition correctness_2048 \
  --w 0.1 \
  --sft-model-path outputs/reproduction-training/objective/helpsteer/sft_2048/merged_checkpoint \
  --margin-adapter-path outputs/reproduction-training/objective/helpsteer/correctness_2048/margin/best_checkpoint \
  --output-dir outputs/reproduction-training/objective/helpsteer/correctness_2048/w0.1
```

Generate responses for the 400 unique validation prompts:

```bash
PYTHONPATH=. python scripts/experiments/objective_pairs/generate.py \
  --config configs/objective_pairs/helpsteer.json \
  --condition correctness_2048 \
  --decoding-config configs/objective_pairs/decoding_profiles.json \
  --decode-profile primary \
  --sft-model-path outputs/reproduction-training/objective/helpsteer/sft_2048/merged_checkpoint \
  --adapter-path outputs/reproduction-training/objective/helpsteer/correctness_2048/w0.1/best_checkpoint \
  --output-dir outputs/reproduction/objective/helpsteer/primary/correctness_2048/w0.1/generations
```

Activate the ArmoRM environment:

```bash
conda activate modpo-armorm
```

Score the generated responses:

```bash
PYTHONPATH=. python scripts/experiments/objective_pairs/score_armorm.py \
  --config configs/objective_pairs/helpsteer.json \
  --input-dir outputs/reproduction/objective/helpsteer/primary/correctness_2048/w0.1/generations \
  --output-dir outputs/reproduction/objective/helpsteer/primary/correctness_2048/w0.1
```

To generate responses from the SFT reference model, omit `--adapter-path` and use the corresponding `sft_512` or `sft_2048` output directory.
`configs/objective_pairs/decoding_profiles.json` contains the main decoding setting and the decoding robustness check.

Return to the training environment:

```bash
conda activate modpo
```

## Inference-time control study

The model roles and target objective weights are listed in `configs/analysis.json`.
The commands below train the harmlessness margin reward model and the MODPO model at `w=0.2`, then interpolate the corresponding adapters for target objective weight `0.1`.

Train the harmlessness margin reward model:

```bash
PYTHONPATH=. accelerate launch scripts/examples/dpo/dpo.py \
  --sft_model_name PKU-Alignment/alpaca-7b-reproduced \
  --sft_model_revision f4f1d993c68a98b2f953bb87208992eab221d1dc \
  --dataset_name PKU-Alignment/PKU-SafeRLHF-10K-safer \
  --dataset_revision c8ee87634695ae44c5fa9bcdc5f1a5e7c5ae8e47 \
  --beta 0.1 \
  --max_length 1024 \
  --num_proc 1 \
  --training_args.output_dir outputs/reproduction-training/pku/rm_safer \
  --training_args.seed 42 \
  --training_args.num_train_epochs 3 \
  --training_args.per_device_train_batch_size 1 \
  --training_args.per_device_eval_batch_size 4 \
  --training_args.gradient_accumulation_steps 8 \
  --peft_config.r 64 \
  --peft_config.lora_alpha 1 \
  --peft_config.lora_dropout 0 \
  --peft_config.target_modules q_proj k_proj v_proj o_proj
```

Train the MODPO model at `w=0.2`:

```bash
PYTHONPATH=. accelerate launch scripts/modpo/beavertails/modpo.py \
  --sft_model_name PKU-Alignment/alpaca-7b-reproduced \
  --sft_model_revision f4f1d993c68a98b2f953bb87208992eab221d1dc \
  --margin_reward_model_name outputs/reproduction-training/pku/rm_safer/best_checkpoint \
  --dataset_name PKU-Alignment/PKU-SafeRLHF-10K-better \
  --dataset_revision c8ee87634695ae44c5fa9bcdc5f1a5e7c5ae8e47 \
  --w 0.2 \
  --beta 0.1 \
  --max_length 1024 \
  --num_proc 1 \
  --training_args.output_dir outputs/reproduction-training/pku/fine/w0.2 \
  --training_args.seed 42 \
  --training_args.num_train_epochs 3 \
  --training_args.per_device_train_batch_size 1 \
  --training_args.per_device_eval_batch_size 4 \
  --training_args.gradient_accumulation_steps 8 \
  --peft_config.r 64 \
  --peft_config.lora_alpha 1 \
  --peft_config.lora_dropout 0 \
  --peft_config.target_modules q_proj k_proj v_proj o_proj
```

Use the same command for the other positive objective weights, changing `--w` and the output directory.
At `w=0.0`, use the DPO adapter trained on harmlessness preference pairs.

Generate responses from the model at `w=0.2` for the 700 validation response pairs:

```bash
PYTHONPATH=. python scripts/modpo/beavertails/utils/gen.py \
  --sft_model_name PKU-Alignment/alpaca-7b-reproduced \
  --sft_model_revision f4f1d993c68a98b2f953bb87208992eab221d1dc \
  --adapter_model_name outputs/reproduction-training/pku/fine/w0.2/best_checkpoint \
  --prompt_template 'BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:' \
  --dataset_name PKU-Alignment/PKU-SafeRLHF-10K-safer \
  --dataset_revision c8ee87634695ae44c5fa9bcdc5f1a5e7c5ae8e47 \
  --dataset_num_proc 1 \
  --eval_manifest configs/eval_manifests/pku_validation700.json \
  --eval_size 700 \
  --max_length 1024 \
  --output_dir outputs/reproduction/pku/fine/w0.2/generations
```

Interpolate the adapters at `w=0.0` and `w=0.2` for target objective weight `0.1`:

```bash
PYTHONPATH=. python scripts/modpo/beavertails/utils/gen_interpolated.py \
  --sft_model_name PKU-Alignment/alpaca-7b-reproduced \
  --sft_model_revision f4f1d993c68a98b2f953bb87208992eab221d1dc \
  --adapter_lower outputs/reproduction-training/pku/rm_safer/best_checkpoint \
  --adapter_upper outputs/reproduction-training/pku/fine/w0.2/best_checkpoint \
  --alpha 0.5 \
  --prompt_template 'BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:' \
  --dataset_name PKU-Alignment/PKU-SafeRLHF-10K-safer \
  --dataset_revision c8ee87634695ae44c5fa9bcdc5f1a5e7c5ae8e47 \
  --dataset_num_proc 1 \
  --eval_manifest configs/eval_manifests/pku_validation700.json \
  --eval_size 700 \
  --max_length 1024 \
  --output_dir outputs/reproduction/pku/interpolated/w0.1/generations
```

Score the generated responses with the Beaver reward and cost models:

```bash
PYTHONPATH=. python scripts/modpo/beavertails/utils/score.py \
  --input_dir outputs/reproduction/pku/interpolated/w0.1/generations \
  --output_dir outputs/reproduction/pku/interpolated/w0.1 \
  --reward_model_revision 375cd6a9f0d7e339d2199b05ba129a4a8906596d \
  --cost_model_revision c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
```

Nearest model selection and direct target models use the same generation command with the corresponding adapter path from `configs/analysis.json`.

## Analysis

After generating and scoring all models listed in `configs/analysis.json`, run the analysis:

```bash
PYTHONPATH=. python scripts/experiments/analyze.py \
  --config configs/analysis.json \
  --scores-root outputs/reproduction \
  --output outputs/reproduction/analysis.json
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{tsoi-donmez-2026-objectives,
    title = "Which Objectives Need a Dial? Predicting Objective Conflict and Covering Trade-offs in Steerable Pluralistic Alignment",
    author = {Tsoi, David  and
      D{\"o}nmez, Esra},
    editor = "Xu, Shanshan  and
      T.Y.S.S., Santosh  and
      Kurniawan, Kemal  and
      Peng, Siyao  and
      Wallbridge, Sarenne  and
      Koto, Fajri  and
      Dziri, Nouha  and
      Plank, Barbara",
    booktitle = "Proceedings of the First Workshop on Pluralistic {AI} {NLP}: Diversity-aware, Sociotechnical, Responsible Alignment ({PANDORA} 2026)",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.pandora-1.10/",
    doi = "10.18653/2026.pandora-1.10",
    pages = "191--206",
    ISBN = "979-8-89176-484-2"
}
```

## License

This repository is released under the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/). See [LICENSE](LICENSE) for the full text.

## Acknowledgements

This code builds on the [MODPO repository](https://github.com/ZHZisZZ/modpo) by Zhou et al. Please also cite their paper:

```bibtex
@inproceedings{zhou2024beyond,
  title={Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization},
  author={Zhou, Zhanhui and Liu, Jie and Shao, Jing and Yue, Xiangyu and Yang, Chao and Ouyang, Wanli and Qiao, Yu},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2024},
  pages={10586--10613},
  year={2024}
}
```
