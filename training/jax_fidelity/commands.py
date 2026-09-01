"""Pure command builders for the pinned MaxText release."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .configuration import ExperimentConfig

MAXTEXT_BASE_CONFIG = "src/maxtext/configs/base.yml"
MAXTEXT_SFT_CONFIG = "src/maxtext/configs/post_train/sft.yml"


def _bool(value: bool) -> str:
    return "true" if value else "false"


def build_hf_to_maxtext_command(
    config: ExperimentConfig,
    *,
    hf_checkpoint: Path | str,
    output_directory: Path | str,
) -> list[str]:
    production = config.production
    return [
        "python3",
        "-m",
        "maxtext.checkpoint_conversion.to_maxtext",
        MAXTEXT_BASE_CONFIG,
        f"model_name={production['maxtext_model_name']}",
        f"base_output_directory={Path(output_directory)}",
        f"use_multimodal={_bool(production['use_multimodal'])}",
        f"scan_layers={_bool(production['scan_layers'])}",
        "--lazy_load_tensors=true",
        f"--hf_model_path={Path(hf_checkpoint)}",
        f"--save_dtype={config.conversion['save_dtype']}",
    ]


def build_train_command(
    config: ExperimentConfig,
    *,
    maxtext_checkpoint: Path | str,
    hf_tokenizer_checkpoint: Path | str,
    prepared_train_jsonl: Path | str,
    output_directory: Path | str,
    run_name: str,
    hardware: str,
    smoke: bool,
) -> list[str]:
    training = config.training
    production = config.production
    steps = training["smoke_steps"] if smoke else training["steps"]
    if hardware not in {"gpu", "tpu"}:
        raise ValueError("MaxText training hardware must be gpu or tpu")
    return [
        "python3",
        "-m",
        "maxtext.trainers.post_train.sft.train_sft_native",
        MAXTEXT_SFT_CONFIG,
        f"run_name={run_name}",
        f"base_output_directory={Path(output_directory)}",
        f"model_name={production['maxtext_model_name']}",
        f"load_parameters_path={Path(maxtext_checkpoint)}",
        f"tokenizer_path={Path(hf_tokenizer_checkpoint)}",
        f"hardware={hardware}",
        "dataset_type=hf",
        "hf_path=json",
        f"hf_train_files={Path(prepared_train_jsonl)}",
        "train_split=train",
        "train_data_columns=['messages']",
        f"steps={steps}",
        f"per_device_batch_size={training['per_device_batch_size']}",
        f"max_target_length={training['max_target_length']}",
        f"learning_rate={training['learning_rate']}",
        f"data_shuffle_seed={training['seed']}",
        f"weight_dtype={training['weight_dtype']}",
        f"dtype={training['dtype']}",
        "sft_train_on_completion_only=True",
        f"use_multimodal={_bool(production['use_multimodal'])}",
        f"scan_layers={_bool(production['scan_layers'])}",
        "lora.enable_lora=True",
        f"lora.lora_rank={training['rank']}",
        f"lora.lora_alpha={training['alpha']}",
        "enable_checkpointing=True",
        "checkpoint_period=5",
    ]


def build_maxtext_to_hf_command(
    config: ExperimentConfig,
    *,
    base_checkpoint: Path | str,
    lora_checkpoint: Path | str,
    hf_tokenizer_checkpoint: Path | str,
    output_directory: Path | str,
) -> list[str]:
    production = config.production
    return [
        "python3",
        "-m",
        "maxtext.checkpoint_conversion.to_huggingface",
        MAXTEXT_BASE_CONFIG,
        f"model_name={production['maxtext_model_name']}",
        f"load_parameters_path={Path(base_checkpoint)}",
        f"lora.lora_restore_path={Path(lora_checkpoint)}",
        f"base_output_directory={Path(output_directory)}",
        f"scan_layers={_bool(production['scan_layers'])}",
        f"use_multimodal={_bool(production['use_multimodal'])}",
        f"weight_dtype={config.conversion['save_dtype']}",
        f"--hf_model_path={Path(hf_tokenizer_checkpoint)}",
        "--parallel_threads=2",
    ]


def build_logit_check_command(
    config: ExperimentConfig,
    *,
    maxtext_checkpoint: Path | str,
    hf_checkpoint: Path | str,
) -> list[str]:
    production = config.production
    return [
        "python3",
        "-m",
        "tests.utils.forward_pass_logit_checker",
        MAXTEXT_BASE_CONFIG,
        f"tokenizer_path={Path(hf_checkpoint)}",
        f"load_parameters_path={Path(maxtext_checkpoint)}",
        f"model_name={production['maxtext_model_name']}",
        f"use_multimodal={_bool(production['use_multimodal'])}",
        f"scan_layers={_bool(production['scan_layers'])}",
        "per_device_batch_size=1",
        "dtype=float32",
        "attention=dot_product",
        f"--max_kl_div={config.conversion['max_kl_divergence']}",
        "--run_hf_model=true",
        f"--hf_model_path={Path(hf_checkpoint)}",
    ]


def shell_join(command: Sequence[str]) -> str:
    """Render a human-reviewable command without invoking a shell."""

    import shlex

    return shlex.join(command)
