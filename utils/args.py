import argparse

from .dataset import DatasetType
from .model import ModelType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture activations of a model")

    # general arguments
    parser.add_argument(
        "--model",
        type=ModelType,
        default=ModelType.LLAMA_2_7B,
        choices=[t.value for t in list(ModelType)],
        help=f"Model name (default: {ModelType.LLAMA_2_7B.value})",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--seqlen",
        default=2048,
        type=int,
        help="Sequence length (default: 2048)",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        type=str,
        help="Hugging Face token",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed (default: 0)",
    )

    # evaluation arguments
    parser.add_argument(
        "--eval_dataset",
        nargs="+",
        type=DatasetType,
        default=[DatasetType.WIKITEXT2],
        choices=[
            DatasetType.WIKITEXT2.value,
            DatasetType.C4_NEW.value,
            DatasetType.PTB_NEW.value,
        ],
        help=f"Dataset to evaluate on (default: {DatasetType.WIKITEXT2.value})",
    )
    parser.add_argument(
        "--eval_samples",
        default=-1,
        type=int,
        help="Number of samples for PPL evaluation (default: -1)",
    )
    parser.add_argument(
        "--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks."
    )
    parser.add_argument(
        "--lm_eval_tasks",
        nargs="+",
        default=[
            "piqa",
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "lambada",
        ],
        help="Tasks to evaluate on (default: piqa hellaswag arc_easy arc_challenge winogrande lambada)",
    )

    # quantization arguments
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize model",
    )
    ## weights
    parser.add_argument(
        "--w_bits",
        default=8,
        type=int,
        help="Weight bits (default: 8)",
    )
    parser.add_argument(
        "--w_asym",
        action="store_true",
        help="Use asymmetric quantization for weights",
    )
    parser.add_argument(
        "--w_per_tensor",
        action="store_true",
        help="Use per-tensor quantization for weights",
    )
    parser.add_argument(
        "--w_group_size",
        default=-1,
        type=int,
        help="Group size for per-channel quantization (default: -1)",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="Use GPTQ quantization",
    )
    parser.add_argument(
        "--gptq_calib_samples",
        default=128,
        type=int,
        help="Number of calibration samples for GPTQ (default: 128)",
    )
    parser.add_argument(
        "--gptq_percdamp",
        default=0.01,
        type=float,
        help="GPTQ per-channel damping factor (default: 0.01)",
    )
    parser.add_argument(
        "--gptq_act_order",
        action="store_true",
        help="GPTQ activation order",
    )
    ## activations
    parser.add_argument(
        "--a_bits",
        default=8,
        type=int,
        help="Activation bits (default: 8)",
    )
    parser.add_argument(
        "--a_asym",
        action="store_true",
        help="Use asymmetric quantization for activations",
    )
    parser.add_argument(
        "--a_per_tensor",
        action="store_true",
        help="Use per-tensor quantization for activations",
    )
    parser.add_argument(
        "--a_group_size",
        default=-1,
        type=int,
        help="Group size for per-channel quantization (default: -1)",
    )
    parser.add_argument(
        "--a_clip_ratio",
        default=1.0,
        type=float,
        help="Clipping ratio for quantization (default: 1.0)",
    )
    ## keys
    parser.add_argument(
        "--k_bits",
        default=16,
        type=int,
        help="Key bits (default: 16)",
    )
    parser.add_argument(
        "--k_asym", action="store_true", help="Use asymmetric quantization for keys"
    )
    parser.add_argument(
        "--k_per_head",
        action="store_true",
        help="Use per-head quantization for keys",
    )
    parser.add_argument(
        "--k_clip_ratio",
        default=1.0,
        type=float,
        help="Clipping ratio for quantization (default: 1.0)",
    )
    parser.add_argument(
        "--k_rotate",
        action="store_true",
        help="Rotate keys",
    )
    ## values
    parser.add_argument(
        "--v_bits",
        default=16,
        type=int,
        help="Value bits (default: 16)",
    )
    parser.add_argument(
        "--v_asym",
        action="store_true",
        help="Use asymmetric quantization for values",
    )
    parser.add_argument(
        "--v_per_head",
        action="store_true",
        help="Use per-head quantization for values",
    )
    parser.add_argument(
        "--v_clip_ratio",
        default=1.0,
        type=float,
        help="Clipping ratio for quantization (default: 1.0)",
    )

    # smoothing arguments
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Smooth model",
    )
    parser.add_argument(
        "--smooth_alpha",
        default=0.5,
        type=float,
        help="Smoothing factor (default: 0.5)",
    )
    parser.add_argument(
        "--smooth_calib_dataset",
        default=DatasetType.WIKITEXT2,
        type=DatasetType,
        choices=[t.value for t in list(DatasetType)],
        help=f"Dataset for calibration (default: {DatasetType.WIKITEXT2.value})",
    )
    parser.add_argument(
        "--smooth_calib_seqlen",
        default=512,
        type=int,
        help="Sequence length for calibration (default: 512)",
    )
    parser.add_argument(
        "--smooth_calib_samples",
        default=512,
        type=int,
        help="Number of calibration samples for smoothing (default: 512)",
    )

    # rotation arguments
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Rotate model",
    )
    parser.add_argument(
        "--spinquant",
        action="store_true",
        help="Use SpinQuant rotation. If False, use QuaRot.",
    )
    parser.add_argument(
        "--optimized_rotation_path",
        default=None,
        type=str,
        help="Path to SpinQuant's optimized rotation matrices",
    )

    # wandb arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to wandb",
    )
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project")

    return parser.parse_args()
