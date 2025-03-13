import argparse

from .model import ModelType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture activations of a model")

    # model arguments
    parser.add_argument(
        "--model",
        type=ModelType,
        default=ModelType.LLAMA_2_7B,
        help="Model name (default: LLAMA_2_7B)",
    )

    # evaluation arguments
    parser.add_argument(
        "--seqlen",
        default=2048,
        type=int,
        help="Sequence length (default: 2048)",
    )
    parser.add_argument(
        "--num_samples",
        default=512,
        type=int,
        help="Number of samples (default: 512), use <=0 for full dataset",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size (default: 1)",
    )

    # quantization arguments
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize model",
    )
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

    # smoothing arguments
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Smooth model",
    )
    parser.add_argument(
        "--alpha",
        default=0.5,
        type=float,
        help="Smoothing factor (default: 0.5)",
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

    # wandb arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to wandb",
    )
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project")

    # other arguments
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

    return parser.parse_args()
