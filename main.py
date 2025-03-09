import argparse

from utils.dataset import get_wikitext2
from utils.eval import evaluate_ppl
from utils.model import ModelType, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture activations of a model")

    parser.add_argument(
        "--model",
        type=ModelType,
        default=ModelType.LLAMA_2_7B,
        help="Model name (default: meta-llama/Llama-2-7b-hf)",
    )
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
        help="Number of samples (default: 512)",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size (default: 1)",
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

    return parser.parse_args()


def main():
    args = parse_args()

    model, tokenizer = load_model(
        args.model, return_tokenizer=True, hf_token=args.hf_token
    )

    input_ids = get_wikitext2(
        tokenizer,
        args.seqlen,
        args.num_samples,
        args.batch_size,
        True,
        args.seed,
        "cpu",
    )

    ppl = evaluate_ppl(model, input_ids, args.device)
    print(f"Perplexity: {ppl}")


if __name__ == "__main__":
    main()
