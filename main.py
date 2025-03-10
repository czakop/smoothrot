from utils.args import parse_args
from utils.dataset import get_wikitext2
from utils.eval import evaluate_ppl
from utils.model import load_model


def main():
    args = parse_args()

    if args.wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=args)

    model, tokenizer = load_model(
        args.model, return_tokenizer=True, hf_token=args.hf_token, device_map="cpu"
    )

    if args.quantize:
        from utils.quant import QuantConfig, quantize_model

        w_quant_config = QuantConfig(
            bits=args.w_bits, asym=args.w_asym, per_tensor=args.w_per_tensor
        )
        a_quant_config = QuantConfig(
            bits=args.a_bits, asym=args.a_asym, per_tensor=args.a_per_tensor
        )
        quantize_model(model, w_quant_config, a_quant_config, args.device)

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
    if args.wandb:
        wandb.log({"ppl": ppl})


if __name__ == "__main__":
    main()
