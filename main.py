import torch
import transformers

from utils.args import parse_args
from utils.dataset import get_wikitext2
from utils.eval import evaluate_ppl, evaluate_zero_shot
from utils.model import load_model
from utils.quant import add_linear_wrappers


def main():
    args = parse_args()
    transformers.set_seed(args.seed)

    if args.wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=args)

    model, tokenizer = load_model(
        args.model,
        args.seqlen,
        return_tokenizer=True,
        hf_token=args.hf_token,
        device_map="cpu",
    )

    if args.smooth:
        from utils.smooth import get_act_scales, smooth_model

        calib_data = get_wikitext2(
            tokenizer,
            args.smooth_calib_seqlen,
            args.smooth_calib_samples,
            args.batch_size,
            False,
            args.seed,
            "cpu",
        )
        act_scales = get_act_scales(model, calib_data, args.device)
        smooth_model(model, act_scales, args.smooth_alpha)
        torch.cuda.empty_cache()

    if args.rotate:
        from utils.rotation import (
            fuse_layernorms,
            online_had_hook_factory,
            rotate_model,
        )

        fuse_layernorms(model)
        rotate_model(model, args.spinquant, args.optimized_rotation_path, args.device)

        for layer in model.model.layers:
            add_linear_wrappers(layer)
            layer.mlp.down_proj.register_forward_pre_hook(online_had_hook_factory())
            if not args.spinquant:
                layer.self_attn.o_proj.register_forward_pre_hook(
                    online_had_hook_factory(
                        had_dim=model.config.hidden_size
                        // model.config.num_attention_heads
                    )
                )
        torch.cuda.empty_cache()
    else:
        add_linear_wrappers(model.model.layers)

    if args.quantize:
        from utils.quant import quantize_model

        quantize_model(model, args)
        torch.cuda.empty_cache()

    input_ids = get_wikitext2(
        tokenizer,
        args.seqlen,
        args.ppl_samples,
        args.batch_size,
        True,
        args.seed,
        "cpu",
    )

    ppl = evaluate_ppl(model, input_ids, args.device)
    print(f"Perplexity: {ppl}")
    if args.wandb:
        wandb.log({"ppl": ppl})

    if args.lm_eval:
        zero_shot_results = evaluate_zero_shot(
            model, tokenizer, args.lm_eval_tasks, args.batch_size, args.device
        )
        print(zero_shot_results)
        if args.wandb:
            wandb.log(zero_shot_results)


if __name__ == "__main__":
    main()
