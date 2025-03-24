from pathlib import Path

import torch
import transformers

from utils.args import parse_args
from utils.dataset import load_dataset
from utils.eval import evaluate_ppl, evaluate_zero_shot
from utils.model import load_model
from utils.quant import add_linear_wrappers


def main():
    args = parse_args()
    if args.wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=args)

    transformers.set_seed(args.seed)

    model, tokenizer = load_model(
        args.model,
        args.seqlen,
        return_tokenizer=True,
        hf_token=args.hf_token,
        device_map="cpu",
    )

    if args.smooth:
        from utils.smooth import get_act_scales, smooth_model

        try:
            artifact_name = (
                args.wandb_act_scale_artifact
                if ":" in args.wandb_act_scale_artifact
                else f"{args.wandb_act_scale_artifact}:latest"
            )
            artifact = wandb.use_artifact(artifact_name)
            act_scale_file = artifact.get_entry(get_act_scale_filename(args)).download(
                ".artifacts/act_scales"
            )
            act_scales = torch.load(act_scale_file)
        except Exception:
            calib_data = load_dataset(
                args.smooth_calib_dataset,
                tokenizer,
                args.smooth_calib_seqlen,
                args.smooth_calib_samples,
                args.batch_size,
                False,
                "cpu",
            )
            act_scales = get_act_scales(model, calib_data, args.device)

            if args.wandb and args.wandb_act_scale_artifact:
                artifact_dir = Path(".artifacts/act_scales")
                act_scale_file = artifact_dir / get_act_scale_filename(args)
                act_scale_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(act_scales, act_scale_file)
                artifact = wandb.Artifact(
                    args.wandb_act_scale_artifact, type="act_scales"
                )
                artifact.add_dir(str(artifact_dir))
                wandb.log_artifact(artifact)

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

    for dataset in args.eval_dataset:
        input_ids = load_dataset(
            dataset,
            tokenizer,
            args.seqlen,
            args.eval_samples,
            args.batch_size,
            True,
            "cpu",
            args.seed,
        )

        if args.wandb and args.wandb_act_artifact:
            from utils.inference import capture_down_proj_input

            save_act_path = Path(".artifacts/activations")
            with capture_down_proj_input(
                model,
                args.save_act_layers,
                save_act_path / args.model.name,
                prefix=get_act_prefix(args) + dataset.name.lower(),
            ):
                ppl = evaluate_ppl(model, input_ids, args.device)

            artifact = wandb.Artifact(args.wandb_act_artifact, type="activations")
            artifact.add_dir(str(save_act_path))
            wandb.log_artifact(artifact)

        else:
            ppl = evaluate_ppl(model, input_ids, args.device)

        print(f"{dataset.name} perplexity: {ppl}")
        if args.wandb:
            wandb.log({f"{dataset.name.lower()} ppl": ppl})

    if args.lm_eval:
        zero_shot_results = evaluate_zero_shot(
            model, tokenizer, args.lm_eval_tasks, args.batch_size, args.device
        )
        print(zero_shot_results)
        if args.wandb:
            wandb.log(zero_shot_results)


def get_act_prefix(args):
    prefix = ""
    if args.smooth:
        prefix += f"smooth_{args.smooth_alpha}_"
    if args.rotate:
        prefix += "rot_"
        prefix += "sq_" if args.spinquant else "qr_"
    return prefix


def get_act_scale_filename(args):
    return f"{args.model.name}/{args.smooth_calib_dataset.name.lower()}_{args.smooth_calib_samples}_{args.smooth_calib_seqlen}_{args.seed}.pt"


if __name__ == "__main__":
    main()
