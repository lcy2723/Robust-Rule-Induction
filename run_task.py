import os
import json
from tasks.arithmetic import Arithmetic
from tasks.list_function import ListFunction
from tasks.crypto import Crypto
from prompts import arithmetic_prompt, list_function_prompt, crypto_prompt
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="arithmetic")
    parser.add_argument("--model", type=str, default="deepseek")
    parser.add_argument("--data_path", type=str, default="datasets")
    parser.add_argument("--config_path", type=str, default="config/api_key.json")
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=10)
    parser.add_argument("--ood_ratio", type=float, default=0.0)
    parser.add_argument("--noise_ratio", type=float, default=0.0)
    parser.add_argument("--mode", type=str, default="base")
    parser.add_argument("--task_type", type=str, default="")
    parser.add_argument("--position", type=int, default=None)
    parser.add_argument("--do_infer", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_batch", type=int, default=None)
    parser.add_argument("--do_parallel", action="store_true")
    parser.add_argument("--eval_round", type=int, default=None)
    return parser.parse_args()


def main(args):
    prompt_lf = {
        "base": list_function_prompt.base_prompt, 
        "noisy": list_function_prompt.base_prompt_with_noise,
        "iterative": list_function_prompt.iterative_prompt_with_noise,
        "cot": list_function_prompt.cot_prompt_with_noise,
        "sr_base": list_function_prompt.self_refine_base,
        "sr_feedback": list_function_prompt.self_refine_feedback,
        "sr_iterative": list_function_prompt.self_refine_iteration
    }
    prompt_ar = {
        "base": arithmetic_prompt.base_prompt, 
        "noisy": arithmetic_prompt.base_prompt_with_noise,
        "iterative": arithmetic_prompt.iterative_prompt_with_noise,
        "cot": arithmetic_prompt.cot_prompt_with_noise,
        "sr_base": arithmetic_prompt.self_refine_base,
        "sr_feedback": arithmetic_prompt.self_refine_feedback,
        "sr_iterative": arithmetic_prompt.self_refine_iteration
    }
    prompt_cr = {
        "base": crypto_prompt.base_prompt, 
        "noisy": crypto_prompt.base_prompt_with_noise,
        "iterative": crypto_prompt.iterative_prompt_with_noise,
        "cot": crypto_prompt.cot_prompt_with_noise,
        "sr_base": crypto_prompt.self_refine_base,
        "sr_feedback": crypto_prompt.self_refine_feedback,
        "sr_iterative": crypto_prompt.self_refine_iteration
    }

    if args.task == "arithmetic":
        task = Arithmetic(
            args.data_path,
            args.n_train,
            args.n_test,
            args.ood_ratio,
            args.noise_ratio,
            args.mode,
            prompt=prompt_ar
        )
    elif args.task == "list_function":
        task = ListFunction(
            args.data_path,
            args.n_train,
            args.n_test,
            args.ood_ratio,
            args.noise_ratio,
            args.mode,
            prompt=prompt_lf
        )
    elif args.task == "crypto":
        task = Crypto(
            args.data_path,
            args.n_train,
            args.n_test,
            args.ood_ratio,
            args.noise_ratio,
            args.mode,
            prompt=prompt_cr
        )
    else:
        raise ValueError("Unknown task")
    output_file = f"{args.model}_{args.task}_response_ood{args.ood_ratio}_noise{args.noise_ratio}_{args.mode}_{args.task_type}.jsonl"
    if args.position is not None:
        output_file = f"{args.model}_{args.task}_response_ood{args.ood_ratio}_noise{args.noise_ratio}_{args.mode}_{args.task_type}_pos{args.position}.jsonl"
    if args.do_batch is not None:
        task.convert_to_batch(
            os.path.join(args.output_path, output_file),
            args.model,
            args.position,
            args.do_batch
        )
    output_path = os.path.join(args.output_path, output_file)
    if args.do_infer:
        if args.mode == "srr":
            task.run_srr(
                args.config_path,
                output_path,
                args.model,
                args.position
            )
        elif args.do_parallel:
            task.get_response_parallel(
                args.config_path,
                output_path,
                args.model,
                args.position
            )
        else:
            task.get_response(
                args.config_path,
                output_path,
                args.model,
                args.position
            )
    if args.do_eval:
        if args.eval_round is not None:
            task.eval_err(output_path, args.eval_round)
        else:
            item_acc, task_acc = task.evaluate(output_path)
            eval_path = os.path.join(os.path.dirname(output_path), "results.jsonl")
            with open(eval_path, "a+") as f:
                    f.write(json.dumps({
                        "task": args.task,
                        "model": args.model,
                        "task_type": args.task_type,
                        "ood_ratio": args.ood_ratio,
                        "noise_ratio": args.noise_ratio,
                        "mode": args.mode,
                        "item_acc": item_acc,
                        "task_acc": task_acc,
                        "position": args.position if args.position is not None else "None"
            }) + "\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)
