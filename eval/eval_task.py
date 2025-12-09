import os
import json
import argparse
import torch
import random
import logging

import sys
import datasets
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# cache_dir = "/root/working/fairy2i/codes/datasets"
# os.environ["HF_DATASETS_OFFLINE"] = "1" 
# datasets.config.HF_DATASETS_CACHE = cache_dir


import numpy as np
from lm_eval import evaluator
from eval_utils import LMEvalAdaptor
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_module.qat_modules import replace_modules_for_qat, convert_to_inference_mode
# import os

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--hf_path", default="1bitLLM/bitnet_b1_58-3B", type=str)
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--tasks", type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
parser.add_argument("--ctx_size", default=2048, type=int)
parser.add_argument(
    "--device", default="cuda:0", type=str, help="device to run the model on"
)
parser.add_argument(
    "--cuda", default=None, type=int, help="Specify GPU ID to use (e.g., --cuda 1 means using cuda:1, has higher priority than --device)"
)

parser.add_argument(
    "--replace_method",
    type=str,
    default=None,
    choices=["bitnet", "complex_phase_v1", "complex_phase_v2", "complex_phase_v3", "complex_phase_v4"],
    help="Before evaluation, replace all linear layers with specified QAT modules. Supported methods: bitnet, complex_phase_v1, complex_phase_v2, complex_phase_v3, complex_phase_v4"
)
parser.add_argument(
    "--skip_lm_head",
    action="store_true",
    help="Whether to skip replacement of lm_head layer (default False, i.e., lm_head will be replaced)"
)
parser.add_argument(
    "--convert_model",
    action="store_true",
    help="Whether to convert model to inference-optimized mode (default False, i.e., no conversion)"
)


def main(args):
    model_str = args.hf_path
    if args.cuda is not None:
        device = f"cuda:{args.cuda}"
        logging.info(f"Using GPU specified by --cuda parameter: {device}")
    else:
        device = args.device
        logging.info(f"Using device specified by --device parameter: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_path,
        device_map=device,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.eval()
    print(model.device)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)
    
    if tokenizer.bos_token and not getattr(tokenizer, 'add_bos_token', False):
        print("Tokenizer does not add BOS token by default. Enabling it.")
        if hasattr(tokenizer, 'add_bos_token'):
            tokenizer.add_bos_token = True

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Pad token not set. Using EOS token ({tokenizer.eos_token}) as pad token.")

    tokenizer.padding_side = "left"
    print(f"Setting padding side to '{tokenizer.padding_side}'")


    logging.info("loaded model!")

    # 3. Execute replacement logic after model loading and before evaluation
    if args.replace_method:
        logging.info(f"Replacing model's linear layers with '{args.replace_method}' QAT modules...")
        logging.info(f"skip_lm_head={args.skip_lm_head}")
        replace_modules_for_qat(model, args.replace_method, skip_lm_head=args.skip_lm_head)
        logging.info("Layer replacement completed.")
        logging.info("Optimizing model for faster inference...")
        convert_to_inference_mode(model)
        logging.info("Inference optimization completed. Quantized weights will be cached, significantly improving inference performance.")
        model.to(device)
        
    task_names = args.tasks.split(",")

    lm_eval_model = LMEvalAdaptor(
        model_str, model, tokenizer, args.batch_size, args.ctx_size, device=device
    )
    logging.info("start evaluating")
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=args.batch_size,
        no_cache=True,
        num_fewshot=args.num_fewshot,
    )

    print(evaluator.make_table(results))


    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        # otherwise cannot save
        results["config"]["model"] = args.hf_path
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
