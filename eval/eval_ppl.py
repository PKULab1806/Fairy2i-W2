import math
import argparse
import torch
import random
import os
import json
import logging
import datasets

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# cache_dir = "/root/working/fairy2i/codes/datasets"
# os.environ["HF_DATASETS_OFFLINE"] = "1" 
# datasets.config.HF_DATASETS_CACHE = cache_dir


import numpy as np
from eval_utils import get_test_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_module.qat_modules import replace_modules_for_qat, convert_to_inference_mode
from tqdm import tqdm

torch.set_grad_enabled(False)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--hf_path", type=str)
parser.add_argument("--output_dir", type=str, help="directory to save the results")
parser.add_argument("--seqlen", default=2048, type=int)
parser.add_argument(
    "--device", default="cuda:0", type=str, help="device to run the model on"
)
parser.add_argument(
    "--cuda", default=None, type=int, help="Specify GPU ID to use (e.g., --cuda 1 means using cuda:1, has higher priority than --device)"
)
parser.add_argument(
    "--convert_model",
    action="store_true",
    help="Whether to convert model to inference-optimized mode (default False, i.e., no conversion)"
)

# 新增的命令行参数
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


def calulate_loss(model, input, loss_fct):
    output = model(
        input, use_cache=False, output_hidden_states=False, output_attentions=False
    )[0]
    shift_logits = output[:, :-1, :].contiguous()
    shift_labels = input[:, 1:].clone()
    shift_labels[input[:, :-1] == model.config.eos_token_id] = -100
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def main(args):
    if args.cuda is not None:
        device = f"cuda:{args.cuda}"
        logging.info(f"Using GPU specified by --cuda parameter: {device}")
    else:
        device = args.device
        logging.info(f"Using device specified by --device parameter: {device}")
    datasets = ["wikitext2", "c4"]
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_path,
        device_map=device,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)
    
    logging.info("loaded model!")

    if args.replace_method:
        logging.info(f"Replacing model's linear layers with '{args.replace_method}' QAT modules...")
        logging.info(f"skip_lm_head={args.skip_lm_head}")
        replace_modules_for_qat(model, args.replace_method, skip_lm_head=args.skip_lm_head)
        logging.info("Layer replacement completed.")
        

        logging.info("Optimizing model for faster inference...")
        convert_to_inference_mode(model)
        logging.info("Inference optimization completed. Quantized weights will be cached, significantly improving inference performance.")
        model.to(device)

    model = model.eval()
    print(model.dtype)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum").to(device)

    ppl = []
    results = {}
    for dataset in datasets:
        testdata = get_test_dataset(dataset, tokenizer, seqlen=args.seqlen)
        acc_loss, count = 0.0, 0
        progress = tqdm(range(len(testdata)))
        for ii in progress:
            input = torch.Tensor(testdata[ii]).long().view(1, -1)
            input = input.to(device)
            loss = calulate_loss(model, input, loss_fct)
            count += input.size(-1) - 1
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/ count / math.log(2)}")

        avg_loss = acc_loss / count / math.log(2)
        current_ppl = 2**avg_loss
        ppl.append(current_ppl)
        results[dataset] = current_ppl
        print(f"{dataset} PPL: {current_ppl}")

    avg_ppl = sum(ppl) / len(ppl)
    results["avg_ppl"] = avg_ppl
    print(results)
    print(f"Avg PPL: {avg_ppl}")

  
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_name = args.hf_path.split("/")[-1]
        output_path = os.path.join(args.output_dir, f"{model_name}_ppl_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)