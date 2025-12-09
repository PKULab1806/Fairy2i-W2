import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_PATH = "meta-llama/Llama-2-7b-hf"

import time
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
import math
from transformers.data.data_collator import MyDataCollatorForLanguageModeling
from torch.optim.lr_scheduler import LambdaLR
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForCausalLM
from fairy2i.codes.qat_modules import replace_modules_for_qat, METHOD_MAP
import argparse
from transformers.trainer_utils import total_processes_number
from transformers.trainer_callback import TrainerCallback
import itertools
import torch
from transformers import set_seed
from accelerate import Accelerator
# from swanlab.data.settings import Settings
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

total_steps = 1  # Will be updated dynamically in create_scheduler
SUBSET = [
    "common_crawl",
    "c4",
    "arxiv",
    "github",
    "stackexchange",
    "wikipedia",
    "books3",
]

EOS_TOKEN_ID = tokenizer.eos_token_id
BOS_TOKEN_ID = tokenizer.bos_token_id
PAD_TOKEN_ID = tokenizer.pad_token_id
BLOCK_SIZE = 2048


def group_and_chunk(example_list):

    full_ids = list(
        itertools.chain.from_iterable(
            ids[1:] if ids and ids[0] == BOS_TOKEN_ID else ids
            for ids in example_list["input_ids"]
        )
    )

    result = {"input_ids": [], "attention_mask": [], "labels": []}

    total_len = len(full_ids)
    full_blocks_len = (total_len // BLOCK_SIZE) * BLOCK_SIZE

    for i in range(0, full_blocks_len, BLOCK_SIZE):
        chunk = full_ids[i : i + BLOCK_SIZE]
        labels = chunk.copy()

        for j, token in enumerate(chunk):
            if token in [BOS_TOKEN_ID, EOS_TOKEN_ID]:
                if j + 1 < BLOCK_SIZE:
                    labels[j + 1] = -100

        result["input_ids"].append(chunk)
        result["attention_mask"].append([1] * BLOCK_SIZE)
        result["labels"].append(labels)
    return result

def get_wsd_lr_lambda(
    total,
    stage_boundary_ratio=0.8, 
    stage_scale=1.0,
    min_lr_ratio=0.006667,         
    warmup=500,
):
    stage_boundary = int(total * stage_boundary_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup:
            return stage_scale * (current_step / warmup)

        elif current_step < stage_boundary:
            return stage_scale

        progress = (current_step - stage_boundary) / (total - stage_boundary)

        # cosine_decay = (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)) + min_lr_ratio
        # return stage_scale * cosine_decay

        lr = stage_scale * (1.0 - progress * (1.0 - min_lr_ratio))
        return lr

    return lr_lambda


def get_custom_lr_lambda(
    total,
    stage_boundary_ratio=0.5,
    first_stage_scale=1,
    second_stage_scale=0.666,
    warmup=500,
):
    def lr_lambda(current_step: int):
        progress = current_step / total
        if progress < stage_boundary_ratio:
            if current_step < warmup:
                return first_stage_scale * (current_step / warmup)
            else:
                return first_stage_scale * (1 - (current_step - warmup) / total)
        else:
            return second_stage_scale * (1 - progress)

    return lr_lambda


def get_two_stage_cosine_lr_lambda(
    total,
    stage_boundary_ratio=0.5,
    first_stage_scale=1.0,
    second_stage_scale=0.667,
    min_lr_ratio = 0.006667,
    warmup=500,
):
    stage_boundary = int(total * stage_boundary_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup:
            return (current_step / warmup) * first_stage_scale

        progress = (current_step - warmup) / (total - warmup)
        if current_step < stage_boundary:
            cosine_decay = (1- min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)) + min_lr_ratio
            return first_stage_scale * cosine_decay
        else:
            cosine_decay = (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)) + min_lr_ratio
            return second_stage_scale * cosine_decay

    return lr_lambda


def get_ws_lr_lambda(
    total,
    warmup_steps=30,
    stable_scale=1.0,
):
    """
    WS (Warm-up Stable) learning rate scheduler
    
    Args:
        total: Total training steps
        warmup_steps: Warm-up steps, default 30
        stable_scale: Learning rate scale for stable phase, default 1.0
    
    Returns:
        lr_lambda function for LambdaLR scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return stable_scale * (current_step / warmup_steps)
        else:
            return stable_scale
    
    return lr_lambda


def get_custom_wsd_lr_lambda(
    total,
    warmup_steps=300,
    stable_steps=8000,
    decay_steps=2000,
    final_lr_ratio=0.6667,
    second_decay_start=None,
    second_decay_steps=None,
    second_final_lr_ratio=None,
):
    """
    Custom WSD learning rate scheduler: Warmup → Stable → Decay → Stable → (optional) Decay2 → Final Stable
    
    Args:
        total: Total training steps
        warmup_steps: Warm-up steps
        stable_steps: End step of first stable phase
        decay_steps: Decay phase steps
        final_lr_ratio: Learning rate ratio after first decay
        second_decay_start: Start step of second decay, None to disable
        second_decay_steps: Duration of second decay
        second_final_lr_ratio: Target ratio for second decay
    
    Returns:
        lr_lambda function for LambdaLR scheduler
    """
    decay_start = stable_steps
    decay_end = stable_steps + decay_steps
    
    enable_second_decay = (second_decay_start is not None and 
                          second_decay_steps is not None and 
                          second_final_lr_ratio is not None)
    
    if enable_second_decay:
        second_decay_end = second_decay_start + second_decay_steps
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        elif current_step < decay_start:
            return 1.0
        elif current_step < decay_end:
            progress = (current_step - decay_start) / decay_steps
            cosine_decay = final_lr_ratio + (1.0 - final_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            return cosine_decay
        elif not enable_second_decay or current_step < second_decay_start:
            return final_lr_ratio
        elif current_step < second_decay_end:
            progress = (current_step - second_decay_start) / second_decay_steps
            cosine_decay = second_final_lr_ratio + (final_lr_ratio - second_final_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            return cosine_decay
        else:
            return second_final_lr_ratio
    
    return lr_lambda


parser = argparse.ArgumentParser(description="Train model with specified QAT method")
parser.add_argument(
    "--quant_method",
    type=str,
    default="complex_phase_v1",
    choices=list(METHOD_MAP.keys()),
    help=f"QAT quantization method to use. Available options: {', '.join(list(METHOD_MAP.keys()))}"
)
parser.add_argument(
    "--skip_lm_head",
    action="store_true",
    help="Whether to skip replacement of lm_head layer (default False, i.e., lm_head will be replaced)"
)
# Use parse_known_args() to avoid conflicts with torchrun/deepspeed arguments
args, _ = parser.parse_known_args()


accelerator = Accelerator()
if accelerator.is_main_process:
    print("Accelerator is initialized successfully.")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)

GLOBAL_BATCH_SIZE = 512
PER_DEVICE_BS = 4
MAX_TOKEN = 104_857_600_000*0.3
TRAIN_NAME = "1208_train_30pct_v2_formal_lr3e-5_wsd2_skip_lm_head" 
OUTPUT_DIR = f"./{TRAIN_NAME}/results"
OUTPUT_MODEL_DIR = f"./{TRAIN_NAME}/saved_model"
LOGGING_DIR = f"./{TRAIN_NAME}/logs"
RESUME = True
MAX_STEPS = int(MAX_TOKEN // (GLOBAL_BATCH_SIZE * 2048))

new_vocab_size = len(tokenizer)
print(
    f"New vocab size: {new_vocab_size},eos{EOS_TOKEN_ID},bos{BOS_TOKEN_ID},pad{PAD_TOKEN_ID}"
)
all_parts = []
for name in SUBSET:
    ds = load_from_disk(f"./test_data/final_100B_data_new_withoutlabel/{name}")
    all_parts.append(ds)
train_dataset = concatenate_datasets(all_parts)
num_samples = len(train_dataset)
num_train_samples = int(num_samples * 0.3)
train_dataset = train_dataset.shuffle(seed=42).select(range(num_train_samples))

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

data_collator = MyDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print(f"Loading model from '{MODEL_PATH}'...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map={"": Accelerator().local_process_index},
)
print(f"Applying QAT method: {args.quant_method}")
replace_modules_for_qat(model, args.quant_method, skip_lm_head=args.skip_lm_head)
print("Model loaded.")
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    max_steps=MAX_STEPS,
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BS,
    learning_rate=3e-5,
    max_grad_norm=2.0,
    warmup_steps=375,
    weight_decay=0,
    logging_dir=LOGGING_DIR,
    save_strategy="steps",
    save_steps=1000,
    bf16=True,
    adam_beta1=0.9,
    adam_beta2=0.95,
    report_to="swanlab",
    logging_steps=10,
    save_total_limit=100,
)

world_size = total_processes_number(training_args.local_rank)
assert GLOBAL_BATCH_SIZE % (world_size * PER_DEVICE_BS) == 0, (
    f"Global batch size {GLOBAL_BATCH_SIZE} must be divisible by "
    f"({world_size} * {PER_DEVICE_BS}) = {world_size * PER_DEVICE_BS}"
)
accumulation_steps = GLOBAL_BATCH_SIZE // (world_size * PER_DEVICE_BS)
training_args.gradient_accumulation_steps = accumulation_steps
if accelerator.is_main_process:
    print(
        f"Global batch size: {GLOBAL_BATCH_SIZE}, Per device batch size: {training_args.per_device_train_batch_size}, "
        f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}"
    )
    print(f"Total parameters: {model.num_parameters() / 1e6:.2f} M")
    print(train_dataset)

callbacks = []
if accelerator.is_main_process:
    import swanlab as swanlab
    from swanlab.integration.transformers import SwanLabCallback
    from swanlab import Settings
    cfg = training_args.to_dict()
    cfg["model_type"] = "Llama2-7B-ComplexPhase-b"
    cfg["dataset"] = "RedPajama-Data-v2"
    cfg["max_length"] = 2048
    swanlab.init(
        workspace="ComplexTrain",
        project="complexnet-training-0606",
        name=f"{time.strftime('%m%d%H%M')}-30pct_re-v2-formal_lr3e-5_wsd2_skip_lm_head",
        config=cfg,
        settings=Settings(
            requirements_collect=False,
        ),
    )
    callbacks.append(SwanLabCallback())


class CustomTrainer(Trainer):
    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        global total_steps
        total_steps = num_training_steps
        if optimizer is None:
            optimizer = self.optimizer

        lr_lambda = get_custom_wsd_lr_lambda(
            total=num_training_steps,
            warmup_steps=300,
            stable_steps=8000,
            decay_steps=2000,
            final_lr_ratio=0.1667,
            second_decay_start=19000,
            second_decay_steps=1000,
            second_final_lr_ratio=0.03333,
        )

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        self.lr_scheduler = scheduler
        self._created_lr_scheduler = True
        return self.lr_scheduler


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    data_collator=data_collator,
    callbacks=callbacks,
)

if not os.path.isdir(training_args.output_dir):
    try:
        os.makedirs(training_args.output_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError("Can't create directory: " + training_args.output_dir) from e
last_ckpt = get_last_checkpoint(training_args.output_dir)
try:
    if not RESUME:
        last_ckpt = None
    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(output_dir=OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

finally:
    if accelerator.is_main_process:
        swanlab.finish()
