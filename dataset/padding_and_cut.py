"""
Process 100B dataset into 2048-token aligned blocks to save memory
"""
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# tokenizer.add_eos_token = True
# tokenizer.pad_token = tokenizer.eos_token
from datasets import load_from_disk, Dataset,load_dataset, concatenate_datasets,disable_caching
import os
import shutil
from multiprocessing import Process
# from transformers import AutoTokenizer
from tqdm import tqdm
# disable_caching()
import itertools
OUTDIR="dataset_100B_redpajama_2048_aligned"#for example
BLOCK_SIZE = 2048
# def padding_and_cut(data):
#     result = {
#         "input_ids": [],
#         "attention_mask": [],
#     }
#     for i in range(len(data["input_ids"])):
#         # print(text)
#         tokens = {
#             "input_ids": data["input_ids"][i],
#             "attention_mask": data["attention_mask"][i],
#         }

#         length = len(tokens["input_ids"])
#         for i in range(0, length, BLOCK_SIZE):
#             chunk = {}
#             for k in tokens.keys():
#                 piece = tokens[k][i : i + BLOCK_SIZE]
#                 pad_len = BLOCK_SIZE - len(piece)
#                 if k == "input_ids":
#                     pad_value = tokenizer.pad_token_id
#                 elif k == "attention_mask":
#                     pad_value = 0
#                 elif k == "special_tokens_mask":
#                     pad_value = 1
#                 else:
#                     pad_value = 0
#                 if pad_len > 0:
#                     piece += [pad_value] * pad_len
#                 chunk[k] = piece
           
#             for k in result:
#                 result[k].append(chunk[k])
#     return result
EOS_TOKEN_ID = tokenizer.eos_token_id
BOS_TOKEN_ID = tokenizer.bos_token_id
PAD_TOKEN_ID = tokenizer.pad_token_id
BLOCK_SIZE = 2048
def group_and_chunk(example_list):
  
    full_ids = list(itertools.chain.from_iterable(
        ids[1:] if ids and ids[0] == BOS_TOKEN_ID else ids
        for ids in example_list["input_ids"]
    ))

    result = {
        "input_ids": [],
        "attention_mask": [],
        # "labels": []
    }

    total_len = len(full_ids)
    full_blocks_len = (total_len // BLOCK_SIZE) * BLOCK_SIZE

    # 完整块
    for i in range(0, full_blocks_len, BLOCK_SIZE):
        chunk = full_ids[i : i + BLOCK_SIZE]
        # labels = chunk.copy()

        # for j, token in enumerate(chunk):
        #     if token in [BOS_TOKEN_ID, EOS_TOKEN_ID]:
        #         if j + 1 < BLOCK_SIZE:
        #             labels[j+1] = -100
        
        result["input_ids"].append(chunk)
        result["attention_mask"].append([1] * BLOCK_SIZE)
        # result["labels"].append(labels)
    return result

if __name__ == "__main__":
    sum = 0
    name = "new_dataset_100B_redpajama_dataset"
    print(f"New vocab size:eos{EOS_TOKEN_ID},bos{BOS_TOKEN_ID},pad{PAD_TOKEN_ID}")
    for i in range(10):
        # if name != "common_crawl":
        print(f"tokenize {name}{i} dataset")
        dataset = load_from_disk(f"/root/working/data_new/{name}{i}")#your sampled-dataset directory
        dataset = dataset.shuffle(seed=42)
        print(f"before:{dataset}")
        ds = dataset.map(
            group_and_chunk,
            batched=True,
            num_proc=16,
            remove_columns=dataset.column_names,
        )
        print(f"after:{ds}")
        sum += len(ds)
        ds.save_to_disk(f"{OUTDIR}/{name}{i}",num_proc=5)
        # else:
        #     for i in range(10):
        #         print(f"tokenize {name}_{i} dataset")
        #         dataset = load_from_disk(f"{name}_{i}")
        #         print(f"before:{dataset}")
        #         ds = dataset.map(
        #             tokenize,
        #             batched=True,
        #             num_proc=10,
        #             remove_columns=dataset.column_names,
        #         )
        #         print(f"after:{ds}")
        #         sum += len(ds)
        #         ds.save_to_disk(f"{OUTDIR}/{name}_{i}",num_proc=10)
    print(F"total rows {sum}")