"""
sample redpajama 1T dataset to 100B tokens
"""
from datasets import load_from_disk, Dataset,load_dataset, concatenate_datasets
import os
os.environ["RED_PAJAMA_DATA_DIR"] = "togethercomputer/RedPajama-Data-1T"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.add_eos_token = True
tokenizer.pad_token = tokenizer.eos_token

import shutil
from multiprocessing import Process
# from transformers import AutoTokenizer
from tqdm import tqdm
import gc
import random
total_tokens = 110_000_000_000
num_processes = 10


save_every_n_tokens = 50_000_000

def sample_worker(proc_id,target_token_count_per_proc,name,dataset:Dataset):
    print(f"proc{proc_id} start sample total{target_token_count_per_proc} tokens from {name},save every {save_every_n_tokens} tokens,with{num_processes} processes")
    buffer = []
    current_token_count = 0
    buffer_token_count = 0
    save_index = 0
    # dataset = dataset.flatten_indices()
    # pbar = tqdm(total=target_token_count_per_proc, position=proc_id, desc=f"Proc {proc_id} from {name}", unit="tok", dynamic_ncols=True)
    for i, example in tqdm(enumerate(dataset),total = len(dataset) ,desc=f"Proc {proc_id} from {name}"):
        # if i % 11 != 0:
        #     continue

        text = example["text"]
        tokens = tokenizer(
            text,
            return_attention_mask=True,
            truncation=False,
            padding=False
        )
        token_len = len(tokens["input_ids"])

        # if current_token_count >= target_token_count_per_proc:
        #     break

        buffer.append({
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        })
        current_token_count += token_len
        buffer_token_count +=token_len

        if buffer_token_count >= save_every_n_tokens:
            ds = Dataset.from_list(buffer)
            save_path = f"{name}_shard_{proc_id}/part_{save_index}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ds.save_to_disk(save_path)
            del ds
            buffer.clear()
            save_index += 1
            buffer_token_count = 0

    # 保存剩余部分
    if buffer:
        ds = Dataset.from_list(buffer)
        save_path = f"{name}_shard_{proc_id}/final"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ds.save_to_disk(save_path)
        del ds
  
    print(f"[Proc {proc_id}] Finished. Total tokens: {current_token_count}")

if __name__ == "__main__":
    
    per_proc_token_count = total_tokens // num_processes
    name = "new_dataset_100B_redpajama"#save dir
    print(f"begin sample {total_tokens/1e6:.2f}M tokens ,with {num_processes} processes,each process will sample {per_proc_token_count/1e6:.2f}M tokens")
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T", split="train")
    print(dataset)
    
    shards = [ dataset.shard(num_shards=num_processes,index = i,contiguous=True) for i in range(num_processes)]
    processes = []
    for pid in range(num_processes):
        p = Process(target=sample_worker, args=(pid,per_proc_token_count,name,shards[pid]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes completed.")
    
    
    for pid in range(num_processes):
        all_parts = []
        shard_dir = f"{name}_shard_{pid}"
        for part in os.listdir(shard_dir):
            part_path = os.path.join(shard_dir, part)
            ds = load_from_disk(part_path)
            all_parts.append(ds)
        final_dataset = concatenate_datasets(all_parts)
        # final_dataset=final_dataset.shuffle()
        print("concatenate done")
        final_dataset.save_to_disk(f"{name}_final_dataset{pid}",num_proc=10)
        all_parts.clear()

    
    # shard_prefix =f"{name}_shard_"
    # root_dir = "."
    # print("正在删除临时采样目录...")
    # for d in os.listdir(root_dir):
    #     if d.startswith(shard_prefix) and os.path.isdir(os.path.join(root_dir, d)):
    #         shutil.rmtree(os.path.join(root_dir, d))
    #         print(f"已删除：{d}")
    # print("全部删除")

    gc.collect()
    