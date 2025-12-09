import torch

import numpy as np
import torch.nn.functional as F

from lm_eval.base import BaseLM
from datasets import load_dataset


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_test_dataset(dataset_name, tokenizer, seqlen=2048):
    if dataset_name == "wikitext2":
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testdata = "".join(testdata["text"]).split("\n")
    elif dataset_name == "c4":
        # The standard "allenai/c4" loading script appears to be bugged and ignores the
        # 'split' parameter, defaulting to the 'train' set.
        # To work around this, we manually specify the URLs for 50% of the validation files
        # and load them directly as a generic JSON dataset.
        c4_validation_urls = [
            f"https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.{i:05d}-of-00008.json.gz"
            for i in range(2)  # There are 4 validation files in total, we take 2 for 25%
        ]
        testdata = load_dataset(
            "json",
            data_files=c4_validation_urls,
            split="train",  # 'train' is the default split name when loading from data_files
        )["text"]
    else:
        raise NotImplementedError

    # Filter empty text
    testdata = [item for item in testdata if item != ""]
    
    # Batch tokenization (significantly faster)
    print(f"Tokenizing {len(testdata)} samples...")
    batch_size = 1000  # Process 1000 samples per batch
    tokenized_text = []
    
    for i in range(0, len(testdata), batch_size):
        batch = testdata[i:i+batch_size]
        # Batch processing to avoid calling tokenizer one by one
        batch_tokenized = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False
        )["input_ids"]
        # Add eos_token to each sequence
        for ids in batch_tokenized:
            tokenized_text.append(ids + [tokenizer.eos_token_id])
        
        # Show progress
        if (i + batch_size) % 10000 == 0 or (i + batch_size) >= len(testdata):
            print(f"  Processed: {min(i + batch_size, len(testdata))}/{len(testdata)} samples")
    
    print(f"Tokenization completed!")

    data, doc = [], [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    for sen in tokenized_text:
        if len(sen) > seqlen:
            continue
        if len(doc) + len(sen) > seqlen:
            data.append(doc)
            doc = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        doc.extend(sen)
    if len(doc) > 1 and len(doc) <= seqlen:
        data.append(doc)
    return data


class LMEvalAdaptor(BaseLM):
    def __init__(
        self, model_name, model, tokenizer, batch_size=1, max_length=-1, device="cuda:0", max_gen_toks=256
    ):
        super().__init__()

        assert isinstance(batch_size, int)

        self.model_name = model_name
        self.model = model
        self.model.eval()

        self._device = device
        self.model.to(self._device)

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = batch_size

        self._max_length = max_length
        self._max_gen_toks = max_gen_toks

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length != -1:
            return self._max_length
        if hasattr(self.model.config, "n_ctx"):
            return self.model.config.n_ctx
        elif hasattr(self.model.config, "max_position_embeddings"):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, "n_positions"):
            return self.model.config.n_positions
        elif "bloom" in self.model_name:
            return 2048
        elif "llama" in self.model_name:
            return 2048  # TODO: did not check this
        elif "mpt" in self.model_name:
            return 2048
        elif "falcon" in self.model_name:
            return 2048
        else:
            print(self.model.config)
            raise NotImplementedError

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, add_special_tokens=True):
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            context, continuation = context.strip(), continuation.strip()
            if context == "":
                # end of text as context
                context_enc = [self.eot_token_id]
            else:
                context_enc = self.tok_encode(context, add_special_tokens=True)

            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            out = self.model(inps)[0]
        return out

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, 
            max_length=max_length, 
            eos_token_id=eos_token_id, 
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )
