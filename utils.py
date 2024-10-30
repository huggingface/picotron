import torch
import random
import os
import numpy as np
import builtins
import fcntl
import src.distributed.process_group_manager as pgm
import torch, torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial
from datasets import Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer

def print(*args, is_print_rank=True, **kwargs):
    """ solves multi-process interleaved print problem """
    if not is_print_rank: return
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

def set_all_seed(seed):
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
def to_readable_format(num, precision=2):
    if num >= 1e12:
        return f"{num / 1e12:.{precision}f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

def save_checkpoint(model, optimizer, trained_steps, trained_tokens, out_dir):
    """Save the model/optimizer states/steps to a checkpoint file."""
    tp_rank, pp_rank = pgm.process_group_manager.tp_rank, pgm.process_group_manager.pp_rank
    tp_world_size, pp_world_size = pgm.process_group_manager.tp_world_size, pgm.process_group_manager.pp_world_size
    ckpt_name = f"weights_tp_rank_world_size={tp_rank}_{tp_world_size}_pp_rank_world_size={pp_rank}_{pp_world_size}.pth"
    path = os.path.join(out_dir, ckpt_name)
    
    # Only DP/CP rank 0 will save the model, the weights are the same across all ranks
    if pgm.process_group_manager.dp_rank == 0 and pgm.process_group_manager.cp_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        raw_model = model.module if pgm.process_group_manager.cp_dp_world_size > 1 else model
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'trained_steps': trained_steps,
            'trained_tokens': trained_tokens
        }
        torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, out_dir):
    """Load the model/optimizer states from the latest checkpoint. Assume the topology is the same."""
    tp_rank, pp_rank = pgm.process_group_manager.tp_rank, pgm.process_group_manager.pp_rank
    tp_world_size, pp_world_size = pgm.process_group_manager.tp_world_size, pgm.process_group_manager.pp_world_size
    ckpt_name = f"weights_tp_rank_world_size={tp_rank}_{tp_world_size}_pp_rank_world_size={pp_rank}_{pp_world_size}.pth"
    path = os.path.join(out_dir, ckpt_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    checkpoint = torch.load(path)

    # Load model weights
    raw_model = model.module if pgm.process_group_manager.cp_dp_world_size > 1 else model
    raw_model.load_state_dict(checkpoint['model'])
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['trained_steps'], checkpoint['trained_tokens']         

class MicroBatchDataLoader(DataLoader):
    def __init__(self,  micro_batch_size, seq_length, dataset_name, tokenizer_name, num_workers, num_proc, grad_acc, split="train", num_samples=None):
        
        self.micro_batch_size = micro_batch_size
        self.seq_length = seq_length
        self.grad_acc = grad_acc

        self.local_batch_size = micro_batch_size * grad_acc
        self.global_batch_size = self.local_batch_size * pgm.process_group_manager.dp_world_size
        self.num_local_micro_batches = self.local_batch_size // self.micro_batch_size
        self.num_global_micro_batches = self.global_batch_size // self.micro_batch_size
        
        self.seq_length_per_gpu = seq_length // pgm.process_group_manager.cp_world_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = load_dataset(dataset_name, split=split)
        if num_samples:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        
        dist.barrier()
        
        # Tokenize and chunk the dataset
        self.tokenized_dataset = self.tokenize_dataset(self.dataset, "text", self.seq_length, num_proc)
        
        self.sampler = DistributedSampler(
            self.tokenized_dataset, 
            num_replicas=pgm.process_group_manager.dp_world_size, 
            rank=pgm.process_group_manager.dp_rank, 
            shuffle=False
        )
        
        super().__init__(
            self.tokenized_dataset, 
            batch_size=micro_batch_size if pgm.process_group_manager.pp_world_size > 1 else self.local_batch_size, # in PP we split a single batch into multiple micro-batches
            collate_fn=self.collate_batch, 
            pin_memory=True, 
            num_workers=num_workers, 
            sampler=self.sampler, 
            shuffle=False
        )

    @staticmethod
    def tokenizer_group_text(examples, tokenizer, sequence_length):
        """Tokenize a list of texts and group them in chunks of sequence_length + 1"""
        tokenized_text_batch = tokenizer.batch_encode_plus(
            examples,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors='np'
        )
        concatenated_tokens = {'input_ids': np.concatenate(tokenized_text_batch['input_ids'])}
        total_length = len(concatenated_tokens['input_ids'])
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        result = {
            'input_ids': [
                concatenated_tokens['input_ids'][i : i + sequence_length + 1]
                for i in range(0, total_length - sequence_length, sequence_length)
            ]
        }
        return result

    def tokenize_dataset(self, dataset, text_column_name, sequence_length, num_proc):
        """Tokenize the dataset and group texts in chunks of sequence_length + 1"""
        # Create a partial function with fixed arguments
        tokenizer_func = partial(
            self.tokenizer_group_text,
            tokenizer=self.tokenizer,
            sequence_length=sequence_length
        )

        tokenized_dataset = dataset.map(
            tokenizer_func,
            input_columns=text_column_name,
            remove_columns=dataset.column_names,
            features=Features({
                "input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)
            }),
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {sequence_length+1}",
        )

        return tokenized_dataset

    def collate_batch(self, batch):
        batch_input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        batch_size = batch_input_ids.size(0)
        start_idx = pgm.process_group_manager.cp_rank * self.seq_length_per_gpu
        end_idx = start_idx + self.seq_length_per_gpu
        input_ids = batch_input_ids[:, start_idx:end_idx].contiguous()
        target_ids = batch_input_ids[:, start_idx+1:end_idx+1].contiguous()
        position_ids = torch.arange(start_idx, end_idx, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous() 
        local_attn_mask = torch.tril(torch.ones((self.seq_length_per_gpu, self.seq_length_per_gpu), dtype=torch.bool))
        attn_mask = local_attn_mask.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "position_ids": position_ids,
            "attn_mask": attn_mask,
            "hidden_states": None
        }
    
    def __iter__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = None
            raise StopIteration
        return batch