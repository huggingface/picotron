import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from functools import partial
from datasets import Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer
from .utils import print

from . import process_group_manager as pgm

class MicroBatchDataLoader(DataLoader):
    def __init__(self,  micro_batch_size, seq_length, dataset_name, tokenizer_name, num_workers, num_proc, grad_acc_steps, device, subset_name=None, split="train", num_samples=None, pin_memory=True):
        self.micro_batch_size = micro_batch_size
        self.seq_length = seq_length
        self.grad_acc_steps = grad_acc_steps
        self.global_batch_size = micro_batch_size * grad_acc_steps * pgm.process_group_manager.dp_world_size
        self.num_global_micro_batches = self.global_batch_size // self.micro_batch_size
        
        self.seq_length_per_gpu = seq_length // pgm.process_group_manager.cp_world_size
        self.dataset = load_dataset(dataset_name, split=split, name=subset_name)

        if pgm.process_group_manager.global_rank == 0:
            print(f"rank {pgm.process_group_manager.global_rank}: Creating tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            objects = [self.tokenizer]
        else:
            objects = [None]

        print(f"rank {pgm.process_group_manager.global_rank}: Broadcasting tokenizer to all ranks", is_print_rank=pgm.process_group_manager.global_rank==0)
        dist.broadcast_object_list(objects, src=0, device=device)
        self.tokenizer = objects[0]
        
        if num_samples:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        
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
            batch_size=micro_batch_size,
            collate_fn=self.collate_batch, 
            pin_memory=pin_memory, 
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
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "position_ids": position_ids,
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
            # Reinitialize the sampler and iterator
            self.sampler.set_epoch(self.sampler.epoch + 1 if hasattr(self.sampler, 'epoch') else 0)
            self._iterator = super().__iter__()
            try:
                batch = next(self._iterator)
            except StopIteration:
                self._iterator = None
                raise StopIteration
        return batch