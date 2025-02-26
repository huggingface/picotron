"""
torchrun --nproc_per_node 2 --master_addr localhost --master_port 25500 test_dataloader.py
"""
from picotron.data import MicroBatchDataLoader
import torch.distributed as dist
import os
import datetime
from picotron.process_group_manager import setup_process_group_manager

import torch
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from functools import partial
from datasets import Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer

import picotron.process_group_manager as pgm

# remove context parallelism split. as a reference
class DummyDataLoader(DataLoader):
    def __init__(self,  micro_batch_size, seq_length, dataset_name, tokenizer_name, num_workers, num_proc, grad_acc_steps, split="train", num_samples=None, pin_memory=True):
        self.micro_batch_size = micro_batch_size
        self.seq_length = seq_length
        self.grad_acc_steps = grad_acc_steps
        self.global_batch_size = micro_batch_size * grad_acc_steps * pgm.process_group_manager.dp_world_size
        self.num_global_micro_batches = self.global_batch_size // self.micro_batch_size

        self.seq_length_per_gpu = seq_length // pgm.process_group_manager.cp_world_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = load_dataset(dataset_name, split=split)
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
        input_ids = batch_input_ids[:, :self.seq_length].contiguous()
        target_ids = batch_input_ids[:, 1:self.seq_length+1].contiguous()
        position_ids = torch.arange(0, self.seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()

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

# test the tokens are split correctly in context parallelism
# TODO: test zigzag behavior
def test_cp_behavior(TP_SIZE, CP_SIZE, PP_SIZE, DP_SIZE, SEQ_LEN=8):
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl"

    assert SEQ_LEN % CP_SIZE == 0, "SEQ_LEN must be divisible by cp_size for Context Parallelism"
    dist.init_process_group(rank=global_rank, world_size=world_size, backend=backend, init_method="env://", timeout=datetime.timedelta(minutes=3))
    setup_process_group_manager(tp_size=TP_SIZE, cp_size=CP_SIZE, pp_size=PP_SIZE, dp_size=DP_SIZE)

    data_loader = MicroBatchDataLoader(
        micro_batch_size=2,
        seq_length=SEQ_LEN,
        dataset_name="roneneldan/TinyStories",
        tokenizer_name="HuggingFaceTB/SmolLM-135M",
        grad_acc_steps=1,
        device=f"cuda:{local_rank}",
        num_workers=1,
        num_proc=1,
        num_samples=10,
        pin_memory=False
    )

    ref_data_loader = DummyDataLoader(
        micro_batch_size=2,
        seq_length=SEQ_LEN,
        dataset_name="roneneldan/TinyStories",
        tokenizer_name="HuggingFaceTB/SmolLM-135M",
        grad_acc_steps=1,
        num_workers=1,
        num_proc=1,
        num_samples=10,
        pin_memory=False
    )

    for i in range(1):
        ref_batch = next(ref_data_loader)
        batch = next(data_loader)
        split_size = ref_batch["input_ids"].shape[1] // pgm.process_group_manager.cp_world_size
        start_idx = split_size * global_rank
        end_idx = start_idx + split_size
        assert torch.equal(ref_batch["input_ids"][:,start_idx:end_idx], batch["input_ids"]), "input_ids are not equal"

# test the infinite loop behavior
def test_infinite_loop():
    local_rank = 0
    global_rank = 0
    world_size = 1
    backend = "nccl"

    dist.init_process_group(rank=global_rank, world_size=world_size, backend=backend, init_method="env://", timeout=datetime.timedelta(minutes=3))
    setup_process_group_manager(tp_size=1, cp_size=1, pp_size=1, dp_size=1)

    data_loader = MicroBatchDataLoader(
        micro_batch_size=2,
        seq_length=256,
        dataset_name="roneneldan/TinyStories",
        tokenizer_name="HuggingFaceTB/SmolLM-135M",
        grad_acc_steps=1,
        device=f"cuda:{local_rank}",
        num_workers=1,
        num_proc=1,
        num_samples=2,
    )

    s = set()
    for _ in range(10):
        batch = next(data_loader)
        # Convert the nested list to a tuple of tuples
        batch_tuple = tuple(tuple(x) for x in batch["input_ids"].tolist())
        if batch_tuple in s:
            assert True
        s.add(batch_tuple)
    assert False


if __name__ == "__main__":
    # test_infinite_loop()
    test_cp_behavior(TP_SIZE=1, CP_SIZE=2, PP_SIZE=1, DP_SIZE=1, SEQ_LEN=8)
