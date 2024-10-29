"""Training script for LLaMA model.
torchrun --nproc_per_node 1 --master_addr localhost --master_port 25500 train.py --use_wandb
torchrun --nproc_per_node 2 --master_addr localhost --master_port 25500 train.py --dp_size 2 --use_wandb 
torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 train.py --tp_size 2 --pp_size 2 --use_wandb 
torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 train.py --tp_size 2 --pp_size 2 --load_path ckpt/150
torchrun --nproc_per_node 8 --master_addr localhost --master_port 25500 train.py --tp_size 2 --dp_size 2 --pp_size 2 --use_wandb
CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -p 5678 -m torch.distributed.run -- --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 train.py --dp_size 2
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun    --nproc_per_node=4    --nnodes=1    --rdzv_backend=c10d    --rdzv_endpoint=localhost:29400    --max_restarts=0    --tee=3    train.py
#VERBOSE=0 torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 train.py --pp_size 2 --dp_size 2
"""

import os
import time
import argparse
from src.parallel.context_parallel import parallel_input
import torch.nn.functional as F
import torch, torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoConfig
import numpy as np
from src.parallel.tensor_parallel.tensor_parallel import TensorParallel
import src.distributed.process_group_manager as pgm
from utils import MicroBatchDataLoader, set_all_seed, print, to_readable_format, save_checkpoint, load_checkpoint
from src.distributed.process_group_manager import setup_process_group_manager
from src.parallel.pipeline_parallel import train_step_pipeline_1f1b, train_step_pipeline_afab, PipelineParallel
from src.parallel.data_parallel.data_parallel_bucket import DataParallel
from model import Llama
import wandb
from src.distributed.distributed_primtives import all_reduce_loss_across_dp_cp_ranks

def train_step(model, data_loader, device):
    acc_loss = 0.0
    
    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1
    for i in range(data_loader.grad_acc):
        # get the next batch
        batch = next(data_loader)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        input_ids, target_ids = parallel_input(input_ids, target_ids) # for context parallel, we need to split the input

        # disable gradient synchronization for all but the last micro-batch
        if requires_grad_sync:
            model.require_backward_grad_sync = (i == data_loader.grad_acc - 1)

        outputs = model(input_ids=input_ids)

        # compute the loss
        batch_size, seq_len = input_ids.shape
        target_ids = target_ids.reshape(-1)
        outputs = outputs.view(seq_len*batch_size, -1)
        loss = F.cross_entropy(outputs, target_ids, reduction='mean') / data_loader.grad_acc
        
        loss.backward()

        acc_loss += loss.item()

    return acc_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--load_path", type=str, default="", help="Path to load the model from")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt", help="Directory to save checkpoints")
    parser.add_argument("--ckpt_freq", type=int, default=300, help="Frequency to save checkpoints")
    
    args = parser.parse_args()
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["FLASH_ATTEN"] = "1" # Use cuda kernels from flash attention repo to accelerate the training. Model dtype should be torch.float16!
    os.environ["DEVICE"] = "cuda" if not args.use_cpu else "cpu"
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not args.use_cpu else torch.float32 # if GPU is not available or not supported, use torch.float32
    assert (dtype == torch.bfloat16 and os.getenv("FLASH_ATTEN") == "1") or os.getenv("FLASH_ATTEN") != "1", "Kernel operations requires dtype=torch.bfloat16"

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])
    
    ## hyperparameters
    SEQ_LEN, LOCAL_BATCH_SIZE, MICRO_BATCH_SIZE, LEARNING_RATE, NUM_SAMPLES, MAX_TOKENS, SEED = 1024, 64, 32, 3e-4, 400000, None, 42
    total_train_steps = 200
    grad_acc = 1
    
    assert SEQ_LEN % args.cp_size == 0, "SEQ_LEN must be divisible by cp_size for Context Parallelism"

    backend = "gloo" if args.use_cpu else "nccl"
    
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    
    dist.init_process_group(rank=local_rank, world_size=world_size, backend=backend, init_method=f"tcp://{host}:{port}")
    
    setup_process_group_manager(tp_size=args.tp_size, cp_size=args.cp_size, pp_size=args.pp_size, dp_size=args.dp_size)
    is_wandb_rank = pgm.process_group_manager.tp_rank == 0 and pgm.process_group_manager.dp_rank == 0 and pgm.process_group_manager.cp_rank == 0 and pgm.process_group_manager.pp_is_last_stage

    # if pgm.process_group_manager.global_rank == 0:
        # display_4D_parallelism_grid()
    
    tokens_per_step = LOCAL_BATCH_SIZE * SEQ_LEN * grad_acc * args.dp_size
    if pgm.process_group_manager.global_rank == 0:
        print("Tokens per step:", to_readable_format(tokens_per_step), is_print_rank=is_wandb_rank)
    set_all_seed(SEED)

    dataset_name = "roneneldan/TinyStories"
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
    # model_name = "meta-llama/Llama-2-7b-hf"
    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = 16
    config.num_attention_heads = 16
    config.num_key_value_heads = 4

    start_time = time.time()
    model = Llama(config=config)
    print("init model time:", time.time()-start_time, is_print_rank=is_wandb_rank)
    
    if is_wandb_rank and args.use_wandb:
        wandb.init(
            project="picotron",
            name=f"test_convergence_GBS_{tokens_per_step}_{pgm.process_group_manager}",
            config={
                "tensor_parallel_size": pgm.process_group_manager.tp_size,
                "pipeline_parallel_size": pgm.process_group_manager.pp_size,
                "data_parallel_size": pgm.process_group_manager.dp_size,
                "model": model_name,
                "dataset": dataset_name,
                "max_tokens": MAX_TOKENS,
                "learning_rate": LEARNING_RATE,
                "seed": SEED,
                "micro_batch_size": MICRO_BATCH_SIZE,
                "global_batch_size": LOCAL_BATCH_SIZE * args.dp_size * grad_acc,
                "gradient_accumulation": grad_acc,
            },
        )

    start_time = time.time()
    if pgm.process_group_manager.tp_world_size > 1:
        TensorParallel(model)

    if pgm.process_group_manager.pp_world_size > 1:
        model = PipelineParallel(model, config)
    
    model.to(dtype).to(device)
    
    # Context parallel and Data parallel both need gradient synchronization
    if pgm.process_group_manager.cp_dp_world_size > 1:
        model = DataParallel(model)
    
    print("init parallel time:", time.time()-start_time, is_print_rank=is_wandb_rank)
    start_time = time.time()
    
    model.train()
    print("model to device time:", time.time()-start_time, is_print_rank=is_wandb_rank)
    
    start_time = time.time()
    data_loader = MicroBatchDataLoader(local_batch_size=LOCAL_BATCH_SIZE, micro_batch_size=MICRO_BATCH_SIZE, seq_length=SEQ_LEN, dataset_name=dataset_name, tokenizer_name=model_name, grad_acc = grad_acc,num_workers=4, num_proc=4, num_samples=NUM_SAMPLES)
    print("init dataloader time:", time.time()-start_time, is_print_rank=is_wandb_rank)
    tensor_shapes = (data_loader.micro_batch_size, data_loader.seq_length_per_gpu, config.hidden_size)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    trained_tokens, step = 0, 0
    if args.load_path:
        step, trained_tokens = load_checkpoint(model, optimizer, args.load_path)
    
    checkpoint_dir = args.ckpt_dir
    checkpoint_freq = args.ckpt_freq

    dist.barrier()

    #TODO: Double-check consumed tokens after each steps (for example, MICRO_BATCH_SIZE=2 and using only dp_size=4, num_local_micro_batches=0 => division by 0)
    #TODO: Check convergence
    #TODO: Try multi-nodes
    #TODO: Add activation checkpointing
    #TODO: add gradient accumulation
    
    while trained_tokens < MAX_TOKENS:
        #TODO: Add epoch support
        # data_loader.set_epoch(step)
        step_start_time = time.time()
        optimizer.zero_grad()
        
        if pgm.process_group_manager.pp_world_size > 1:
            loss = train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype)
        else:
            loss = train_step(model, data_loader, device)
            
        loss = all_reduce_loss_across_dp_cp_ranks(loss, device)
        
        optimizer.step()
        trained_tokens += tokens_per_step
        step += 1
        
        # In DDP implementation I need to reset the gradient buffers
        if hasattr(model, 'reset'):
            model.reset()

        step_duration = time.time() - step_start_time
        
        if is_wandb_rank:
            print(f"[rank {pgm.process_group_manager.global_rank}] Step: {step}, Loss: {loss:.4f}, "
                f"Global batch size: {to_readable_format(tokens_per_step)}, "
                f"Tokens/s: {to_readable_format(tokens_per_step / step_duration)}, "
                f"Tokens/s/GPU: {to_readable_format(tokens_per_step / step_duration / world_size)}, "
                f"Tokens: {to_readable_format(trained_tokens)}{('/' + to_readable_format(MAX_TOKENS)) if MAX_TOKENS else ''}, "
                f"Memory usage: {torch.cuda.memory_reserved() / 1e9:.2f}GB"
            , is_print_rank=is_wandb_rank)
        
            if args.use_wandb:
                wandb.log({"loss": loss, "tokens_per_step": tokens_per_step, "tokens_per_second": tokens_per_step / step_duration,\
                    "memory_usage": torch.cuda.memory_reserved() / 1e9, "trained_tokens": trained_tokens})
        
        if step % checkpoint_freq == 0:
            save_checkpoint(model, optimizer, step, trained_tokens, checkpoint_dir+f"/{step}")
        
        if step >= total_train_steps:
            break
    
    if is_wandb_rank and args.use_wandb:
        wandb.finish()

    dist.destroy_process_group()
