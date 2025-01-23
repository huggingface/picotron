"""Training script for LLaMA model.
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 train.py --config tmp/dummy/llama2_7b_benchmark.json
CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -p 5678 -m torch.distributed.run -- --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 train.py --config tmp/dummy/llama2_7b_benchmark.json
"""
import os
import inspect
import json
import time
import datetime
import argparse
import torch.nn.functional as F
import torch, torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoConfig
from .picotron.context_parallel.context_parallel import apply_context_parallel
from .picotron.tensor_parallel.tensor_parallel import apply_tensor_parallel
from .picotron import process_group_manager as pgm
from .picotron.utils import average_loss_across_dp_cp_ranks, set_all_seed, print, to_readable_format, get_mfu, get_num_params
from .picotron.checkpoint import CheckpointManager
from .picotron.checkpoint import init_model_with_dematerialized_weights, init_model_with_materialized_weights
from .picotron.data import MicroBatchDataLoader
from .picotron.process_group_manager import setup_process_group_manager
from .picotron.pipeline_parallel.pipeline_parallel import train_step_pipeline_1f1b, train_step_pipeline_afab, PipelineParallel
from .picotron.data_parallel.data_parallel import DataParallelBucket
from .picotron.model import Llama
import wandb

def train_step(model, data_loader, device):
    acc_loss = 0.0
    
    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1
    for i in range(data_loader.grad_acc_steps):
        # get the next batch
        batch = next(data_loader)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        # disable gradient synchronization for all but the last micro-batch
        if requires_grad_sync:
            model.require_backward_grad_sync = (i == data_loader.grad_acc_steps - 1)

        outputs = model(input_ids=input_ids)

        # compute the loss
        batch_size, seq_len = input_ids.shape
        target_ids = target_ids.reshape(-1)
        outputs = outputs.view(seq_len*batch_size, -1)
        loss = F.cross_entropy(outputs, target_ids, reduction='mean') / data_loader.grad_acc_steps
        
        loss.backward()

        acc_loss += loss.item()

    return acc_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    
    os.environ["OMP_NUM_THREADS"] = config["environment"]["OMP_NUM_THREADS"]
    os.environ["TOKENIZERS_PARALLELISM"] = config["environment"]["TOKENIZERS_PARALLELISM"]
    os.environ["FLASH_ATTEN"] = config["environment"]["FLASH_ATTEN"]
    os.environ["DEVICE"] = "cpu" if config["distributed"]["use_cpu"] else "cuda"
    if config["environment"].get("HF_TOKEN") is None:
        if "HF_TOKEN" not in os.environ: raise ValueError("HF_TOKEN is neither set in the config file nor in the environment")
    else:
        if "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = config["environment"]["HF_TOKEN"]
        else:
            print("Warning: HF_TOKEN is set in the environment and the config file. Using the environment variable.")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not config["distributed"]["use_cpu"] else torch.float32
    assert (dtype == torch.bfloat16 and os.getenv("FLASH_ATTEN") == "1") or os.getenv("FLASH_ATTEN") != "1", "Kernel operations requires dtype=torch.bfloat16"

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "gloo" if config["distributed"]["use_cpu"] else "nccl"
    
    assert config["training"]["seq_length"] % config["distributed"]["cp_size"] == 0, "seq_length must be divisible by cp_size for Context Parallelism"
    assert world_size == config["distributed"]["tp_size"] * config["distributed"]["pp_size"] * config["distributed"]["dp_size"] * config["distributed"]["cp_size"], "world_size must be equal to tp_size * pp_size * dp_size * cp_size"

    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(rank=global_rank, world_size=world_size, backend=backend, init_method=f"env://", timeout=datetime.timedelta(minutes=3))
    setup_process_group_manager(
        tp_size=config["distributed"]["tp_size"],
        cp_size=config["distributed"]["cp_size"],
        pp_size=config["distributed"]["pp_size"],
        dp_size=config["distributed"]["dp_size"]
    )
    is_wandb_rank = pgm.process_group_manager.tp_rank == 0 and pgm.process_group_manager.dp_rank == 0 and pgm.process_group_manager.cp_rank == 0 and pgm.process_group_manager.pp_is_last_stage

    set_all_seed(config["training"]["seed"])

    start_time = time.time()
    data_loader = MicroBatchDataLoader(
        micro_batch_size=config["training"]["micro_batch_size"],
        seq_length=config["training"]["seq_length"],
        dataset_name=config["dataset"]["name"],
        tokenizer_name=config["model"]["name"],
        grad_acc_steps=config["training"]["gradient_accumulation_steps"],
        device=device,
        num_workers=config["dataset"]["num_workers"],
        num_proc=config["dataset"]["num_proc"],
        num_samples=config["training"].get("num_samples", None),
        subset_name=config["dataset"].get("subset_name", None),
        split=config["dataset"].get("split", "train")
    )

    # download model on the first rank, assume all ranks have access to the same filesystem
    if pgm.process_group_manager.global_rank == 0:
        download_model(config["model"]["name"], os.environ["HF_TOKEN"])

    dist.barrier()

    print(f"init dataloader time: {time.time()-start_time:.2f}s", is_print_rank=is_wandb_rank)
    tokens_per_step = data_loader.global_batch_size * config["training"]["seq_length"]
    
    if pgm.process_group_manager.global_rank == 0:
        print("Tokens per step:", to_readable_format(tokens_per_step), is_print_rank=is_wandb_rank)

    if is_wandb_rank and config["logging"]["use_wandb"]:
        wandb.init(
            project="picotron",
            name=f"{config['logging']['run_name']}_{to_readable_format(tokens_per_step)}_{pgm.process_group_manager}",
            config={
                "tensor_parallel_size": pgm.process_group_manager.tp_world_size,
                "context_parallel_size": pgm.process_group_manager.cp_world_size,
                "pipeline_parallel_size": pgm.process_group_manager.pp_world_size,
                "data_parallel_size": pgm.process_group_manager.dp_world_size,
                "model": config["model"]["name"],
                "dataset": config["dataset"]["name"],
                "max_tokens": config["training"]["max_tokens"],
                "learning_rate": config["training"]["learning_rate"],
                "seed": config["training"]["seed"],
                "micro_batch_size": data_loader.micro_batch_size,
                "global_batch_size": data_loader.global_batch_size,
                "gradient_accumulation": data_loader.grad_acc_steps,
            },
        )

    if pgm.process_group_manager.global_rank == 0:
        print(f"rank {pgm.process_group_manager.global_rank}: Creating model config")
        model_config = AutoConfig.from_pretrained(config["model"]["name"])
        # twist the model structure if specified in the config file
        model_config.num_hidden_layers = model_config.num_hidden_layers if "num_hidden_layers" not in config["model"] else config["model"]["num_hidden_layers"]
        model_config.num_attention_heads = model_config.num_attention_heads if "num_attention_heads" not in config["model"] else config["model"]["num_attention_heads"]
        model_config.num_key_value_heads = model_config.num_key_value_heads if "num_key_value_heads" not in config["model"] else config["model"]["num_key_value_heads"]
        model_config.max_position_embeddings = config["training"]["seq_length"]
        objects = [model_config]
    else:
        objects = [None]

    dist.broadcast_object_list(objects, src=0, device=device)
    model_config = objects[0]
    print(f"rank {pgm.process_group_manager.global_rank}: Broadcasting model_config to all ranks", is_print_rank=pgm.process_group_manager.global_rank==0)

    dist.barrier()

    print(f"rank {pgm.process_group_manager.global_rank}: Initializing model meta device", is_print_rank=is_wandb_rank)

    start_time = time.time()

    with init_model_with_dematerialized_weights():
        model = Llama(config=model_config)

        if pgm.process_group_manager.tp_world_size > 1:
            model = apply_tensor_parallel(model)

        if pgm.process_group_manager.pp_world_size > 1:
            model = PipelineParallel(model, model_config)

    model = init_model_with_materialized_weights(model, model_config, save_dir=f"./hf_model_safetensors/")

    #TODO: load existing checkpoint here to continue pre-training

    if pgm.process_group_manager.cp_world_size > 1:
        model = apply_context_parallel(model)

    model.to(dtype).to(device)
    
    if pgm.process_group_manager.dp_world_size > 1:
        model = DataParallelBucket(model)
    
    print(f"init model parallel time: {time.time()-start_time:.2f}s", is_print_rank=is_wandb_rank)
    
    model.train()
    num_params = get_num_params(model)
    print(f"Number of parameters: {to_readable_format(num_params)}", is_print_rank=is_wandb_rank)
    
    tensor_shapes = (data_loader.micro_batch_size, data_loader.seq_length_per_gpu, model_config.hidden_size)
    
    extra_args = dict()
    if config["model"]["use_fused_adam"]:
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"], **extra_args)
    
    checkpoint_manager = CheckpointManager()

    trained_tokens, step = 0, 0
    if config["checkpoint"]["load_path"]:
        step, trained_tokens = checkpoint_manager.load_checkpoint(model, optimizer, config["checkpoint"]["load_path"])
    
    dist.barrier()
    
    while config["training"]["max_tokens"] is None or trained_tokens < config["training"]["max_tokens"]:
        step_start_time = time.time()
        optimizer.zero_grad()
        
        if pgm.process_group_manager.pp_world_size > 1:
            if config["distributed"]["pp_engine"] == "afab":
                loss = train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype)
            elif config["distributed"]["pp_engine"] == "1f1b":
                loss = train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device, dtype)
            else:
                raise ValueError(f"Invalid pipeline parallel engine: {config['distributed']['pp_engine']}")
        else:
            loss = train_step(model, data_loader, device)
            
        loss = average_loss_across_dp_cp_ranks(loss, device)
        
        optimizer.step()
        trained_tokens += tokens_per_step
        step += 1
        
        if hasattr(model, 'reset'):
            model.reset()

        step_duration = time.time() - step_start_time
        tokens_per_second = tokens_per_step / step_duration
        tokens_per_second_per_gpu = tokens_per_second / world_size
        mfu = get_mfu(tokens_per_second_per_gpu, num_params, model_config)
        
        if is_wandb_rank:
            print(
                f"[rank {pgm.process_group_manager.global_rank}] "
                f"Step: {step:<5d} | "
                f"Loss: {loss:6.4f} | "
                f"Global batch size: {to_readable_format(tokens_per_step):>7s} | "
                f"Tokens/s: {to_readable_format(tokens_per_second):>7s} | "
                f"Tokens/s/GPU: {to_readable_format(tokens_per_second_per_gpu):>7s} | "
                f"Tokens: {to_readable_format(trained_tokens):>7s}{('/' + to_readable_format(config['training']['max_tokens'])) if config['training']['max_tokens'] else ''} | "
                f"MFU: {mfu:5.2f}% | "
                f"Memory usage: {torch.cuda.memory_reserved() / 1e9:6.2f}GB",
                is_print_rank=is_wandb_rank
            )
        
            if config["logging"]["use_wandb"]:
                wandb.log({
                    "loss": loss,
                    "tokens_per_step": tokens_per_step,
                    "tokens_per_second": tokens_per_step / step_duration,
                    "mfu": mfu,
                    "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                    "memory_usage": torch.cuda.memory_reserved() / 1e9,
                    "trained_tokens": trained_tokens
                })
        
        if step % config["checkpoint"]["save_frequency"] == 0:
            checkpoint_manager.save_checkpoint(model, optimizer, step, trained_tokens, config["checkpoint"]["save_dir"]+f"/{step}")
        
        if step >= config["training"]["total_train_steps"]:
            break
    
    if is_wandb_rank and config["logging"]["use_wandb"]:
        wandb.finish()

    dist.destroy_process_group()