import torch
import random
import numpy as np
import builtins
import fcntl
from . import process_group_manager as pgm
import torch, torch.distributed as dist

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

# ref: 
# https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L289
# https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_02.py#L950
def get_mfu(tokens_per_second, num_params, model_config, theoretical_flops = 989.5 * 10 ** 12):
    num_layers = model_config.num_hidden_layers
    hidden_dim = model_config.hidden_size
    seq_len = model_config.max_position_embeddings
    flops_per_token = 6 * num_params + 12 * num_layers * hidden_dim * seq_len
    mfu = tokens_per_second * flops_per_token / theoretical_flops * 100 # percentage
    return mfu

def get_num_params(model):
    """Calculate total number of parameters accounting for tensor parallelism and pipeline parallelism.
    
    For TP: Parameters in attention/mlp/embed/final_proj are sharded, so multiply by tp_world_size
    For PP: Need to gather parameter counts across pipeline stages
    For DP: Parameters are replicated, so only count once
    
    Note: 
    FSDP: Parameters are sharded across data parallel ranks
    """
    tp_world_size = pgm.process_group_manager.tp_world_size
    
    # Count parameters in current PP rank
    local_num_params = 0
    for name, param in model.named_parameters():
        # Parameters split across TP ranks
        # TODO: LayerNorm is also split across TP ranks for sequence parallelism
        if any(tp_keyword in name.lower() for tp_keyword in ['attention', 'mlp', 'embed', 'final_proj']):
            local_num_params += param.numel() * tp_world_size
        else:
            # Parameters replicated across TP ranks (layer norm, biases)
            local_num_params += param.numel()
            
    # Gather parameter counts from all PP ranks
    param_counts = torch.tensor(local_num_params, device='cuda')
    
    # Sum up parameters across all PP ranks
    dist.all_reduce(param_counts, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.pp_group)
    
    return param_counts.item()
    
def assert_no_meta_tensors(model):
    meta_tensors = []
    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            meta_tensors.append(f"Parameter '{name}' with shape {param.shape}")
    
    for name, buffer in model.named_buffers():
        if buffer.device == torch.device("meta"):
            meta_tensors.append(f"Buffer '{name}' with shape {buffer.shape}")
    
    assert len(meta_tensors) == 0, f"Found {len(meta_tensors)} meta tensors:\n" + "\n".join(meta_tensors)

def average_loss_across_dp_cp_ranks(loss, device):
    reduced_loss = torch.tensor([loss if loss is not None else 0.0], dtype=torch.float32, device=device)
    if pgm.process_group_manager.pp_is_last_stage:
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)
        reduced_loss /= pgm.process_group_manager.cp_dp_world_size
    return reduced_loss.item()