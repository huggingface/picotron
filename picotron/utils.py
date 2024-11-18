import os
import torch
import random
import numpy as np
import builtins
import fcntl
import picotron.process_group_manager as pgm

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