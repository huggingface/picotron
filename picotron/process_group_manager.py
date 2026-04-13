import os

import torch
import torch.distributed as dist


class ProcessGroupManager:
    def __init__(self, tp_size, cp_size, pp_size, dp_size):
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.global_rank % self.world_size))
        
        assert self.world_size == tp_size * cp_size * pp_size * dp_size, f"World size ({self.world_size}) != TP ({tp_size}) * CP ({cp_size}) * PP ({pp_size}) * DP ({dp_size})"

        self.grid = torch.arange(self.world_size).view(dp_size, pp_size, cp_size, tp_size)  # DP * PP * CP * TP grid
        # Find the position of the current process in the grid
        self.dp_rank, self.pp_rank, self.cp_rank, self.tp_rank = (self.grid == self.global_rank).nonzero().flatten().tolist()

        # Process group creation - Update indexing to match new grid order
        self.tp_group = dist.new_subgroups_by_enumeration([self.grid[d, p, c, :].tolist() for d in range(dp_size) for p in range(pp_size) for c in range(cp_size)])[0]
        self.cp_group = dist.new_subgroups_by_enumeration([self.grid[d, p, :, t].tolist() for d in range(dp_size) for p in range(pp_size) for t in range(tp_size)])[0]
        self.pp_group = dist.new_subgroups_by_enumeration([self.grid[d, :, c, t].tolist() for d in range(dp_size) for c in range(cp_size) for t in range(tp_size)])[0]
        self.dp_group = dist.new_subgroups_by_enumeration([self.grid[:, p, c, t].tolist() for p in range(pp_size) for c in range(cp_size) for t in range(tp_size)])[0]
        self.cp_dp_group = dist.new_subgroups_by_enumeration([self.grid[:, p, :, t].flatten().tolist() for p in range(pp_size) for t in range(tp_size)])[0]
        self.pp_dp_group = dist.new_subgroups_by_enumeration([self.grid[:, :, c, t].flatten().tolist() for c in range(cp_size) for t in range(tp_size)])[0]

        self.world_group = dist.group.WORLD
        
        # Update group IDs with new grid ordering
        self.tp_group_ids = self.grid[self.dp_rank, self.pp_rank, self.cp_rank, :].tolist()
        self.cp_group_ids = self.grid[self.dp_rank, self.pp_rank, :, self.tp_rank].tolist()
        self.pp_group_ids = self.grid[self.dp_rank, :, self.cp_rank, self.tp_rank].tolist()
        self.dp_group_ids = self.grid[:, self.pp_rank, self.cp_rank, self.tp_rank].tolist()
        self.cp_dp_group_ids = self.grid[:, self.pp_rank, :, self.tp_rank].flatten().tolist()
               
        # Tensor parallelism
        self.tp_world_size = dist.get_world_size(group=self.tp_group)
        self.tp_first_rank = self.tp_group_ids[0]
        self.tp_last_rank = self.tp_group_ids[-1]
        
        # Context parallelism
        self.cp_world_size = dist.get_world_size(group=self.cp_group)
        self.cp_first_rank = self.cp_group_ids[0]
        self.cp_last_rank = self.cp_group_ids[-1]
        self.cp_send_rank = self.cp_group_ids[(self.cp_rank + 1) % self.cp_world_size]
        self.cp_recv_rank = self.cp_group_ids[(self.cp_rank - 1) % self.cp_world_size]

        # Pipeline parallelism
        self.pp_world_size = dist.get_world_size(group=self.pp_group)
        self.pp_first_rank = self.pp_group_ids[0]
        self.pp_last_rank = self.pp_group_ids[-1]
        self.pp_is_first_stage = self.pp_rank == 0
        self.pp_is_last_stage = self.pp_rank == self.pp_world_size - 1
        self.pp_next_rank = None if self.pp_rank == self.pp_world_size - 1 else int(self.grid[self.dp_rank, self.pp_rank + 1, self.cp_rank, self.tp_rank].item())
        self.pp_prev_rank = None if self.pp_rank == 0 else int(self.grid[self.dp_rank, self.pp_rank - 1, self.cp_rank, self.tp_rank].item())

        # Data parallelism
        self.dp_world_size = dist.get_world_size(group=self.dp_group)
        self.dp_first_rank = self.dp_group_ids[0]
        self.dp_last_rank = self.dp_group_ids[-1]
        
        # Context + Data paralellism
        self.cp_dp_world_size = dist.get_world_size(group=self.cp_dp_group)
        
    def __str__(self):
        return f"TP({self.tp_world_size})-CP({self.cp_world_size})-PP({self.pp_world_size})-DP({self.dp_world_size})-Rank({self.global_rank})"

def setup_process_group_manager(tp_size, cp_size, pp_size, dp_size):
    global process_group_manager
    process_group_manager = ProcessGroupManager(tp_size, cp_size, pp_size, dp_size)