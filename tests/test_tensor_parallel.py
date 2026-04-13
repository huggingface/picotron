"""
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 test_tensor_parallel.py
CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -p 5678 -m torch.distributed.run -- --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 test_tensor_parallel.py
"""

import datetime
import os

import torch
import torch.distributed as dist

from picotron.process_group_manager import setup_process_group_manager
from picotron.tensor_parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from picotron.utils import set_all_seed

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device("cuda", local_rank)

dist.init_process_group(rank=global_rank, world_size=world_size, backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=3))
setup_process_group_manager(tp_size=world_size, cp_size=1, pp_size=1, dp_size=1)

set_all_seed(42)

batch_size, seq_len = 2, 4
input_size, output_size = 8, 16
bias = True                 # linear layer with/without bias
async_all_reduce = False    # async all-reduce or not for column parallel linear layer

# Initialize input tensor
tensor_shape = (batch_size, seq_len, input_size) 
tensor = torch.randn(tensor_shape, device=device, requires_grad=True)
column_parallel_tensor = tensor.clone().detach().requires_grad_(True)
row_parallel_tensor = tensor.clone().chunk(world_size, dim=-1)[local_rank].detach().requires_grad_(True)

# Initialize column/row parallel layers
column_parallel_linear = ColumnParallelLinear(input_size, output_size, bias=bias, gather_output=True, async_all_reduce=async_all_reduce).to(device)
row_parallel_linear = RowParallelLinear(input_size, output_size, bias=bias).to(device)
linear_layer = torch.nn.Linear(input_size, output_size, bias=bias, device=device)

# copy weight and bias from reference linear layer to column/row parallel layers
column_parallel_linear.weight = torch.nn.Parameter(linear_layer.weight.chunk(world_size, dim=0)[local_rank])
row_parallel_linear.weight = torch.nn.Parameter(linear_layer.weight.chunk(world_size, dim=1)[local_rank])
if bias:
    column_parallel_linear.bias = torch.nn.Parameter(linear_layer.bias.chunk(world_size, dim=0)[local_rank])
    row_parallel_linear.bias = torch.nn.Parameter(linear_layer.bias)  

### forward pass ###
output_reference = linear_layer(tensor)
output_column_parallel = column_parallel_linear(column_parallel_tensor)
output_row_parallel = row_parallel_linear(row_parallel_tensor)

# check forward output consistency
assert torch.all(torch.eq(output_reference, output_column_parallel)), "Column Parallel Linear is not equal to the reference"
torch.testing.assert_close(output_reference, output_row_parallel) # not strictly equal. precision issue

### backward pass ###
output_reference.backward(torch.ones_like(output_reference))
output_column_parallel.backward(torch.ones_like(output_column_parallel))
output_row_parallel.backward(torch.ones_like(output_row_parallel))

# check backward weight gradient, bias gradient, and input gradient consistency
# column parallel linear test
torch.testing.assert_close(linear_layer.weight.grad.chunk(world_size, dim=0)[local_rank], column_parallel_linear.weight.grad)
torch.testing.assert_close(tensor.grad, column_parallel_tensor.grad)
if bias:
    torch.testing.assert_close(linear_layer.bias.grad.chunk(world_size, dim=0)[local_rank], column_parallel_linear.bias.grad)

# row parallel linear test
torch.testing.assert_close(linear_layer.weight.grad.chunk(world_size, dim=1)[local_rank], row_parallel_linear.weight.grad)
torch.testing.assert_close(tensor.grad.chunk(world_size, dim=-1)[local_rank], row_parallel_tensor.grad)
if bias:
    torch.testing.assert_close(linear_layer.bias.grad, row_parallel_linear.bias.grad)

print(f"Rank {dist.get_rank()}: All tests passed")