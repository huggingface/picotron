import torch.distributed as dist
import torch
from .. import process_group_manager as pgm
import torch.nn.functional as F

from typing import Tuple

def merge_first_two_dims(grad_output: torch.Tensor, input_: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge the first two dimensions of tensors."""
    return grad_output.contiguous().view(-1, *grad_output.shape[2:]), input_.contiguous().view(-1, *input_.shape[2:])

def split_tensor_along_last_dim(tensor, num_partitions):
    """Split a tensor along its last dimension into num_partitions chunks."""
    last_dim = tensor.dim() - 1
    assert tensor.size()[last_dim] % num_partitions == 0, f"{tensor.size()[last_dim]} is not divisible by {num_partitions}"
    last_dim_size = tensor.size()[last_dim] // num_partitions
    return torch.split(tensor, last_dim_size, dim=last_dim)

class CopyToModelParallelRegion(torch.autograd.Function):
    """
    Copy in forward pass, all-reduce in backward pass.
    This is the `f` function in the paper: https://arxiv.org/abs/1909.08053
    """
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
          return grad_output
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return grad_output

class ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    All-reduce in forward pass, identity in backward pass.
    This is the `g` function in the paper: https://arxiv.org/abs/1909.08053
    """
    @staticmethod
    def forward(ctx, x):
        if pgm.process_group_manager.tp_world_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather in forward pass, split in backward pass."""
    @staticmethod
    def forward(ctx, x):
        if pgm.process_group_manager.tp_world_size == 1:
            return x
        last_dim = x.dim() - 1
        # Need contiguous tensors for collectives -> https://github.com/pytorch/pytorch/blob/main/torch/distributed/nn/functional.py#L321
        x = x.contiguous()
        tensor_list = [torch.empty_like(x) for _ in range(pgm.process_group_manager.tp_world_size)]
        tensor_list[pgm.process_group_manager.tp_rank] = x
        dist.all_gather(tensor_list, x, group=pgm.process_group_manager.tp_group)
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        # Split gradient according to TP size
        chunks = split_tensor_along_last_dim(grad_output, pgm.process_group_manager.tp_world_size)
        return chunks[pgm.process_group_manager.tp_rank].contiguous()

class LinearWithAsyncAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        output = input_ @ weight.t() + bias if bias is not None else input_ @ weight.t()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        The key difference with "linear_with_all_reduce" is that the all reduce of input_ gradeint is before 
        the calculation of the gradient of weights and bias, instead of after. So we can overlap the computation and communication
        This is only applicable to Column Parallel Linear

        Before: grad_output -> grad_input, grad_weight, grad_bias  -> grad_input all reduce
        Now:    grad_output -> grad_input -> grad_input all reduce -> grad_weight, grad_bias
        """
        input_, weight = ctx.saved_tensors
        grad_input = grad_output @ weight # (b, s, out_size) @ (out_size, input_size) = (b, s, input_size)
        # all-reduce input gradient. 
        input_gradient_all_reduce_handle = dist.all_reduce(grad_input, group=pgm.process_group_manager.tp_group, async_op=True)
        # merge first two dims to allow matrix multiplication
        grad_output, input_ = merge_first_two_dims(grad_output, input_)     # grad_output, input_: (b, s, out_size), (b, s, input_size) -> (b*s, out_size), (b*s, input_size)
        grad_weight = grad_output.t() @ input_                              # (out_size, b*s) @ (b*s, input_size) -> (out_size, input_size)
        grad_bias = grad_output.sum(0) if ctx.use_bias else None
        input_gradient_all_reduce_handle.wait()
        return grad_input, grad_weight, grad_bias

def linear_with_all_reduce(x, weight, bias):
    input_parallel = CopyToModelParallelRegion.apply(x)
    output = F.linear(input_parallel, weight, bias) # XW_i^T + b, output is Y_i
    return output

def linear_with_async_all_reduce(x, weight, bias):
    return LinearWithAsyncAllReduce.apply(x, weight, bias)