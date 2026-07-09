from typing import List
import torch
import torch.distributed as dist
from torch import nn

class Bucket:
    def __init__(self, params: List[torch.nn.Parameter], grad_data: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> None:
        """
        Initializes a Bucket instance.

        Args:
            params (List[torch.nn.Parameter]): List of parameters assigned to this bucket.
            grad_data (torch.Tensor): Tensor to store the gradients for this bucket.
            process_group (torch.distributed.ProcessGroup): Process group used for synchronizing gradients.
        """
        self.params = set(params)    # Set of parameters in this bucket.
        self.params_with_grad_ready = set() # Parameters that have their gradients ready for synchronization. launch all reduce when all parameters have gradients ready
        self.grad_data = grad_data  # Tensor that stores gradients for all parameters in this bucket.
        self.process_group = process_group  # Process group for gradient synchronization.
        self.process_group_size = dist.get_world_size(group=self.process_group) 
        self.handle = None # Handle for the async allreduce operation.
        
        self.reset()
    
    def sync_gradient(self) -> None:
        """
        Launch an asynchronous all-reduce operation to synchronize gradients across processes.
        """
        assert self.handle is None
        self.grad_data /= self.process_group_size
        self.handle = dist.all_reduce(self.grad_data, group=self.process_group, async_op=True)
    
    def reset(self) -> None:
        """
        Reset the bucket to its initial state. Typically called after the gradient synchronization is finished.
        """
        self.handle = None
        self.params_with_grad_ready.clear() # Clear the set of parameters ready for gradient synchronization.
        self.grad_data.zero_() # Zero the gradient tensor.

    def wait(self) -> None:
        """
        wait for the allreduce operation to finish    
        """
        assert self.handle is not None, "You should launch an allreduce operation before waiting for it to finish"
        self.handle.wait() # Block until the all-reduce operation finishes.

    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        """
        Mark a parameter as ready for gradient synchronization. Launches synchronization when all parameters in the
        bucket have their gradients ready.
        """
        assert param in self.params and param not in self.params_with_grad_ready
        self.params_with_grad_ready.add(param)
        # When all parameters in the bucket have their gradients ready, synchronize gradients
        if len(self.params_with_grad_ready) == len(self.params):
            self.sync_gradient()

class BucketManager:
    def __init__(self, params: List[torch.nn.Parameter], process_group: torch.distributed.ProcessGroup, bucket_size: int, grad_type: torch.dtype = torch.float32) -> None:
        """
        Initializes the BucketManager.

        Args:
            params (List[torch.nn.Parameter]): List of model parameters.
            process_group (torch.distributed.ProcessGroup): Process group used for gradient synchronization.
            bucket_size (int): Maximum size of each bucket in terms of gradient elements.
            grad_type (torch.dtype, optional): Data type of gradients, defaults to torch.float32.
        """
        self.params = list(params) # Convert parameter generator to a list.
        self.device = self.params[0].device if self.params[0].is_cuda else torch.device("cpu")
        self.buckets = [] # List of buckets.
        self.process_group = process_group
        self.process_group_size = dist.get_world_size(group=self.process_group)
        self.params_to_bucket_location = {} # Map each parameter to its corresponding bucket/place (start, end, bucket_idx).
        self.bucket_size = bucket_size
        self.bucket_sizes = None # Actual sizes of each bucket.
        self.grad_data_list = [] # List of tensors to store gradients, one tensor per bucket.
        self.grad_type = grad_type
        # Divide gradients into buckets based on the provided bucket size.
        self._initialize_buckets()
    
    
    def _initialize_buckets(self) -> None:
        """
        Divides model parameters into buckets for gradient synchronization based on the bucket size.
        """
        cur_bucket_size = 0 
        cur_bucket_idx = 0
        
        # Assign parameters to buckets.
        for param in self.params:
            if not param.requires_grad:
                continue

            # If the parameter cannot fit in the current (non-empty) bucket, start a new bucket.
            if cur_bucket_size > 0 and cur_bucket_size + param.numel() > self.bucket_size:
                cur_bucket_idx += 1
                cur_bucket_size = 0

            self.params_to_bucket_location[param] = (cur_bucket_size, cur_bucket_size + param.numel(), cur_bucket_idx)
            cur_bucket_size += param.numel()

        # Gather information about the bucket sizes and the parameters in each bucket
        bucket_sizes = [0] * (cur_bucket_idx + 1)
        buckets_to_params = [[] for _ in range(cur_bucket_idx + 1)]
        for param, (_, end, idx) in self.params_to_bucket_location.items():
            bucket_sizes[idx] = max(bucket_sizes[idx], end)
            buckets_to_params[idx].append(param)
        
        # Create tensors for storing gradients and initialize Bucket objects.
        for i in range(len(bucket_sizes)):
            self.grad_data_list.append(torch.zeros(bucket_sizes[i], dtype=self.grad_type, device=self.device))
            self.buckets.append(Bucket(buckets_to_params[i], self.grad_data_list[i], self.process_group))
        
        # Create gradient views for each parameter.
        for param in self.params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.params_to_bucket_location[param]
            # param.main_grad is used for gradient calculation
            param.main_grad = self._get_view_from_tensor(self.grad_data_list[bucket_id], param.shape, data_start_index, data_end_index)
            
    def _get_view_from_tensor(self, tensor: torch.Tensor, shape: torch.Size, start: int, end: int) -> torch.Tensor:
        """
        Create a view of the given tensor with the specified shape from start to end indices.
        """
        return tensor[start:end].view(shape)
    
    def reset(self) -> None:
        """
        Reset all buckets by clearing the gradients and internal states.
        """
        for bucket in self.buckets:
            bucket.reset()
    
    def wait(self) -> None:
        """
        Wait for all buckets to complete their gradient synchronization
        """
        for bucket in self.buckets:
            bucket.wait()
    
    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        """
        Mark a parameter's gradient as ready for synchronization.
        """
        bucket_idx = self.params_to_bucket_location[param][2]
        self.buckets[bucket_idx].mark_param_as_ready(param)


