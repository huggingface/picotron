import os
import torch
from torch import distributed as dist
from typing import List

from .. import process_group_manager as pgm

STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"

class ContextCommunicate:
    def __init__(self, msg: str = ""):
        global STEP
        global VERBOSE
        self._pending_operations: List[dist.P2POp] = []
        self._active_requests = None
        self.rank = pgm.process_group_manager.cp_rank
        self.world_size = pgm.process_group_manager.cp_world_size
        self.send_rank = pgm.process_group_manager.cp_send_rank
        self.recv_rank = pgm.process_group_manager.cp_recv_rank
        if VERBOSE: print(f"RingComm ({msg}) | initialized | RANK:{self.rank} | "f"WORLD_SIZE:{self.world_size} | SEND_RANK:{self.send_rank} | "f"RECV_RANK:{self.recv_rank}", flush=True)

    def send_recv(self, tensor_to_send, recv_tensor=None):
        if recv_tensor is None:
            result_tensor = torch.zeros_like(tensor_to_send)
        else:
            result_tensor = recv_tensor

        send_operation = dist.P2POp(dist.isend, tensor_to_send, self.send_rank, group=pgm.process_group_manager.cp_group)
        recv_operation = dist.P2POp(dist.irecv, result_tensor, self.recv_rank, group=pgm.process_group_manager.cp_group)
        
        self._pending_operations.extend([send_operation, recv_operation])

        if VERBOSE:
            print(f"RingComm | send_recv | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:sending | TO:{self.send_rank} | TENSOR:{tensor_to_send}", flush=True)
            print(f"RingComm | send_recv | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:receiving | FROM:{self.recv_rank} | TENSOR:{result_tensor}", flush=True)
        return result_tensor

    def commit(self):
        if self._active_requests is not None: raise RuntimeError("Commit called twice")
        self._active_requests = dist.batch_isend_irecv(self._pending_operations)
        if VERBOSE: print(f"RingComm | commit | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:committed | NUM_OPS:{len(self._pending_operations) // 2}", flush=True)

    def wait(self):
        if self._active_requests is None: raise RuntimeError("Wait called before commit")
        for i, request in enumerate(self._active_requests):
            request.wait()
            if VERBOSE:
                operation_type = "send" if i % 2 == 0 else "receive"
                peer_rank = self.send_rank if operation_type == "send" else self.recv_rank
                print(f"RingComm | wait | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:completed_{operation_type} | "f"{'FROM' if operation_type == 'receive' else 'TO'}:{peer_rank}", flush=True)
        torch.cuda.synchronize()
        self._active_requests = None
        self._pending_operations = []
        if VERBOSE: print(f"RingComm | wait | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:all_operations_completed", flush=True)