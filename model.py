import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from src.parallel.context_parallel import ring_attention, update_rope
from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn.layers.rotary import apply_rotary_emb
from flash_attn.ops.triton.layer_norm import layer_norm_fn
import src.distributed.process_group_manager as pgm

def apply_rotary_pos_emb(x, cos, sin):
    #TODO: Maybe do class RotaryEmbedding(nn.Module) later
    batch_size, num_head, seq_length, head_dim = x.size()
    x1 = x[..., : head_dim // 2]  
    x2 = x[..., head_dim // 2 :]  
    rotate_half = torch.cat([-x2, x1], dim=-1)
    x = x * cos + rotate_half * sin
    return x

def get_cos_sin(seq_length, head_dim, base=500000.0):
    assert head_dim%2==0
    # Results on CUDA and CPU are different even with the same formula, To match transformers implementation. frequency should be computed on CPU
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to('cpu') / head_dim))
    dtype = torch.bfloat16 if os.getenv('DTYPE', 'bfloat16') == 'bfloat16' else torch.float32
    device = torch.device('cuda') if os.getenv('DEVICE', 'cuda') == 'cuda' else torch.device('cpu')
    position = torch.arange(seq_length).to(device).unsqueeze(1).float() # [seq_length, 1]
    # To match transformers implementation. m * theta should be computed on GPU
    theta = theta.to(device)
    return torch.cos(position.float()*theta.float()).to(dtype).repeat(1,2), torch.sin(position.float()*theta.float()).to(dtype).repeat(1,2) # [seq_length, head_dim], [seq_length, head_dim]

def flash_attention(q, k, v, causal = True):
    q = q.permute(0, 2, 1, 3) # [batch_size, seq_length, num_head , head_dim]
    k = k.permute(0, 2, 1, 3) # [batch_size, seq_length, num_head , head_dim]
    v = v.permute(0, 2, 1, 3) # [batch_size, seq_length, num_head , head_dim]
    return flash_attn_func(q, k, v, causal=causal)

class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)

    def forward(
        self, hidden_states, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):
        return layer_norm_fn(
            hidden_states,
            self.weight,
            None,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
            return_dropout_mask=return_dropout_mask,
        )

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = self.hidden_size//self.num_heads
        assert config.num_attention_heads % pgm.process_group_manager.tp_world_size == 0, "num_attention_heads should be divisible by tp world size"
        assert config.num_key_value_heads % pgm.process_group_manager.tp_world_size == 0, "num_key_value_heads should be divisible by  tp world size"
        self.num_local_heads = config.num_attention_heads // pgm.process_group_manager.tp_world_size # TP parallelism
        self.num_local_kv_heads = config.num_key_value_heads // pgm.process_group_manager.tp_world_size # TP parallelism
       
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads*self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_values*self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_values*self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.layer_idx = layer_idx
        
        ## TODO support mask
    
    def forward(self, x, cos, sin, attention_mask=None, position_ids=None):
        batch_size, seq_length, hidden_dim = x.size()
        q = self.q_proj(x) # [batch_size, seq_length, num_heads*head_dim]
        k = self.k_proj(x) # [batch_size, seq_length, num_key_values*head_dim]
        v = self.v_proj(x) # [batch_size, seq_length, num_key_values*head_dim]
        if os.getenv('FLASH_ATTEN', '1') != '1':
            q = q.view(batch_size, seq_length, self.num_local_heads, self.head_dim).transpose(1, 2)       # [batch_size, num_heads, seq_length, head_dim]
            k = k.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_key_values, seq_length, head_dim]
            v = v.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_key_values, seq_length, head_dim]
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        else:
            q = q.view(batch_size, seq_length, self.num_local_heads, self.head_dim)       # [batch_size, seq_length, num_heads, head_dim]
            k = k.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim)  # [batch_size, seq_length, num_key_values, head_dim]
            q = apply_rotary_emb(q,cos[:, :self.head_dim // 2], sin[:, :self.head_dim // 2],interleaved=False) # [batch_size, seq_length, num_heads, head_dim]
            k = apply_rotary_emb(k,cos[:, :self.head_dim // 2], sin[:, :self.head_dim // 2],interleaved=False) # [batch_size, seq_length, num_key_values, head_dim]
            q = q.transpose(1, 2)                                                                   # [batch_size, num_heads, seq_length, head_dim]
            k = k.transpose(1, 2)                                                                   # [batch_size, num_key_values, seq_length, head_dim]
            v = v.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1,2)   # [batch_size, num_key_values, seq_length, head_dim]
        
        k = k.repeat_interleave(self.num_local_heads // self.num_local_kv_heads, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        v = v.repeat_interleave(self.num_local_heads // self.num_local_kv_heads, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        
        causal = True if q.size(2) == k.size(2) else False # During decoding phase. The lenghth of q is usually 1. 
        
        if pgm.process_group_manager.cp_world_size > 1:
            # Ring attention for context parallelism
            sm_scale = 1.0 / (q.size(-1) ** 0.5)
            out = ring_attention(q, k, v, sm_scale, causal).transpose(1, 2) # [batch_size, seq_length, num_heads, head_dim]
        elif os.getenv('FLASH_ATTEN', '1') == '1':
            # flash attention, this is faster! 
            out = flash_attention(q, k, v, causal = causal) # [batch_size, seq_length, num_heads, head_dim] 
        else:
            # Pytorch scaled dot product attention
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal) # [batch_size, num_heads, seq_length, head_dim]
            out = out.transpose(1, 2) # [batch_size, seq_length, num_heads, head_dim]
        
        out = out.reshape(batch_size, seq_length, self.num_local_heads * self.head_dim) # [batch_size, seq_length, hidden_dim]
        out = self.out_proj(out) # [batch_size, seq_length, hidden_dim]
        return out


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        #TODO: dont do single line operations as it is harder to debug
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    # RMSNorm -> Attention -> Residual -> RMSNorm -> MLP -> Residual
    def __init__(self, config, layer_idx):
        super().__init__()
        RMSNorm = LlamaRMSNorm if os.getenv('FLASH_ATTEN', '1') != '1' else TritonRMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Attention(config, layer_idx = layer_idx)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx
        head_dim = config.hidden_size // config.num_attention_heads
        self.cos, self.sin = get_cos_sin(config.max_position_embeddings, head_dim=head_dim , base=config.rope_theta) # [max_position_embeddings, head_dim]

        # For context parallelism, we split the input. We need to get the correct cos and sin for each split
        self.cos, self.sin = update_rope(self.cos, self.sin)

    def forward(self, x, attention_mask = None, position_ids = None):
        #TODO: Use the default position_ids for RoPE during training. If we have time, work on generation
        cos, sin = self.cos, self.sin 
        x = x + self.attention(self.input_layernorm(x), cos, sin, attention_mask, position_ids) # Attention 
        x = x + self.mlp(self.post_attention_layernorm(x)) # MLP
        return x
    
class Llama(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # sanity check 
        assert config.hidden_size % config.num_attention_heads==0
        assert config.num_attention_heads % config.num_key_value_heads==0 
        
        # params
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads 
        self.head_dim = self.hidden_size//self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.model_config = config
        
        # modules
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config,layer_idx = i) for i in range(self.num_layers)])
        self.final_proj = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        RMSNorm = LlamaRMSNorm if os.getenv('FLASH_ATTEN', '1') != '1' else TritonRMSNorm
        self.final_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, input_ids, attention_mask=None, position_ids: torch.Tensor = None):
        x = self.embedding(input_ids)
        for layer in self.decoder_layers:
            x = layer(x)  # [batch_size, seq_length, hidden_dim]
        x = self.final_norm(x)
        logits = self.final_proj(x)
        
        return logits  # [batch_size, seq_length, vocab_size]
    
    # https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L289-L303
    # TODO: Need to check the formula.
    def get_flops(self, fwdbwd_per_iter, dt, num_params):
        L, H, T = self.num_layers , self.hidden_size, self.max_position_embeddings
        flops_per_fwdbwd = 6 * num_params * T + 12* L* H* T ** 2
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        return flops_achieved