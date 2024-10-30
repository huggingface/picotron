
"""
python create_config.py --out_dir tmp --exp_name test_2_node --tp 2 --cp 2 --pp 2 --dp 2 --model_name HuggingFaceTB/SmolLM-360M-Instruct --num_attention_heads 16 --num_key_value_heads 4 --grad_acc 1 --mbs 32 --seq_len 4096 --use_wandb
"""
from copy import deepcopy
from transformers import AutoConfig
import os
import shutil
import argparse
import json
from typing import Optional

def create_single_config(
    out_dir: str,
    tp: int,
    cp: int,
    pp: int,
    dp: int,
    model_name: str,
    num_hidden_layers: Optional[int],
    num_attention_heads: Optional[int],
    num_key_value_heads: Optional[int],
    grad_acc: int,
    mbs: int,
    seq_len: int,
    exp_name: str,
    use_wandb: bool = False
):
    run_path = os.path.join(out_dir, exp_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open("template/base_config.json", "r") as f:
        base_config = json.load(f)

    config_content = deepcopy(base_config)
    config_content["training"]["seq_length"] = seq_len
    config_content["checkpoint"]["save_dir"] = run_path
    
    config_content["model"]["name"] = model_name
    
    tmp_model_config = AutoConfig.from_pretrained(model_name) 
    config_content["model"]["num_hidden_layers"] = tmp_model_config.num_hidden_layers if num_hidden_layers is None else num_hidden_layers
    config_content["model"]["num_attention_heads"] = tmp_model_config.num_attention_heads if num_attention_heads is None else num_attention_heads
    config_content["model"]["num_key_value_heads"] = tmp_model_config.num_key_value_heads if num_key_value_heads is None else num_key_value_heads
    del tmp_model_config

    config_content['distributed']['tp_size'] = tp
    config_content['distributed']['cp_size'] = cp
    config_content['distributed']['pp_size'] = pp
    config_content['distributed']['dp_size'] = dp

    config_content['logging']['use_wandb'] = use_wandb
    config_content['logging']['run_name'] = exp_name

    gbs = dp * mbs * grad_acc
    gbs_token = gbs * seq_len
    print(f"Gbs_token: {gbs_token:,}, Gbs: {gbs}, dp: {dp}, seq_len: {seq_len}, grad_acc: {grad_acc}, mbs: {mbs}")
    
    config_content['training']['gradient_accumulation_steps'] = grad_acc
    config_content['training']['micro_batch_size'] = mbs    
    
    if os.path.exists(run_path):
        shutil.rmtree(run_path)
    
    os.makedirs(run_path)
    with open(os.path.join(run_path, "config.json"), "w") as new_config:
        json.dump(config_content, new_config, indent=4)
    del config_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, help="Output directory to store the configs", default="tmp")
    parser.add_argument("--tp", type=int, help="number of tensor parallelism", default=1)
    parser.add_argument("--cp", type=int, help="number of context parallelism", default=1)
    parser.add_argument("--pp", type=int, help="number of pipeline parallelism", default=1)
    parser.add_argument("--dp", type=int, help="number of data parallelism", default=1)
    parser.add_argument("--model_name", type=str, help="Model name to create configs for", default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--num_hidden_layers", type=int, help="Number of hidden layers", default=None)
    parser.add_argument("--num_attention_heads", type=int, help="Number of attention heads", default=None)
    parser.add_argument("--num_key_value_heads", type=int, help="Number of key value heads", default=None)
    parser.add_argument("--grad_acc", type=int, help="grad accumulation", default=1)
    parser.add_argument("--mbs", type=int, help="micro batch size", default=1)
    parser.add_argument("--seq_len", type=int, help="Sequence length", default=1024)
    parser.add_argument("--exp_name", type=str, help="Experiment name", default="dummy_exp")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")

    args=parser.parse_args()
    
    create_single_config(
        out_dir=args.out_dir,
        tp=args.tp,
        cp=args.cp,
        dp=args.dp,
        pp=args.pp,
        model_name=args.model_name,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        grad_acc=args.grad_acc,
        mbs=args.mbs,
        seq_len=args.seq_len,
        exp_name=args.exp_name,
        use_wandb=args.use_wandb,
    )    
