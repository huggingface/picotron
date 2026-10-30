# picotron

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

秉承 [NanoGPT](https://github.com/karpathy/nanoGPT) 的极简设计哲学，我们打造了 Picotron：一个用于通过 [4D 并行](https://arxiv.org/abs/2407.21783)（数据并行 DP、张量并行 TP、流水线并行 PP、上下文并行 CP）预训练类 Llama 模型的极简、最具可魔改性（most-hackable）的代码库。它专为简洁直观与**教学**目的而设计，是深入理解与实验探索分布式大模型训练的绝佳工具。

![](assets/banière.png)
- **代码极简且可读性高**：`train.py`、`model.py` 以及核心并行实现 `[data|tensor|pipeline|context]_parallel.py` 单文件代码量均在 **300** 行以内。
- **性能持续演进**：虽然极致吞吐并非唯一目标，但性能仍在积极优化中。我们在 64 张 H100 GPU 上训练 LLaMA-2-7B 模型时测得 38% 的 MFU（模型算力利用率，Model Flops Utilization），在 8 张 H100 GPU 上训练 SmolLM-1.7B 模型时达到近 50% MFU。完整的性能基准测试将很快发布。
- **定位清晰**：相较于生产级分布式框架 [Nanotron](https://github.com/huggingface/nanotron/tree/main)，Picotron 主要侧重于教学与研究，旨在帮助开发者快速熟悉并掌握现代分布式大模型训练的各项核心底层技术。

# 视频教程 (Tutorial videos)

- 从零开始手把手构建 Picotron 分布式训练框架的系统教程：
    - [Picotron 教学视频（播放列表）](https://www.youtube.com/playlist?list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S) 🎬
    - [Picotron 教程配套代码库](https://github.com/huggingface/picotron_tutorial) 👷🏻‍♂️

# 安装指南 (Install)

```sh
pip install -e .
```

# 快速上手 (Quick start)

- 在[此处](https://huggingface.co/settings/tokens)获取 Hugging Face Token，以便从 HuggingFace Hub 下载模型配置与权重。

- **GPU 训练**
    ```sh
    # 创建默认保存在 tmp 目录下的 JSON 格式配置文件
    python create_config.py --out_dir tmp --exp_name llama-1B --dp 8 --model_name HuggingFaceTB/SmolLM-1.7B --num_hidden_layers 15  --grad_acc_steps 32 --mbs 4 --seq_len 1024 --hf_token <HF_TOKEN>

    # 本地单节点运行
    torchrun --nproc_per_node 8 train.py --config tmp/llama-1B/config.json

    # 3D 并行训练配置 (DP + TP + PP)
    python create_config.py --out_dir tmp --dp 4 --tp 2 --pp 2 --pp_engine 1f1b --exp_name llama-7B --model_name meta-llama/Llama-2-7b-hf  --grad_acc_steps 32 --mbs 4 --seq_len 1024 --hf_token <HF_TOKEN>

    # 提交 Slurm 集群任务
    python submit_slurm_jobs.py --inp_dir tmp/llama-7B --qos high --hf_token <HF_TOKEN>
    ```

- **CPU 训练**（速度较慢，主要用于功能调试与流程验证）
    ```sh
    # 在 CPU 上运行 3D 并行
    python create_config.py --out_dir tmp --exp_name llama-1B-cpu --dp 2 --tp 2 --pp 2 --pp_engine 1f1b --model_name HuggingFaceTB/SmolLM-1.7B --num_hidden_layers 5  --grad_acc_steps 2 --mbs 4 --seq_len 128 --hf_token <HF_TOKEN> --use_cpu

    # 本地运行
    torchrun --nproc_per_node 8 train.py --config tmp/llama-1B-cpu/config.json
    ```

# 学术引用 (Citation)

如果您在研究或项目中使用了 Picotron，请引用如下信息：

```bibtex
@misc{zhao2025picotron,
  author = {Haojun Zhao and Ferdinand Mom},
  title = {Picotron: Distributed training framework for education and research experimentation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/picotron}}
}
```

# 致谢 (Acknowledgements)

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [FairScale](https://github.com/facebookresearch/fairscale)
- [LitGPT](https://github.com/Lightning-AI/lit-gpt)

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月30日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
