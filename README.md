# Tiny-Megatron

**Tiny-Megatron** is a minimalistic, educational re-implementation of the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) library for distributed deep learning. This project provides clean, understandable implementations of various parallelism strategies used in large-scale language model training.

## 🚀 Features

### Multiple Parallelism Strategies
- **Tensor Parallelism (TP)**: Split individual layers across multiple devices
- **Data Parallelism (DP)**: Replicate model across devices, shard data batches  
- **Pipeline Parallelism (PP)**: Distribute model layers into sequential stages
- **3D Hybrid Parallelism**: Combine TP + DP + PP for maximum scalability

### Core Components
- **Custom Neural Network Modules**: Optimized implementations of Linear, Embedding, LayerNorm
- **Automatic Kernel Selection**: Runtime auto-tuner for optimal performance
- **Flexible Parallel Context**: Easy configuration of multi-dimensional parallelism
- **Wrapper-Based Design**: Non-intrusive parallelization of existing models

### Educational Focus
- **Clean, Readable Code**: Well-documented implementations for learning
- **Modular Architecture**: Each parallelism strategy is independently implemented
- **Complete Examples**: Full training scripts demonstrating each approach

## 📁 Project Structure

```
Tiny-Megatron/
├── tiny_megatron/core/             # 🏗️ Core Library
│   ├── dist/                       # Distributed Parallelism
│   │   ├── tp/                     # • Tensor Parallelism (TP)
│   │   ├── dp/                     # • Data Parallelism (DP)
│   │   ├── pp/                     # • Pipeline Parallelism (PP)
│   │   ├── hybrid/                 # • 3D Hybrid Parallelism
│   │   └── utils/                  # • Communication utilities
│   ├── module/                     # Custom NN Modules
│   │   ├── linear.py               # • Optimized Linear layers
│   │   ├── embedding.py            # • Embedding layers  
│   │   ├── normalization.py        # • LayerNorm implementation
│   │   └── ops/                    # • Low-level operations
│   └── autotuner/                  # Performance Optimization
│       └── runtime_tuner.py        # • Automatic kernel selection
│
├── example/                        # 🚀 Training Examples
│   ├── model.py                    # • GPT-2 model implementation
│   ├── tp/train.py                 # • Tensor parallelism demo
│   ├── dp/train.py                 # • Data parallelism demo  
│   ├── pp/train.py                 # • Pipeline parallelism demo
│   └── hybrid/train.py             # • 3D hybrid parallelism demo
```

### 🎯 Key Components

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| **Distributed Parallelism** | Core parallel strategies | `dist/{tp,dp,pp,hybrid}/` |
| **Custom Modules** | Optimized NN building blocks | `module/{linear,embedding}.py` |
| **ParallelContext** | Multi-dimensional coordination | `dist/utils/comm.py` |
| **Auto-tuner** | Performance optimization | `autotuner/runtime_tuner.py` |
| **Examples** | Complete training demos | `example/{tp,dp,pp,hybrid}/` |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- NCCL for multi-GPU communication

### Setup
```bash
git clone https://github.com/liangyuwang/Tiny-Megatron.git
cd Tiny-Megatron
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
```

## 🎯 Quick Start

### 1. Tensor Parallelism (2 GPUs)
```bash
# Split model layers across 2 GPUs
torchrun --nproc_per_node=2 example/tp/train.py
```

### 2. Data Parallelism (2 GPUs)  
```bash
# Replicate model, distribute data batches
torchrun --nproc_per_node=2 example/dp/train.py
```

### 3. Pipeline Parallelism (2 GPUs)
```bash
# Split model into sequential stages
torchrun --nproc_per_node=2 example/pp/train.py
```

### 4. 3D Hybrid Parallelism (8 GPUs)
```bash
# Combine all parallelism strategies: PP=2 x TP=2 x DP=2
torchrun --nproc_per_node=8 example/hybrid/train.py
```

## 💡 Usage Examples

### Basic Tensor Parallelism
```python
import torch
from tiny_megatron.core import ParallelContext, apply_tensor_parallel
from example.model import GPT2Model, GPTConfig

# Initialize distributed environment
# ... (distribution setup code)

# Create model and parallel context
config = GPTConfig()
model = GPT2Model(config).cuda()

# Configure parallelism
parallel_config = {"tp": 2}  # Use 2 GPUs for tensor parallelism
context = ParallelContext(parallel_config)

# Apply tensor parallelism
model = TPWrapper(
        model, 
        parallel_context=parallel_context,
        column_linear_names=["c_attn", "c_fc"],
        row_linear_names=["c_proj"]
    )  # TPWrapper automatically moves parameters to rank's GPU

# Train normally
optimizer = torch.optim.AdamW(model.parameters())
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### 3D Hybrid Parallelism
```python
from tiny_megatron.core import ParallelContext, apply_hybrid_parallel

# Configure 3D parallelism for 8 GPUs
parallel_config = {
    "pp": 2,  # 2 pipeline stages
    "tp": 2,  # 2-way tensor parallelism  
    "dp": 2   # 2-way data parallelism
}

context = ParallelContext(parallel_config)

# Apply 3D hybrid parallelism
column_linear_names = ["c_attn", "c_fc"] if parallel_config.get("tp", 1) > 1 else None
row_linear_names = ["c_proj"] if parallel_config.get("tp", 1) > 1 else None
block_names = ["transformer.h"] if parallel_config.get("pp", 1) > 1 else None
model = apply_hybrid_parallel(
    model=model,
    parallel_context=parallel_context,
    column_linear_names=column_linear_names,
    row_linear_names=row_linear_names,
    block_names=block_names,
)
```

## 🏗️ Architecture Overview

### Parallelism Strategies

#### Tensor Parallelism (TP)
- **Column Parallel**: Split weight matrices column-wise (e.g., attention projections)
- **Row Parallel**: Split weight matrices row-wise (e.g., MLP layers)
- **Communication**: All-gather for activations, all-reduce for gradients

#### Data Parallelism (DP)  
- **Model Replication**: Same model on each device
- **Data Sharding**: Different data batches per device
- **Gradient Synchronization**: All-reduce after backward pass

#### Pipeline Parallelism (PP)
- **Layer Distribution**: Different model layers on different devices
- **Sequential Execution**: Forward/backward through pipeline stages
- **Point-to-Point Communication**: Send/recv activations between stages

#### 3D Hybrid Parallelism
- **Nested Structure**: PP (outer) → TP (middle) → DP (inner)
- **Flexible Configuration**: Support arbitrary combinations
- **Optimal Scaling**: Maximize hardware utilization

### Key Components

#### ParallelContext
Central coordination for multi-dimensional parallelism:
```python
context = ParallelContext({
    "tp": tensor_parallel_size,
    "dp": data_parallel_size,  
    "pp": pipeline_parallel_size
})
```

#### Custom Modules
Optimized implementations with built-in parallelism support:
- `Linear`: Matrix multiplication with automatic kernel selection
- `Embedding`: Token/position embeddings
- `LayerNorm`: Layer normalization

#### Runtime Auto-Tuner
Automatic selection of optimal kernels:
```python
tuner = RuntimeAutoTuner(
    warmup_iterations=10,
    measure_iterations=100
)
```

## 🔧 Configuration

### Environment Variables
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=8
export LOCAL_RANK=0
```

### Parallel Configuration
```python
parallel_config = {
    "tp": 2,    # Tensor parallel size
    "dp": 2,    # Data parallel size  
    "pp": 2,    # Pipeline parallel size
}
```

## 📚 Examples

Each parallelism strategy includes complete training examples:

- **`example/tp/train.py`**: Tensor parallelism with GPT-2
- **`example/dp/train.py`**: Data parallelism training
- **`example/pp/train.py`**: Pipeline parallelism implementation  
- **`example/hybrid/train.py`**: 3D hybrid parallelism demo

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [Tiny-DeepSpeed](https://github.com/liangyuwang/Tiny-DeepSpeed): Minimalistic DeepSpeed re-implementation
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM): Original Megatron library
- [FairScale](https://github.com/facebookresearch/fairscale): Facebook's model parallelism library

## 📖 Citation

If you use Tiny-Megatron in your research, please cite:

```bibtex
@misc{tiny-megatron,
    title={Tiny-Megatron: A Minimalistic Re-implementation of Megatron-LM},
    author={Liangyu Wang},
    year={2024},
    url={https://github.com/liangyuwang/Tiny-Megatron}
}
```
