# Tiny-Megatron

**Tiny-Megatron** is a minimalistic, educational re-implementation of the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) library for distributed deep learning. This project provides clean, understandable implementations of various parallelism strategies used in large-scale language model training.

## 🚀 Features

### Multiple Parallelism Strategies
- **Tensor Parallelism (TP)**: Split individual layers across multiple devices
- **Data Parallelism (DP)**: Replicate model across devices, shard data batches  
- **2D Hybrid Parallelism**: Combine TP + DP for effective scalability

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
│   │   ├── hybrid/                 # • 2D Hybrid Parallelism (TP + DP)
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
│   └── hybrid/train.py             # • 2D hybrid parallelism demo
```

### 🎯 Key Components

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| **Distributed Parallelism** | Core parallel strategies | `dist/{tp,dp,hybrid}/` |
| **Custom Modules** | Optimized NN building blocks | `module/{linear,embedding}.py` |
| **ParallelContext** | Multi-dimensional coordination | `dist/utils/comm.py` |
| **Auto-tuner** | Performance optimization | `autotuner/runtime_tuner.py` |
| **Examples** | Complete training demos | `example/{tp,dp,hybrid}/` |

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

### 3. 2D Hybrid Parallelism (4 GPUs)
```bash
# Combine TP and DP: TP=2 x DP=2
torchrun --nproc_per_node=4 example/hybrid/train.py
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
tp_config = {
    "column_linear_patterns": ["attn.c_attn", "mlp.c_fc"],
    "row_linear_patterns": ["attn.c_proj", "mlp.c_proj"]
}
model = apply_tensor_parallel(
    model=model, 
    parallel_context=context,
    tp_config=tp_config
)

# Train normally
optimizer = torch.optim.AdamW(model.parameters())
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### 2D Hybrid Parallelism
```python
from tiny_megatron.core import ParallelContext, apply_hybrid_parallel

# Configure 2D parallelism for 4 GPUs
parallel_config = {
    "tp": 2,  # 2-way tensor parallelism  
    "dp": 2   # 2-way data parallelism
}

context = ParallelContext(parallel_config)

# Apply 2D hybrid parallelism
tp_config = {
    "column_linear_patterns": ["attn.c_attn", "mlp.c_fc"],
    "row_linear_patterns": ["attn.c_proj", "mlp.c_proj"]
}
model = apply_hybrid_parallel(
    model=model,
    parallel_context=context,
    tp_config=tp_config
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

#### 2D Hybrid Parallelism
- **Combined Strategy**: Tensor Parallelism (TP) + Data Parallelism (DP)
- **Flexible Configuration**: Support various TP and DP combinations
- **Efficient Scaling**: Optimal resource utilization for medium-scale training

### Key Components

#### ParallelContext
Central coordination for multi-dimensional parallelism:
```python
context = ParallelContext({
    "tp": tensor_parallel_size,
    "dp": data_parallel_size
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
export WORLD_SIZE=4
export LOCAL_RANK=0
```

### Parallel Configuration
```python
parallel_config = {
    "tp": 2,    # Tensor parallel size
    "dp": 2,    # Data parallel size
}
```

## 📚 Examples

Each parallelism strategy includes complete training examples:

- **`example/tp/train.py`**: Tensor parallelism with GPT-2
- **`example/dp/train.py`**: Data parallelism training
- **`example/hybrid/train.py`**: 2D hybrid parallelism demo

## 🛣️ Roadmap

### Currently Supported
- ✅ **Tensor Parallelism (TP)**: Column and row parallelism for linear layers
- ✅ **Data Parallelism (DP)**: Standard gradient synchronization
- ✅ **2D Hybrid Parallelism**: TP + DP combinations

### Future Plans
To maintain code simplicity and readability, we are currently focusing on TP and DP implementations. Future releases will include:

- 🔄 **Pipeline Parallelism (PP)**: Layer-wise model partitioning
- 🔄 **ZeRO Optimizer States**: Memory-efficient optimizer state sharding
- 🔄 **Expert Parallelism (EP)**: Mixture-of-experts model scaling
- 🔄 **Sequence Parallelism (SP)**: Sequence dimension parallelism for long contexts
- 🔄 **5D Hybrid Parallelism**: TP + EP + SP + DP (ZeRO) + PP combinations

These advanced strategies will be added incrementally while maintaining the educational and minimalistic nature of the codebase.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM): Original Megatron library
- [Tiny-FSDP](https://github.com/liangyuwang/Tiny-FSDP): Minimalistic PyTorch FSDP re-implementation
- [Tiny-DeepSpeed](https://github.com/liangyuwang/Tiny-DeepSpeed): Minimalistic DeepSpeed re-implementation

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
