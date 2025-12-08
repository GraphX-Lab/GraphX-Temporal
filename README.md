# TGX: Temporal Graph Learning Framework

[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)

A comprehensive, modular framework for temporal graph learning built on PyTorch Lightning, providing unified interfaces for data handling, model development, and downstream task evaluation.

## 🌟 Key Features

- **🔧 Unified Architecture**: Modular design with abstract base classes for easy extension
- **⚡ PyTorch Lightning Integration**: Simplified training loops, automatic GPU support, and experiment management
- **📊 Comprehensive Evaluation System**: Built-in evaluators for link prediction, link classification, and node classification
- **🏗️ Flexible Data Pipeline**: Support for benchtemp format with transductive and inductive settings
- **🎯 Baseline Models**: Includes SimpleGNN, TGAT, and CL-OND implementations
- **📈 Extensive Metrics**: AUC-ROC, F1, Precision, Recall, MAP, MRR, and custom metrics support
- **🔌 Pluggable Components**: Easy to add new models, datasets, and evaluation metrics

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Framework Architecture](#-framework-architecture)
- [Supported Tasks](#-supported-tasks)
- [Available Models](#-available-models)
- [Dataset Format](#-dataset-format)
- [Configuration System](#-configuration-system)
- [Examples](#-examples)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

## 🚀 Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/TGL-X/GraphX-Temporal.git
cd GraphX-Temporal

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# For cuda 12.6 + conda 
bash setup_env_conda_cu126.sh

# # Install in development mode
# pip install -e .

# # Install dependencies
# pip install -r requirements.txt
```

### Requirements

- Python 3.12+
- PyTorch >= 2.7.0
- PyTorch Lightning
- PyTorch Geometric
- Additional dependencies in `requirements.txt`

## 🎯 Quick Start

### 1. Train a Simple Model

```bash
# Train SimpleGNN for link prediction on MOOC dataset
python -m tgx.cli.train --config tgx/baselines/simple/gcn-mooc-lp.yaml fit 

# Train CL-OND for link classification
python -m tgx.cli.train --config tgx/baselines/cl_ond/mooc-lc.yaml fit
```

### 2. Custom Training with Parameter Override

```bash
python -m tgx.cli.train \
  --config tgx/baselines/simple/gcn-mooc-lp.yaml \
  fit \
  --model.init_args.hidden_dim 256 \
  --trainer.max_epochs 50 \
  --trainer.devices 1
```

### 3. Using Custom Models

First, implement your model:

```python
# my_model.py
import torch
from tgx.models import TemporalModel

class MyTemporalModel(TemporalModel):
  def __init__(self, hidden_dim:int, **kwargs):
    super().__init__(**kwargs)
    self.hidden_dim = hidden_dim
    self.lin = torch.nn.Linear(172, hidden_dim)
  
  def forward(self, batch):
    x = self.lin(batch.x)
    src_x = x[batch.edge_index[0]]
    dst_x = x[batch.edge_index[1]]
    edge_x = (src_x + dst_x).relu()
    return {'node_embeddings': x, 'edge_embeddings': edge_x}

```

Then, recommend writing a simple YAML configuration:

```yaml
fit:
  model:
    class_path: my_model.MyTemporalModel
    init_args:
      hidden_dim: 128
      evaluator:
        class_path: tgx.models.LinkPredictionEvaluator  # Use the Link Prediction Evaluator
        init_args:
          input_dim: ${fit.model.init_args.hidden_dim} 

  data:
    class_path: tgx.data.LPDataModule # Use the Link Prediction DataModule
    init_args:
      dataset: mooc
      mode: inductive

  trainer:  # Train up to 100 epochs on a single GPU
    accelerator: gpu
    devices: 1
    max_epochs: 100
```

## 🏗️ Framework Architecture

```
TGX Framework (v0.1.0 alpha)
├── 📊 Data Layer
│   ├── tgx/data/                    # Core data structures
│   │   ├── temporal_data.py         # Data class for temporal graphs
│   │   └── datamodules/             # Lightning data modules
│   │       ├── link_pred.py         # Link prediction data module
│   │       └── link_cls.py          # Link classification data module
│   └── tgx/dataset/                 # Dataset loaders
│       ├── lp.py                    # Link prediction datasets
│       └── lc.py                    # Link classification datasets
│
├── 🤖 Model Layer
│   ├── tgx/models/                  # Model architecture base
│   │   ├── temporal_model.py        # TemporalModel abstract base class
│   │   ├── temporal_evaluator.py    # Evaluation system base
│   │   └── evaluators/              # Task-specific evaluators
│   │       ├── link_pred.py         # Link prediction evaluator
│   │       ├── link_cls.py          # Link classification evaluator
│   │       └── node_cls.py          # Node classification evaluator
│   └── tgx/baselines/               # Baseline model implementations
│       ├── simple/                  # Static GNN baselines
│       ├── tgat/                    # TGAT integration
│       └── cl_ond/                  # CL-OND model
│
├── 🔧 Utility Layer
│   ├── tgx/sampling/                # Sampling strategies
│   ├── tgx/utils/                   # Type definitions and utilities
│   └── tgx/cli/                     # Command-line interface
│
└── ⚡ Training Layer
    └── PyTorch Lightning Integration
```

## 📊 Supported Tasks

### 1. Link Prediction (LP)
- **Task**: Predict future connections in temporal graphs
- **Evaluator**: `LinkPredictionEvaluator`
- **Metrics**: AUC-ROC, Average Precision, F1, Precision, Recall
- **Settings**: Transductive and Inductive

### 2. Link Classification (LC)
- **Task**: Classify existing edges into categories
- **Evaluator**: `LinkClassificationEvaluator`
- **Metrics**: Accuracy, AUC-ROC, F1 (binary/multi-class)
- **Features**: Label smoothing support

### 3. Node Classification (NC)
- **Task**: Classify nodes based on temporal patterns
- **Status**: Interface defined (extensible)

## 🤖 Available Models

### SimpleGNN Baselines
```python
# Static GNN + MLP (ignores temporal structure)
- GCN: Graph Convolutional Network
- GAT: Graph Attention Network
- GraphSAGE: Graph Sample and Aggregate
```

### TGAT (Temporal Graph Attention Transformer)
```python
# Time-aware attention mechanism
- Temporal encoding integration
- Multi-head attention
- Transformer-style processing
```

### CL-OND (Community-Level Online Node Discovery)
```python
# Contrastive learning for dynamic community detection
- Pretraining + downstream tasks
- Inductive evaluation support
- Community-aware representations
```

## 📁 Dataset Format

### Benchtemp Format
The framework uses the **benchtemp** format for temporal graph datasets:

```
datasets/{dataset_name}/
├── ml_{dataset}.csv              # Edge information (u,i,ts,label,idx)
├── ml_{dataset}.npy              # Edge features [n_edges, edge_feat_dim]
└── ml_{dataset}_node.npy         # Node features [n_nodes, node_feat_dim]
```

### Supported Datasets
- **MOOC**: Online course interactions
- **Reddit**: Social media discussions
- **Wikipedia**: Collaborative editing
- **LastFM**: Music listening patterns
- [Any datasets in BenchTemp format](https://zenodo.org/records/8267846)

### Chronological Splitting
- **Training**: 0-70% of timestamps
- **Validation**: 70-85% of timestamps
- **Testing**: 85-100% of timestamps

## ⚙️ Configuration System

### YAML-Based Configuration

```yaml
# Example: tgx/baselines/simple/gcn-mooc-lp.yaml
fit:
  model:
    class_path: tgx.baselines.simple.arch.LPGNN
    init_args:
      task: link_prediction
      hidden_dim: 128
      gnn_type: gcn
      evaluator:
        class_path: tgx.models.LinkPredictionEvaluator
        init_args:
          input_dim: ${fit.model.init_args.hidden_dim}

  data:
    class_path: tgx.data.LPDataModule
    init_args:
      dataset: mooc
      batch_size: 200
      mode: transductive

  trainer:
    accelerator: gpu
    devices: 1
    max_epochs: 100
```

### Configuration Override
```bash
# Override specific parameters
python -m tgx.cli.train \
  --config configs/model.yaml \
  --model.init_args.hidden_dim 256 \
  --trainer.max_epochs 50 \
  --data.init_args.batch_size 64
```

## 💡 Examples

### Training Examples

#### Link Prediction with Simple GNN
```bash
# GCN baseline
python -m tgx.cli.train --config tgx/baselines/simple/gcn-mooc-lp.yaml

# Custom parameters
python -m tgx.cli.train \
  --config tgx/baselines/simple/gcn-mooc-lp.yaml \
  --model.init_args.gnn_type gat \
  --model.init_args.hidden_dim 256
```

### Custom Model Development

#### Create Custom Model
```python
from tgx.models import TemporalModel
from tgx.utils.typing import ModelOutputs
from torch_geometric.data import Data

class MyTemporalModel(TemporalModel):
    def __init__(self, hidden_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        # Initialize your model components here

    def forward(self, batch: Data) -> ModelOutputs:
        # Implement forward pass
        return {
            "node_embeddings": node_embeddings,
            "edge_embeddings": edge_embeddings
        }

    def compute_loss(self, outputs: ModelOutputs, targets) -> torch.Tensor:
        # Implement custom loss
        return self.evaluator.compute_loss(outputs, targets)
```

#### Create Custom Evaluator
```python
from tgx.models.temporal_evaluator import TemporalEvaluator
from torchmetrics import Metric

class CustomMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        # Custom metric update logic
        pass

    def compute(self):
        # Custom metric computation
            pass

class CustomEvaluator(TemporalEvaluator):
    def __init__(self, **kwargs):
        super().__init__(metric_names=["custom_metric"], **kwargs)

    def compute_loss(self, outputs, targets):
        # Custom loss computation
        pass
```

## 📚 Documentation

- **[Architecture Guide](docs/modules.md)**: Detailed framework architecture
- **[Evaluation System](docs/downstream.md)**: Evaluator classes and metrics
- **[Dataset Documentation](docs/dataset.md)**: Data format and loading
- **[Development Guide](docs/model-baselines.md)**: Development guidelines


## 🛠️ Development

### Adding New Models

1. **Create Model Class**:
```python
# tgx/baselines/my_model/model.py
from tgx.models import TemporalModel

class MyModel(TemporalModel):
    def forward(self, batch):
        # Implementation
        pass
```

2. **Create Configuration**:
```yaml
# tgx/baselines/my_model/config.yaml
fit:
  model:
    class_path: tgx.baselines.my_model.model.MyModel
    init_args:
      hidden_dim: 128
```

3. **Train Model**:
```bash
python -m tgx.cli.train --config tgx/baselines/my_model/config.yaml
```

### Adding New Datasets

1. **Prepare Data**: Convert to benchtemp format
2. **Update Dataset Loader**: Modify `tgx/dataset/` if needed
3. **Create Configuration**: Set dataset parameters

### Testing

```bash
# Run tests
python -m pytest test/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [BenchTemp](https://github.com/johnnyhuangcs/benchtemp)
- [DyGLib](https://github.com/yule-BUAA/DyGLib)
- [PyTorch Lightning](https://lightning.ai/) for the training framework
- [PyTorch Geometric](https://pyg.org/) for graph neural network utilities
- Original authors of the temporal graph learning models in our experimental_codes


## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/TGL-X/GraphX-Temporal/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TGL-X/GraphX-Temporal/discussions)
- **Email**: [Project Email]

## 📊 Citation

If you use TGX in your research, please cite:

```bibtex
```

---

**Note**: This is an alpha release (v0.1.0). APIs may change as we improve the framework. We welcome feedback and contributions!