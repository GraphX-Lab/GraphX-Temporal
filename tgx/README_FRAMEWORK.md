# Temporal Graph Learning Framework

This document provides usage instructions for the PyTorch Lightning-based temporal graph learning framework.

## Framework Structure

```
tgx/
├── models/
│   ├── __init__.py
│   └── temporal_model.py     # Base TemporalModel class
├── baselines/gnn/
│   ├── __init__.py
│   ├── mooc.yaml            # Configuration file
│   └── arch/
│       ├── __init__.py
│       └── lpgnn.py          # LPGNN model implementation
└── cli/
    ├── __init__.py
    └── train.py             # Training CLI script
```

## Usage Examples

### 1. Training with Configuration File (Recommended)

```bash
# Activate conda environment
conda activate tgl-x

# Train using YAML configuration - all parameters including data are defined in the config
python -m tgx.cli.train --config tgx/baselines/gnn/mooc.yaml
```

### 2. Training with Command Line Arguments

```bash
# Model parameters can be set via CLI (data parameters are handled by DataModule through config)
python -m tgx.cli.train \
  --model.class_path tgx.baselines.gnn.LPGNN \
  --model.init_args.task link_prediction \
  --model.init_args.hidden_dim 256 \
  --model.init_args.gnn_type gat \
  --trainer.max_epochs 50
```

### 3. Mixed Approach (Config + CLI Overrides)

```bash
# Use config file but override specific model parameters
python -m tgx.cli.train \
  --config tgx/baselines/gnn/mooc.yaml \
  --model.init_args.hidden_dim 256 \
  --model.init_args.dropout 0.2
```

### 3. Using as a Python Module

```python
from tgx.models.temporal_model import TemporalModel
from tgx.baselines.gnn.arch.lpgnn import LPGNN
from tgx.cli.train import TemporalGraphDataModule

# Create model
model = LPGNN(
    input_dim=64,
    hidden_dim=128,
    gnn_type='gcn',
    task='link_prediction'
)

# Create data module
data_module = TemporalGraphDataModule(
    dataset='mooc',
    batch_size=32
)

# Train using PyTorch Lightning
import lightning.pytorch as pl
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, data_module)
```

## Model Architecture

### LPGNN (Link Prediction Graph Neural Network)

- **Backbone**: Standard GNN (GCN, GAT, GraphSAGE)
- **Predictor**: Multi-layer Perceptron (MLP)
- **Features**:
  - Node features or node embeddings
  - Optional time encoding as edge features
  - Ignores temporal structures (as required)

### Supported GNN Types

- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network
- **SAGE**: GraphSAGE

### Configuration Parameters

#### Model Parameters
- `input_dim`: Input feature dimension (default: 64)
- `hidden_dim`: Hidden dimension (default: 128)
- `gnn_type`: GNN type ('gcn', 'gat', 'sage')
- `num_layers`: Number of GNN layers (default: 2)
- `mlp_hidden_dims`: MLP hidden dimensions (default: [128, 64])
- `dropout`: Dropout rate (default: 0.1)
- `use_time_encoding`: Use timestamp encoding (default: True)
- `time_dim`: Time encoding dimension (default: 32)
- `task`: Task type ('link_prediction', 'node_classification')

#### Data Parameters
- `dataset`: Dataset name ('mooc', etc.)
- `data_dir`: Data directory path
- `split_ratio`: Train/val/test split ratio (default: [0.7, 0.15, 0.15])
- `batch_size`: Batch size (default: 32)
- `num_workers`: Number of data loader workers

#### Training Parameters
- `max_epochs`: Maximum number of epochs
- `learning_rate`: Learning rate (default: 0.001)
- `weight_decay`: Weight decay (default: 1e-5)

## Adding New Models

To add a new model, inherit from `TemporalModel`:

```python
from tgx.models.temporal_model import TemporalModel

class MyModel(TemporalModel):
    def __init__(self, **kwargs):
        super().__init__(task=kwargs.get('task', 'link_prediction'), **kwargs)
        # Your model initialization

    def forward(self, batch):
        # Forward pass implementation
        pass

    def compute_loss(self, predictions, targets):
        # Loss computation implementation
        pass
```

## Next Steps

1. **Data Loading**: Replace dummy data loading with actual temporal graph datasets
2. **Evaluation Metrics**: Add task-specific evaluation metrics (AUC, AP, etc.)
3. **More Models**: Add additional temporal graph models
4. **Hyperparameter Tuning**: Integrate with hyperparameter optimization tools
5. **Visualization**: Add training visualization and result analysis tools