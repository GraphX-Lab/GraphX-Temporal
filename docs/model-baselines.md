# TGX方法实现与基准方法支持

本文档介绍GraphX-Temporal核心框架（tgx目录）中的方法实现、配置以及进行下游任务。

## 模型

最基本的实现用于动态图的模型步骤如下：
1. 继承`tgx.models.temporal_model.TemporalModel`类
2. 自定义`__init__`方法进行初始化
3. 实现`forward(self, batch: torch_geometric.data.Data) -> ModelOutputs`方法主体用于特征编码
4. `forward`方法在字典（`ModelOutputs`）中返回特征嵌入。


### 模型特征编码

在本项目中，将下游任务的预测和模型推理时的特征编码进行了解耦：
- 对于“模型”部分，主要基于给定的动态图，输出节点或者边的特征。
- 下游任务的预测下放到了`Evaluator`类中，且已经提供了默认的链接预测和分类相关实现。

因此，在模型实现阶段，只需要进行到特征编码即可。

例如：
```python
def __init__(
    self,
    hidden_dim: int = 128,  # 任何模型自定义的参数
    **kwargs,   # 其他用于父类初始化的参数
):
    super().__init__(**kwargs)  # 父类初始化
    self.hidden_dim = hidden_dim
    ...

def forward(self, batch: Data) -> ModelOutputs:
    x = ...
    edge_embeddings = ...
    return {
            "node_embeddings": x,
            "edge_embeddings": edge_embeddings,
        }
```

`forward`输入的`batch`中主要包含如下字段。
- 节点数据：
  - `x`：节点特征
- 消息传递边数据：
  - `edge_index`：用于消息传递的链接
  - `t`：链接时间戳
  - `edge_attr`：链接特征
- 监督任务边数据（可能包含负样本）：
  - `edge_label_index`：用于下游分类的链接
  - `edge_label`：链接标签
  - `edge_label_t`：链接时间戳
  - `edge_label_attr`：链接特征

输出中：
- `edge_embeddings`和`node_embeddings`字段二者包含一个即可，但是建议都进行实现。
- `edge_embeddings`如果不存在，`Evaluator`中会默认基于`node_embeddings`使用两端节点特征之和。
- 可以包含其他任意字段。

`TemporalModel`继承自`lightning.pytorch.LightningModule`，因此可以通过实现其他方法进行细粒度的过程控制。
返回的编码特征会被后续的`Evaluator`类处理，用于下游预测。

### 下游任务预测

在父类的模型中接受一个`Evaluator`类实例，一般可以直接通过配置文件初始化。
一个yaml配置文件案例（`tgx/baselines/simple/mlp-mooc-lc.yaml`）参考如下：
```yaml
# Lightning CLI configuration for MOOC dataset
fit:
  model:
    class_path: tgx.baselines.simple.arch.MLPModel  # 自己实现的模型类路径
    init_args:
      # 各种初始化参数
      input_dim: 172            
      hidden_dim: 128
      output_dim: 128
      dropout: 0.1
      ## 配置 Link classification 下游任务评估（TemporalModel类的参数）
      evaluator:    
        class_path: tgx.models.LinkClassificationEvaluator
        init_args:
          input_dim: ${fit.model.init_args.output_dim}  # 支持 ${} 语法静态替换配置文件中的项，input_dim 默认和模型返回的特征维度保持一致即可

  # Data parameters (data class parameters) - LC DataModule
  data:
    class_path: tgx.data.LCDataModule   # Link classification 数据模块
    init_args:
      dataset: mooc
      data_dir: ./datasets

  # Training parameters
  trainer:
    accelerator: gpu  # auto, cpu, gpu, tpu, mps
    devices: 1  # 默认使用一张 gpu
    max_epochs: 100
```

在完成了配置文件的编辑后，即可在命令行中开始训练和评估：
```sh
python -m tgx.cli.train --config tgx/baselines/simple/mlp-mooc-lc.yaml fit
```

## 示例

更多详细的实现请参考本项目 `tag/baselines` 目录。
譬如，在cl_ond目录下实现了一个预训练+微调的模型。
