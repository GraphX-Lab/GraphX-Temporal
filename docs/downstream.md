# TGX 下游任务评估系统

本文档详细介绍了 TGX (Temporal Graph Learning) 框架的评估系统，包括 Evaluator 类的设计、已支持的下游任务、自定义指标配置以及各评估方法的详细说明。

## 评估系统架构

### TemporalEvaluator 基类

`TemporalEvaluator` 是所有评估器的抽象基类，提供统一的评估接口：

```python
class TemporalEvaluator(nn.Module, ABC):
    def __init__(
        self,
        mode: Optional[str] = None,
        metric_names: Optional[List[str]] = None
    ):
        # 默认指标为 AUC-ROC 和 Average Precision
        self.metric_names = ["auroc", "ap"] if metric_names is None else metric_names

        # 创建指标集合
        self.metric_collection = MetricCollection(
            {n: self.init_metric(n) for n in self.metric_names}
        )
```

#### 核心抽象方法

1. **compute_loss(outputs, targets) -> torch.Tensor**
   - 计算任务特定的损失函数
   - 支持自定义损失计算逻辑

2. **compute_metrics(outputs, targets, node_info=None) -> Dict[str, float]**
   - 计算评估指标
   - 返回指标名称到值的映射

3. **log_metrics(metrics, stage, logger) -> None**
   - 将指标记录到 PyTorch Lightning 日志系统
   - 支持 train/val/test 不同阶段

4. **forward(model_output) -> torch.Tensor**
   - 任务特定的输出处理
   - 将原始模型输出转换为任务预测

#### 关键辅助方法

- **preprocess_inputs(batch, stage) -> Data**: 预处理输入批次
- **get_targets(batch, outputs, stage) -> torch.Tensor**: 提取目标标签
- **postprocess_outputs(model_outputs, batch, stage) -> ModelOutputs**: 后处理模型输出
- **create_edge_embeddings()**: 从节点嵌入创建边嵌入
- **init_metric()**: 初始化 torchmetrics 指标

### 支持的评估指标

框架内置了丰富的指标映射系统：

```python
METRIC_MAPPING = {
    "acc": BinaryAccuracy,
    "precision": BinaryPrecision,
    "recall": BinaryRecall,
    "f1": BinaryF1Score,
    "auroc": BinaryAUROC,
    "ap": BinaryAveragePrecision,
    "map": RetrievalMAP,
    "mrr": RetrievalMRR,
}
```

## DummyEvaluator

`DummyEvaluator` 是一个空评估器，在以下场景中非常有用：

1. **预训练阶段**: 当只关注嵌入学习而不需要具体任务评估时
2. **无评估要求**: 快速测试模型基础功能时
3. **自定义评估流程**: 完全绕过内置评估系统时

```python
class DummyEvaluator(TemporalEvaluator):
    def compute_loss(self, outputs, targets):
        """返回零损失"""
        return torch.tensor(0.0, device=self.device)

    def compute_metrics(self, outputs, targets):
        """返回空指标"""
        return {}

    def log_metrics(self, metrics, stage, logger):
        """无操作日志记录"""
        pass

    def forward(self, model_output):
        """直接返回模型输出"""
        return model_output
```

## 已实现的下游任务评估器

### 1. LinkPredictionEvaluator

链接预测评估器，支持 transductive 和 inductive 评估模式：

#### 初始化参数
```python
evaluator = LinkPredictionEvaluator(
    input_dim=256,           # 边嵌入维度
    threshold=0.5,           # 二分类阈值
    hidden_dim=128,          # MLP隐藏层维度
    dropout=0.1,            # Dropout率
    output_dim=1,            # 输出维度（固定为1）
    mode="inductive",        # "transductive" 或 "inductive"
    metric_names=["auroc", "ap", "f1", "recall"]  # 自定义指标
)
```

#### Inductive 评估支持

在 inductive 模式下，评估器会额外计算以下细粒度指标：

```python
# 新节点相关指标
- "auroc/new": 只包含新节点的边的 AUC
- "ap/new": 只包含新节点的边的 AP
- "f1/new": 只包含新节点的边的 F1

# 旧节点相关指标
- "auroc/old": 只包含旧节点的边的 AUC
- "ap/old": 只包含旧节点的边的 AP
- "f1/old": 只包含旧节点的边的 F1

# 边类型指标
- "auroc/new_old": 新节点→旧节点
- "auroc/old_new": 旧节点→新节点
- "auroc/new_new": 新节点→新节点
```

#### 损失函数
使用二元交叉熵损失：
```python
def compute_loss(self, outputs, targets):
    targets_float = targets.float().to(self.device)
    predictions = outputs["predictions"]
    return F.binary_cross_entropy_with_logits(predictions, targets_float)
```

#### 前向传播
通过 MLP 头将边嵌入转换为预测概率：
```python
def forward(self, edge_embeddings):
    predictions = self.mlp_head(edge_embeddings)
    if predictions.dim() > 1:
        predictions = predictions.squeeze(-1)
    return predictions
```

### 2. LinkClassificationEvaluator

链接分类评估器，支持二元和多类分类：

#### 初始化参数
```python
evaluator = LinkClassificationEvaluator(
    input_dim=256,           # 边嵌入维度
    num_classes=3,          # 类别数量（默认为2）
    hidden_dim=128,          # MLP隐藏层维度
    dropout=0.1,            # Dropout率
    label_smoothing=0.1,     # 标签平滑系数
    metric_names=["auroc", "ap", "acc"]  # 自定义指标
)
```

#### 损失函数
支持二元分类和多类分类的不同损失函数：
```python
def compute_loss(self, outputs, targets):
    predictions = outputs["predictions"]

    if self.num_classes == 2:
        # 二元分类：使用带标签平滑的 BCE
        smoothed_targets = targets.float()
        smoothed_targets = smoothed_targets * (1.0 - self.label_smoothing) + \
                          (1.0 - smoothed_targets) * self.label_smoothing
        return F.binary_cross_entropy_with_logits(predictions, smoothed_targets)
    else:
        # 多类分类：使用带标签平滑的交叉熵
        targets_long = targets.long()
        return F.cross_entropy(predictions, targets_long,
                             label_smoothing=self.label_smoothing)
```

#### 输出处理
根据类别数量调整输出形状：
```python
def forward(self, edge_embeddings):
    predictions = self.mlp_head(edge_embeddings)

    # 二元分类时去除多余维度
    if self.num_classes == 2 and predictions.dim() > 1:
        predictions = predictions.squeeze(-1)

    return predictions
```

### 3. NodeClassificationEvaluator

节点分类评估器（当前标记为未实现，预留接口）：

#### 计划功能
- 支持多类节点分类
- 宏平均和微平均指标
- per-class 详细指标
- 节点嵌入到类别的映射

## 自定义评估指标

### 添加新指标

1. **重写 init_metric() 方法，增加对自定义指标的初始化或者拓展 METRIC_MAPPING**:

2. **创建自定义指标类**:
```python
from torchmetrics import Metric

class CustomMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化状态变量
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        # 更新指标状态
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).long()
        self.correct += (predictions == targets).sum()
        self.total += targets.numel()

    def compute(self):
        # 计算最终指标值
        return self.correct.float() / self.total
```

3. **在评估器中使用**:
```python
# 初始化时指定自定义指标
evaluator = MyLinkPredictionEvaluator(
    input_dim=256,
    metric_names=["auroc", "ap", "custom_metric"]
)
```
