# 数据集格式

本文档描述了GraphX-Temporal项目中使用的时序图学习数据集格式。

## 数据格式概述

数据集遵循标准的时序图基准格式（benchtemp格式），包含三个主要文件：

### 1. 边信息文件 (CSV)
**文件**: `ml_{dataset}.csv`
**列**: `u,i,ts,label,idx`
- `u`: 源节点ID
- `i`: 目标节点ID
- `ts`: 时间戳 (浮点数)
- `label`: 链接分类（动态节点分类）的二元标签 (0 或 1)
- `idx`: 边的索引/ID

**示例**:
```csv
u,i,ts,label,idx
4988,9703,21.846975001182933,1,0
2448,7947,52.27102028848574,0,1
2839,8992,61.27097883490418,1,2
```

### 2. 边特征文件 (NumPy)
**文件**: `ml_{dataset}.npy`
**形状**: `(n_edges, edge_feat_dim)`
- 每条边的特征向量
- 边特征与CSV文件中的边索引相对应

### 3. 节点特征文件 (NumPy)
**文件**: `ml_{dataset}_node.npy`
**形状**: `(n_nodes, node_feat_dim)`
- 每个节点的特征向量
- 节点ID从0开始索引

## 核心数据集

TGX框架主要支持Benchtemp格式的标准时序图数据集。
经典数据集如下：

| 数据集 | 描述 | 节点特征维度 | 边特征维度 | 规模 | 典型用途 |
|---------|-------------|-------------|-----------|-------|--------------|
| `mooc` | 大规模开放在线课程学习者交互数据 | 172维 | 4维 | ~411K 交互 | 链接预测、节点分类 |
| `reddit` | Reddit社交网络用户互动数据 | 172维 | 172维 | ~672K 交互 | 链接预测、节点分类 |
| `wikipedia` | Wikipedia编辑和协作互动数据 | 172维 | 172维 | ~157K 交互 | 链接预测、节点分类 |
| `lastfm` | LastFM音乐平台用户听歌记录 | 172维 | 2维 | ~1293K 交互 | 链接预测、节点分类 |

更多其他数据集及其下载：[Datasets for Paper "BenchTemp: A General Benchmark for Evaluating Temporal Graph Neural Networks"](https://zenodo.org/record/8267846)。


## 数据分割

数据集默认按时间顺序分割：
- **训练集**: 时间戳的前70%分位数。
- **验证集**: 时间戳的70%-85%分位数。
- **测试集**: 时间戳的85%-100%分位数。

链接预测任务的数据集支持两种模式：
- **直推式(Transductive)**：在历史上训练，在近期验证，在未来测试。
- **归纳式(Inductive)**：处理在验证/测试阶段出现的新节点。随机选择 10\% 的节点作为 unseen 节点，确保这些节点及其相关的边不在训练集中出现，用于验证模型的归纳能力。


## 数据加载

### PyTorch Lightning 数据模块

```python
from tgx.data import LPDataModule

# 创建链接数据模块
datamodule = LPDataModule(
    dataset="mooc",
    data_dir="./datasets",
    batch_size=32,
    mode="transductive",
)

# 设置数据 (训练/验证/测试分割)
datamodule.setup()

# 访问数据加载器
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()
```

### 数据集信息

提供了基础的数据集统计脚本：

```sh
python datastes/statis.py
```

针对每一个`datasets`目录下的数据集输出如下:

```
Dataset: lastfm                
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Property                         ┃ Value   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Number of Nodes in Node Features │ 1981    │
│ Number of Nodes in Edges         │ 1981    │
│ Number of Node Features          │ 172     │
├──────────────────────────────────┼─────────┤
│ Number of Edges                  │ 1293103 │
│ Number of Edge Features          │ 2       │
└──────────────────────────────────┴─────────┘
...
```

## 数据模块及其参数

本项目默认提供了用于链接预测和动态节点分类的数据模块（DataModule）：

- 链接预测：`tgx.data.LPDataModule`
- 链接分类（动态节点分类）：`tgx.data.LCDataModule`

它们公共参数分别如下：

|参数|类型|默认值|说明|
|---|---|---|---|
|dataset|`str`|`"mooc"`|数据集名称|
|data_dir|`str`|`"./datasets"`|数据集目录|
|batch_size|`int`|-1|默认full batch|
|pin_memory|`bool`|True|是否固定内存|
|shuffle_train|`bool`|True|是否打乱训练集|
|split_ratio|`List[float]`|`[0.7, 0.15, 0.15]`|数据集划分，当指定训练集大小时，默认时间轴向右对其|

`LPDataModule` 参数：

|参数|类型|默认值|说明|
|---|---|---|---|
|mode|`str`|`"transductive"`|模式|
|negative_sampling_ratio|`float`|`1.0`|负采样比率|
|different_new_nodes_between_val_and_test|`bool`|inductive模式下是否使用不同的测试验证unseen点集|

`LPDataModule` 参数：

|参数|类型|默认值|说明|
|---|---|---|---|
|num_classes|`int`|`None`|类别数量，默认二分类|