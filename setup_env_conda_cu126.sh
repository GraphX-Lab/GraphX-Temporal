#!/bin/bash
set -euo pipefail  # 关键：任何命令失败/未定义变量/管道失败都会退出

# 创建环境
echo "⚙️ 正在创建并配置 conda 环境 tgl-x ..."
conda create -n tgl-x python=3.12 -y

# 使用 conda run 执行后续命令（避免 activate 问题）
echo "⚙️ 安装 CUDA Toolkit 12.6 ..."
conda run -n tgl-x conda install -c nvidia cuda-toolkit=12.6 -y

# 安装 PyTorch (cu126)
echo "⚙️ 安装 PyTorch 2.7.1 (CUDA 12.6) ..."
conda run -n tgl-x pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# 安装 PyG 核心包
echo "⚙️ 安装 PyTorch Geometric 相关包 ..."
conda run -n tgl-x pip install torch_geometric
conda run -n tgl-x pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.1+cu126.html

# 创建 requirements 文件并安装
echo "⚙️ 安装其他依赖包 ..."
cat > req.txt <<'REQ'
omegaconf
joblib
lightning
networkx
numpy
pandas
pytest
rich
scikit-learn
torchmetrics
tensorboard
REQ

# 注意：移除了 cuda-toolkit（已通过 conda 安装，pip 无法安装）
conda run -n tgl-x pip install -r req.txt

# 清理临时文件
rm -f req.txt

echo "✅ 环境 tgl-x 配置成功！"