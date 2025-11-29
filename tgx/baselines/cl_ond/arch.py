from pathlib import Path
from typing import List

import torch
from torch import nn
from torch_geometric.nn import GCN, MLP
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_undirected
from torch_scatter import scatter

from tgx.data import Data as TemporalData
from tgx.models import TemporalModel
from tgx.utils.typing import ModelOutputs


def compute_intra_community_density(r, adj):
    """计算社区内部密度"""
    n = r.size(0)
    k = r.size(1)
    # 公式(5): Dintra = (1/N) * sum_{i,j} sum_k [A[i,j] - d(k)] * R[i,k] * R[j,k]
    # 简化为使用A[i,j] - 0.5
    diff = adj - 0.5
    r_outer = torch.matmul(r, r.t())
    intra = (1 / n) * torch.sum(diff * r_outer)
    return intra


def compute_inter_community_density(r, adj):
    """计算社区外部密度"""
    # 公式(6): Dinter = (1/N) * sum_{i,j} sum_{k1≠k2} A[i,j] * R[i,k1] * R[j,k2]
    n = r.size(0)
    k = r.size(1)
    r_outer = torch.matmul(r, r.t())
    # 仅保留k1≠k2的元素
    r_outer = r_outer - torch.diag_embed(torch.diag(r_outer))
    inter = (1 / n) * torch.sum(adj * r_outer)
    return inter


class CLONDBackbone(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 172,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        super(CLONDBackbone, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.mlp = MLP(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
        )
        self.conv = GCN(
            in_channels=hidden_dim,
            out_channels=output_dim,
            hidden_channels=hidden_dim,
            num_layers=2,
        )

    def forward(self, x, edge_index):
        x = self.adjust_x_dim(x)
        x = self.mlp(x)
        x = self.conv(x, edge_index)
        return x

    def adjust_x_dim(self, x):
        x_dim = x.size(-1)
        if x_dim < self.input_dim:
            # Pad with zeros
            pad_size = self.input_dim - x_dim
            padding = torch.zeros(x.size(0), pad_size, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=-1)
        elif x_dim > self.input_dim:
            # Truncate
            x = x[:, : self.input_dim]
        return x


class PretrainModel(TemporalModel):
    def __init__(
        self,
        input_dim: int = 172,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_communities=10,
        alpha=0.5,
        lambda_=0.5,
        temperature=0.1,
        **kwargs,
    ):
        # hyperparameters
        self.alpha = alpha
        self.lambda_ = lambda_
        self.temperature = temperature

        super(PretrainModel, self).__init__(**kwargs)

        self.backbone = CLONDBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.community_centers = nn.Parameter(torch.randn(num_communities, output_dim))

    def forward(self, view_list: List[Data]):
        view_x = []
        community_affiliations = []
        num_views = len(view_list)
        sc_loss = 0.0
        tc_loss = 0.0

        pre_adj = None  # the adjacency matrix of the previous view
        for t, view in enumerate(view_list):
            x = view.x
            edge_index = view.edge_index
            h = self.backbone(x, edge_index)
            view_x.append(h)
            aff = torch.cosine_similarity(
                h.unsqueeze(1), self.community_centers.unsqueeze(0), dim=-1
            )
            community_affiliations.append(aff)

            adj = to_dense_adj(to_undirected(edge_index), max_num_nodes=h.size(0))[0]

            sc_loss_t = self.compute_sc_loss(aff, adj)
            sc_loss += sc_loss_t

            if t > 0:
                tc_loss_t = self.compute_tc_loss(
                    view_x[t - 1],
                    view_list[t - 1].edge_index,
                    view_list[t - 1].neg,
                    h,
                    edge_index,
                    view.neg,
                )
                tc_loss += tc_loss_t
            
        tc_loss = tc_loss / (num_views - 1)
        sc_loss = sc_loss / num_views
        return {
            "sc_loss": sc_loss,
            "tc_loss": tc_loss,
            "loss": self.alpha * sc_loss + (1 - self.alpha) * tc_loss,
        }

    def compute_sc_loss(self, aff, adj):
        """
        计算社区检测损失 (eq. 5-8)
        """
        D_intra = compute_intra_community_density(aff, adj)  # eq. 5
        D_inter = compute_inter_community_density(aff, adj)  # eq. 6
        sc_loss = self.lambda_ * D_inter - D_intra  # eq. 7
        return sc_loss

    def compute_tc_loss(self, h_i, e_i, neg_i, h_j, e_j, neg_j):
        """
        计算时间一致性损失 (eq. 9-12)
        j = i + 1
        """
        # positive inter-view same nodes
        pos_u_i = torch.cosine_similarity(h_i, h_j, dim=1)

        # positive inter-view neighbors
        sim_v_i = torch.cosine_similarity(h_i[e_i[0]], h_i[e_i[1]], dim=1)
        pos_v_i = torch.zeros_like(pos_u_i)
        scatter(sim_v_i, e_i[0], dim=0, reduce="mean", out=pos_v_i)

        # positive intra-view neighbors
        sim_v_j = torch.cosine_similarity(h_j[e_j[0]], h_j[e_j[1]], dim=1)
        pos_v_j = torch.zeros_like(pos_u_i)
        scatter(sim_v_j, e_j[0], dim=0, reduce="mean", out=pos_v_j)

        # [num_nodes]
        pos = pos_u_i + pos_v_i + pos_v_j

        # negative samples
        h_i_unsq = h_i.unsqueeze(1)
        neg_u_i = 1 - torch.cosine_similarity(h_i_unsq, h_i[neg_i], dim=-1)
        neg_u_j = 1 - torch.cosine_similarity(h_i_unsq, h_j[neg_i], dim=-1)
        # [num_nodes, num_neg_samples]
        neg = neg_u_i + neg_u_j

        pos_exp = torch.exp(pos / self.temperature)
        neg_exp = torch.exp(neg / self.temperature).sum(dim=1)

        tc_loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8)).mean()
        return tc_loss

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        for k, v in outputs.items():
            self.log(f"train/{k}", v, on_step=True, prog_bar=k == 'loss')
        
        return outputs['loss']

    def validation_step(self, batch, batch_idx):
        return 
    
    def test_step(self, batch, batch_idx):
        return 
    

    def configure_callbacks(self):
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
        cbs = [
            ModelCheckpoint(
                monitor="train/loss",
                mode="min",
                save_top_k=1,
                filename="best",
                verbose=False,
                dirpath=Path('checkpoints/cl-ond/') / self.trainer.datamodule.dataset / f"pretrain-{self.trainer.datamodule.mode}",
            ),
            EarlyStopping(
                monitor="train/loss",
                patience=10,
                mode="min",
                verbose=False
            ),
            RichProgressBar(leave=False),
        ]
        return cbs
    
class DownstreamModel(TemporalModel):
    backbone_name = "backbone"

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        pretrained_ckpt_path: Path = None,
        frozen_backbone: bool = False,
        **kwargs,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.frozen_backbone = frozen_backbone

        super(DownstreamModel, self).__init__(**kwargs)
        
        self.backbone = CLONDBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.mlp = MLP(
            in_channels=output_dim * 2,
            hidden_channels=output_dim,
            out_channels=output_dim,
            num_layers=1,
        )

    def forward(self, batch, *args, **kwargs) -> ModelOutputs: 
        x = batch.x
        edge_index = batch.edge_index
        h = self.backbone(x, edge_index)

        edge_label_index =  getattr(batch, 'edge_label_index', edge_index)
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_e = self.mlp(torch.cat([h_src, h_dst], dim=-1))
        # h_e = h_src + h_dst

        return {
            "node_embeddings": h,
            "edge_embeddings": h_e
        }
    
    def on_fit_start(self):
        if self.pretrained_ckpt_path is not None:
            self.load_pretrained(self.pretrained_ckpt_path)
        
            if self.frozen_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.print("Froze backbone parameters.")
        return super().on_fit_start()

    def load_pretrained(self, path: Path):
        state_dict = torch.load(path)['state_dict']
        backbone_state_dict = {
            k.replace(f"{self.backbone_name}.", ""): v
            for k, v in state_dict.items()
            if k.startswith(f"{self.backbone_name}.")
        }

        self.backbone.load_state_dict(backbone_state_dict)

        self.print(f"Loaded pretrained backbone from {path}")

    def configure_callbacks(self):
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
        task = 'lc' if 'LCDataModule' == self.trainer.datamodule.__class__.__name__ else 'lp'
        cbs = [
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                filename="best",
                verbose=False,
                dirpath=Path('checkpoints/cl-ond/') / self.trainer.datamodule.dataset / f"{task}-{self.trainer.datamodule.mode}",
            ),
            EarlyStopping(
                monitor="val/loss",
                patience=10,
                mode="min",
                verbose=False
            ),
            RichProgressBar(leave=False),
        ]
        return cbs