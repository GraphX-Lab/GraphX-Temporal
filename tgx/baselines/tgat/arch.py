from typing import Literal, Optional
from torch.nn import ModuleList
import torch
from torch_geometric.nn import MLP

from tgx.models.temporal_model import TemporalModel
from tgx.utils.functions import to_neighbor_list
from tgx.nn.tgat import TGATConv


class TGATModel(TemporalModel):

    model_name = "TGAT"

    def __init__(
        self,
        input_dim: int = 172,
        hidden_dim: int = 256,
        edge_dim: int = 4,
        time_enc: Optional[Literal['time', 'pos']] = "time",
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super(TGATModel, self).__init__(*args, **kwargs)
        self.mlp = MLP(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
        )
        self.model = ModuleList(
            [
                TGATConv(
                    hidden_channels=hidden_dim,
                    edge_dim=edge_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    time_enc=time_enc,
                )
                for _ in range(num_layers)
            ]
        )
        # self.mlp_edge = MLP(
        #     in_channels=hidden_dim * 2,
        #     hidden_channels=hidden_dim,
        #     out_channels=hidden_dim,
        #     num_layers=1,
        # )

    def forward(self, batch):
        k = self.trainer.datamodule.num_neighbors
        x = self.mlp(batch.x)
        src_nodes, edge_ids, neighbors = to_neighbor_list(batch.edge_index)
        node_msk  = neighbors != -1
        edge_msk = edge_ids != -1

        x_i = x[src_nodes]
        time_i = torch.zeros((x_i.size(0), 1), device=x_i.device)
        time_j = torch.zeros((x_i.size(0), neighbors.size(1)), device=x_i.device, dtype=torch.long)
        edge_attr_j = torch.zeros((x_i.size(0), neighbors.size(1), batch.edge_attr.size(1)), device=x_i.device)

        time_j[edge_msk] = batch.edge_time[edge_ids[edge_msk]]
        edge_attr_j[edge_msk] = batch.edge_attr[edge_ids[edge_msk]]
        for i, conv in enumerate(self.model):
            x_j= x[neighbors]
            conv_x_i = conv.compute_node_temporal_embeddings(
                x_i = x_i,
                time_i = time_i,
                x_j = x_j,
                edge_attr_j = edge_attr_j,
                time_j = time_j,
                msk = node_msk,
            )
            x[src_nodes] = conv_x_i
        node_emb = x

        edge_emb = x[batch.edge_label_index[0]] + x[batch.edge_label_index[1]]  # (num_edges, hidden_dim)
        return {"node_embeddings": node_emb, "edge_embeddings": edge_emb}
