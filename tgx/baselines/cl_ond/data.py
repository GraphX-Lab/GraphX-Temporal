from typing import List

import torch
from torch_geometric.data import DataLoader
from torch_geometric.sampler import NegativeSampling
from torch_geometric.utils import to_dense_adj
from joblib import Parallel, delayed

from tgx.data import LPDataModule
from tgx.sampling.snapshot import SnapshotSampler
from tgx.data import Data as TemporalData


class PretrainDataModule(LPDataModule):
    def __init__(
        self,
        dataset: str = "mooc",
        data_dir: str = "./datasets",
        batch_size: int = 32,
        split_ratio: List[float] = [0.8, 0.1, 0.1],
        num_neg_sampling: int = 10,
        num_pos_sampling: int = 10,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            data_dir=data_dir,
            batch_size=batch_size,
            split_ratio=split_ratio,
            negative_sampling_ratio=0.0,
            **kwargs,
        )
        self.view_samper = None
        self.num_neg_sampling = num_neg_sampling
        self.num_pos_sampling = num_pos_sampling

    def train_dataloader(self):
        # Return training dataloader
        # data = self._create_pyg_data(self.train_data)
        if self.view_samper is None:
            self.view_samper = SnapshotSampler(
                self.train_data,
                # negative_sampling_ratio=self.negative_sampling_ratio,
                mode="sequential",
                duration_len=0.1,
                drop_last=True,
            )
        views = self.view_samper.parallel_sample(num_workers=2, return_snapshot=True)
        # pyg_views = [self._create_pyg_data(view) for view in views]
        pyg_views = [self.process_view(g) for g in views]

        # pyg_views = Parallel(n_jobs=4, verbose=0)(
        #     delayed(self.process_view)(view) for view in views
        # )

        return DataLoader(
            [pyg_views],
            collate_fn=lambda x: x[0],
        )

    def process_view(self, view: TemporalData):
        """
        过于稀疏，大部分节点在快照中没有正样本，导致采样器报错。
        """
        
        g = self._create_pyg_data(view, self.train_node_type_info)
        pos_adj = to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes)[0].to(torch.bool)
        neg_adj = 1 - pos_adj.to(torch.int) - torch.eye(pos_adj.size(0))[0]

        neg_sampler = NegativeSampling("triplet", src_weight=neg_adj)
        neg_samples = neg_sampler.sample(self.num_neg_sampling, "src")
        g.neg = neg_samples

        return g

    def val_dataloader(self):
        # Return validation dataloader
        return None

    def test_dataloader(self):
        # Return test dataloader
        raise None

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        # Transfer batch to device
        if isinstance(batch, list):
            return [b.to(device) for b in batch]
        else:
            return batch.to(device)
