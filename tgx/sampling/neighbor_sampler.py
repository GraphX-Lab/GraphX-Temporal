"""Neighbor sampling strategies for temporal graphs."""

from typing import List, Dict, Optional, Tuple
import numpy as np


class TemporalNeighborSampler:
    """
    Temporal neighbor sampler for temporal graph neural networks.

    Samples temporal neighbors for each node in the batch,
    considering the temporal ordering of edges.
    """

    def __init__(
        self,
        num_neighbors: List[int],
        temporal_strategy: str = "uniform",
        max_time_window: Optional[float] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize temporal neighbor sampler.

        Args:
            num_neighbors: Number of neighbors to sample at each layer
            temporal_strategy: Strategy for temporal sampling ('uniform', 'recent', 'time_weighted')
            max_time_window: Maximum time window for sampling (None for no limit)
            seed: Random seed for reproducibility
        """
        self.num_neighbors = num_neighbors
        self.temporal_strategy = temporal_strategy
        self.max_time_window = max_time_window
        self.seed = seed

        if seed is not None:
            self.random_state = np.random.RandomState(seed)

    def sample_neighbors(
        self,
        node_ids: np.ndarray,
        timestamps: np.ndarray,
        edge_index: np.ndarray,
        edge_timestamps: np.ndarray,
        num_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample temporal neighbors for given nodes.

        Args:
            node_ids: Array of node IDs to sample neighbors for
            timestamps: Query timestamps for each node
            edge_index: Edge indices [2, num_edges]
            edge_timestamps: Timestamps for each edge
            num_samples: Number of neighbors to sample

        Returns:
            Tuple of (sampled_neighbors, sampled_timestamps, edge_ids)
        """
        sampled_neighbors = []
        sampled_timestamps = []
        sampled_edge_ids = []

        for i, node_id in enumerate(node_ids):
            query_time = timestamps[i]

            # Find neighbors that appear before the query time
            mask = (edge_index[1] == node_id) & (edge_timestamps < query_time)
            if self.max_time_window is not None:
                mask &= (edge_timestamps >= query_time - self.max_time_window)

            neighbor_nodes = edge_index[0][mask]
            neighbor_times = edge_timestamps[mask]
            neighbor_edge_ids = np.where(mask)[0]

            if len(neighbor_nodes) == 0:
                # No neighbors found, use dummy values
                sampled_neighbors.append([node_id])
                sampled_timestamps.append([query_time])
                sampled_edge_ids.append([0])
            else:
                # Sample neighbors based on strategy
                if self.temporal_strategy == "uniform":
                    indices = self._uniform_sample(len(neighbor_nodes), num_samples)
                elif self.temporal_strategy == "recent":
                    indices = self._recent_sample(neighbor_times, num_samples)
                elif self.temporal_strategy == "time_weighted":
                    indices = self._time_weighted_sample(neighbor_times, num_samples)
                else:
                    raise ValueError(f"Unknown temporal strategy: {self.temporal_strategy}")

                sampled_neighbors.append(neighbor_nodes[indices])
                sampled_timestamps.append(neighbor_times[indices])
                sampled_edge_ids.append(neighbor_edge_ids[indices])

        return (
            np.array(sampled_neighbors),
            np.array(sampled_timestamps),
            np.array(sampled_edge_ids)
        )

    def _uniform_sample(self, num_available: int, num_samples: int) -> np.ndarray:
        """Uniform random sampling."""
        if num_available <= num_samples:
            return np.arange(num_available)

        if self.seed is not None:
            return self.random_state.choice(num_available, num_samples, replace=False)
        else:
            return np.random.choice(num_available, num_samples, replace=False)

    def _recent_sample(self, timestamps: np.ndarray, num_samples: int) -> np.ndarray:
        """Sample most recent neighbors."""
        sorted_indices = np.argsort(timestamps)[::-1]  # Sort in descending order
        return sorted_indices[:min(num_samples, len(sorted_indices))]

    def _time_weighted_sample(self, timestamps: np.ndarray, num_samples: int) -> np.ndarray:
        """Time-weighted sampling (more recent = higher probability)."""
        if len(timestamps) <= num_samples:
            return np.arange(len(timestamps))

        # Use exponential decay weights based on recency
        max_time = np.max(timestamps)
        weights = np.exp((timestamps - max_time) * 0.1)  # Decay factor

        if self.seed is not None:
            return self.random_state.choice(
                len(timestamps), num_samples, replace=False, p=weights / weights.sum()
            )
        else:
            return np.random.choice(
                len(timestamps), num_samples, replace=False, p=weights / weights.sum()
            )