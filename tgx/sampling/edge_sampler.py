"""Edge sampling strategies for temporal graphs."""

from typing import Tuple, Optional
import numpy as np


class RandomEdgeSampler:
    """
    Random edge sampler for temporal graphs.

    Samples random source-destination pairs from the given node sets.
    This is commonly used for negative sampling in temporal graph learning.
    """

    def __init__(self, src_list: np.ndarray, dst_list: np.ndarray, seed: Optional[int] = None):
        """
        Initialize the random edge sampler.

        Args:
            src_list: Array of source node IDs
            dst_list: Array of destination node IDs
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.random_state = np.random.RandomState(seed)

    def sample(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample random source-destination pairs.

        Args:
            size: Number of samples to generate

        Returns:
            Tuple of (source_nodes, destination_nodes)
        """
        if self.seed is None:
            src_indices = np.random.randint(0, len(self.src_list), size)
            dst_indices = np.random.randint(0, len(self.dst_list), size)
        else:
            src_indices = self.random_state.randint(0, len(self.src_list), size)
            dst_indices = self.random_state.randint(0, len(self.dst_list), size)

        return self.src_list[src_indices], self.dst_list[dst_indices]

    def reset_random_state(self):
        """Reset the random state to the initial seed."""
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def set_seed(self, seed: int):
        """
        Set a new random seed.

        Args:
            seed: New random seed
        """
        self.seed = seed
        self.random_state = np.random.RandomState(seed)