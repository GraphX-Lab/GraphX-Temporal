"""Test snapshot sampling functionality on real temporal graph datasets.

This module contains pytest test cases for the SnapshotSampler class using real-world temporal graph datasets including Reddit, MOOC, Wikipedia,
and LastFM datasets.
"""

import numpy as np
import pytest
import os

from tgx.data.temporal_data import Data
from tgx.sampling.snapshot import SnapshotSampler
from tgx.dataset.lp import LPDatasetLoader


@pytest.fixture(params=["reddit", "mooc", "wikipedia", "lastfm"])
def dataset_name(request):
    """Parameterize dataset names."""
    return request.param


@pytest.fixture
def real_temporal_data(dataset_name):
    """Load real temporal graph data using LPDatasetLoader."""
    # Skip test if data doesn't exist
    data_path = './datasets'
    if not os.path.exists(f"{data_path}/{dataset_name}/ml_{dataset_name}.csv"):
        pytest.skip(f"Dataset {dataset_name} not found at {data_path}")

    try:
        loader = LPDatasetLoader(
            dataset_path=data_path,
            dataset_name=dataset_name,
            val_ratio=0.15,
            test_ratio=0.15
        )
        node_features, edge_features, full_data, *_ = loader.load()
        return full_data
    except Exception as e:
        pytest.skip(f"Could not load dataset {dataset_name}: {e}")


class TestSnapshotSampler:
    """Test SnapshotSampler functionality on real datasets."""

    def test_snapshot_sampler_random_mode(self, real_temporal_data):
        """Test SnapshotSampler with random mode."""
        sampler = SnapshotSampler(
            temporal_graph=real_temporal_data,
            num_samples=5,
            mode="random",
            duration_len=0.1,
            drop_last=False
        )

        # Test time intervals
        time_intervals = list(sampler.sample(to_snapshot=False))
        assert len(time_intervals) == 5

        total_time = real_temporal_data.timestamps.max() - real_temporal_data.timestamps.min()
        for start_time, end_time in time_intervals:
            assert isinstance(start_time, (float, np.floating))
            assert isinstance(end_time, (float, np.floating))
            assert start_time >= real_temporal_data.timestamps.min()
            assert end_time <= real_temporal_data.timestamps.max() + 1e-6
            assert end_time > start_time

            # Check duration is approximately 10% of total time
            expected_duration = 0.1 * total_time
            actual_duration = end_time - start_time
            assert abs(actual_duration - expected_duration) < expected_duration * 0.1

    def test_snapshot_sampler_sequential_mode(self, real_temporal_data):
        """Test SnapshotSampler with sequential mode."""
        sampler = SnapshotSampler(
            temporal_graph=real_temporal_data,
            num_samples=5,
            mode="sequential",
            duration_len=0.1,
            drop_last=True
        )

        time_intervals = list(sampler.sample(to_snapshot=False))
        assert len(time_intervals) <= 5

        # Test temporal continuity in sequential mode
        for i in range(1, len(time_intervals)):
            prev_end = time_intervals[i-1][1]
            curr_start = time_intervals[i][0]
            # In sequential mode, current start should equal previous end
            assert abs(curr_start - prev_end) < 1e-6

    def test_snapshot_sampler_with_snapshots(self, real_temporal_data):
        """Test snapshot creation with return_snapshot=True."""
        sampler = SnapshotSampler(
            temporal_graph=real_temporal_data,
            num_samples=3,
            mode="random",
            duration_len=0.05,
            drop_last=False
        )

        snapshots = list(sampler.sample(to_snapshot=True))
        assert len(snapshots) == 3

        for snapshot in snapshots:
            assert isinstance(snapshot, Data)
            assert len(snapshot) > 0
            assert len(snapshot) <= len(real_temporal_data)

            # Check temporal consistency
            assert snapshot.timestamps.min() >= real_temporal_data.timestamps.min()
            assert snapshot.timestamps.max() <= real_temporal_data.timestamps.max()

            # Check temporal ordering within snapshot
            if len(snapshot) > 1:
                assert np.all(np.diff(snapshot.timestamps) >= 0)

    def test_snapshot_sampler_random_duration(self, real_temporal_data):
        """Test SnapshotSampler with random duration length."""
        sampler = SnapshotSampler(
            temporal_graph=real_temporal_data,
            num_samples=8,
            mode="random",
            duration_len='random',
            drop_last=False
        )

        snapshots = list(sampler.sample(to_snapshot=True))
        assert len(snapshots) == 8

        durations = []
        total_time = real_temporal_data.timestamps.max() - real_temporal_data.timestamps.min()

        for snapshot in snapshots:
            if len(snapshot) > 1:
                duration = snapshot.timestamps.max() - snapshot.timestamps.min()
                durations.append(duration)

                # Check duration is within reasonable bounds (5% to 50% of total time)
                assert duration >= 0.05 * total_time * 0.8  # Allow tolerance
                assert duration <= 0.5 * total_time * 1.2

        # Should have varied durations since it's random
        if len(durations) > 1:
            assert np.std(durations) > 0

    @pytest.mark.parametrize("num_workers", [1, 2, 4])
    def test_parallel_sample(self, real_temporal_data, num_workers):
        """Test parallel sampling functionality."""
        sampler = SnapshotSampler(
            temporal_graph=real_temporal_data,
            num_samples=6,
            mode="random",
            duration_len=0.08,
            drop_last=False
        )

        # Test parallel sampling with snapshots
        snapshots = list(sampler.parallel_sample(return_snapshot=True, num_workers=num_workers))
        assert len(snapshots) == 6

        for snapshot in snapshots:
            assert isinstance(snapshot, Data)
            assert len(snapshot) > 0

            # Check temporal consistency
            if len(snapshot) > 1:
                assert np.all(np.diff(snapshot.timestamps) >= 0)

    def test_parallel_sample_time_intervals(self, real_temporal_data):
        """Test parallel sampling returning time intervals."""
        sampler = SnapshotSampler(
            temporal_graph=real_temporal_data,
            num_samples=4,
            mode="random",
            duration_len=0.06,
            drop_last=False
        )

        time_intervals = list(sampler.parallel_sample(return_snapshot=False, num_workers=2))
        assert len(time_intervals) == 4

        for start_time, end_time in time_intervals:
            assert isinstance(start_time, (float, np.floating))
            assert isinstance(end_time, (float, np.floating))
            assert end_time > start_time

    @pytest.mark.parametrize("duration_len", [0.02, 0.05, 0.1, 0.2])
    def test_different_duration_lengths(self, real_temporal_data, duration_len):
        """Test sampler with different duration lengths."""
        sampler = SnapshotSampler(
            temporal_graph=real_temporal_data,
            num_samples=3,
            mode="random",
            duration_len=duration_len,
            drop_last=False
        )

        snapshots = list(sampler.sample(to_snapshot=True))
        assert len(snapshots) == 3

        total_time = real_temporal_data.timestamps.max() - real_temporal_data.timestamps.min()
        expected_max_duration = duration_len * total_time

        for snapshot in snapshots:
            if len(snapshot) > 1:
                actual_duration = snapshot.timestamps.max() - snapshot.timestamps.min()
                assert actual_duration <= expected_max_duration * 1.2  # Allow tolerance

    def test_drop_last_parameter(self, real_temporal_data):
        """Test drop_last parameter behavior."""
        sampler_drop_last = SnapshotSampler(
            temporal_graph=real_temporal_data,
            num_samples=10,
            mode="sequential",
            duration_len=0.15,
            drop_last=True
        )

        sampler_keep_last = SnapshotSampler(
            temporal_graph=real_temporal_data,
            num_samples=10,
            mode="sequential",
            duration_len=0.15,
            drop_last=False
        )

        intervals_drop = list(sampler_drop_last.sample(to_snapshot=False))
        intervals_keep = list(sampler_keep_last.sample(to_snapshot=False))

        # Keep last should have >= as many intervals as drop last
        assert len(intervals_keep) >= len(intervals_drop)



class TestCrossDatasetValidation:
    """Test functionality across different real datasets."""

    @pytest.mark.parametrize("mode", ["random", "sequential"])
    def test_different_modes_across_datasets(self, real_temporal_data, mode):
        """Test different sampling modes across datasets."""
        # Skip invalid combinations
        if mode == "random":
            num_samples = 5
            duration_len = 0.1
        else:  # sequential
            num_samples = 5
            duration_len = 0.1

        sampler = SnapshotSampler(
            temporal_graph=real_temporal_data,
            num_samples=num_samples,
            mode=mode,
            duration_len=duration_len,
            drop_last=False
        )

        snapshots = list(sampler.sample(to_snapshot=True))

        # Basic validation
        assert len(snapshots) <= num_samples

        for snapshot in snapshots:
            assert isinstance(snapshot, Data)
            assert len(snapshot) > 0

            # Should maintain temporal ordering
            if len(snapshot) > 1:
                assert np.all(np.diff(snapshot.timestamps) >= 0)

    def test_dataset_characteristics_preservation(self, real_temporal_data):
        """Test that sampling preserves dataset characteristics."""
        original_stats = {
            'num_edges': len(real_temporal_data),
            'time_span': real_temporal_data.timestamps.max() - real_temporal_data.timestamps.min(),
            'min_time': real_temporal_data.timestamps.min(),
            'max_time': real_temporal_data.timestamps.max()
        }

        sampler = SnapshotSampler(
            temporal_graph=real_temporal_data,
            num_samples=5,
            mode="random",
            duration_len=0.1
        )

        snapshots = list(sampler.sample(to_snapshot=True))

        # All snapshots should preserve basic characteristics
        for snapshot in snapshots:
            # Time range should be within original bounds
            assert snapshot.timestamps.min() >= original_stats['min_time']
            assert snapshot.timestamps.max() <= original_stats['max_time']

            # Should be smaller than original
            assert len(snapshot) <= original_stats['num_edges']

            # Duration should be reasonable
            if len(snapshot) > 1:
                snapshot_duration = snapshot.timestamps.max() - snapshot.timestamps.min()
                assert snapshot_duration <= original_stats['time_span']