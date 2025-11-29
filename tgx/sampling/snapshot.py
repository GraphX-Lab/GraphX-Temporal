from typing import Iterator, Literal, Tuple

import numpy as np

from tgx.data.temporal_data import Data as TemporalData

# Duration = Tuple[float, float]


class SnapshotSampler:
    def __init__(
        self,
        temporal_graph: TemporalData,
        num_samples: int = -1,
        mode: Literal["sequential", "random"] = "random",
        duration_len: Literal['random'] | float = 0.1,
        drop_last: bool = False,
    ):
        self.temporal_graph = temporal_graph
        self.num_samples = num_samples
        self.duration_len = duration_len
        self.mode = mode
        self.drop_last = drop_last

        self.min_time = self.temporal_graph.timestamps.min()
        self.max_time = self.temporal_graph.timestamps.max()
        self.total_time = self.max_time - self.min_time
        
        assert mode != "random" or num_samples != -1, "num_samples must be specified in random mode."
        assert mode != "sequential" or duration_len != 'random', "duration_len must be specified in sequential mode."

    def sample(self, to_snapshot:bool=False) -> Iterator[TemporalData|Tuple[float, float]]:
        """Randomly samples snapshots from the dataset.
        Returns:
            list: A list of randomly sampled snapshots.
        """
        ...
        for start_time, end_time in self.iter_durations():
            if to_snapshot:
                yield self.temporal_graph.get_temporal_slice(start_time, end_time)
            else:
                yield (start_time, end_time)

    def parallel_sample(self, return_snapshot: bool = False, num_workers: int = None) -> Iterator[TemporalData | Tuple[float, float]]:
        """
        并行采样方法（推荐使用）
        :param return_snapshot: 是否返回子图（True时并行化）
        :param num_workers: 线程池大小（默认使用CPU核心数）
        :return: 生成器，按处理完成顺序返回结果
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        if not return_snapshot:
            # 不需要并行化时直接返回原始生成器
            yield from self.iter_durations()
            return
        if num_workers == 1:
            yield from self.sample(to_snapshot=return_snapshot)
            return
        
        # 1. 生成所有时间段（避免多次计算）
        time_intervals = list(self.iter_durations())
        total_intervals = len(time_intervals)
        
        if total_intervals == 0:
            return

        # 2. 设置默认worker数量（CPU核心数 - 1）
        if num_workers is None:
            num_workers = 4  # 避免过多进程

        # 3. 并行处理子图构建
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self.temporal_graph.get_temporal_slice, start_time, end_time): (start_time, end_time)
                for start_time, end_time in time_intervals
            }

            # 4. 按完成顺序yield结果
            for future in as_completed(futures.keys()):
                start_time, end_time = futures[future]
                try:
                    snapshot = future.result()
                    yield snapshot
                except Exception as e:
                    # 保留原始错误信息
                    raise RuntimeError(f"Snapshot build failed for {start_time}-{end_time}: {str(e)}") from e

    def iter_durations(self) -> Iterator[Tuple[float, float]]:
        """Generates durations for snapshot sampling.

        Returns:
            list: A list of durations for each snapshot.
        """
            
        cnt = 0
        while True:
            
            # get duration length
            if self.duration_len == 'random':
                duration_len = np.random.uniform(0.05, 0.5) * self.total_time
            else:
                duration_len = self.duration_len * self.total_time

            # get start and end time
            if self.mode == "random":
                start_time = np.random.uniform(self.min_time, self.max_time - duration_len)
            else:
                start_time = self.min_time + cnt * duration_len
            end_time = start_time + duration_len

            # check for last sample in sequential mode
            if self.drop_last and end_time > self.max_time:
                break

            yield start_time, min(end_time, self.max_time)

            # update counter
            cnt += 1
            
            # check for num_samples limit
            if self.num_samples > 0 and cnt >= self.num_samples:
                break
