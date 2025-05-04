REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .original_runner import EpisodeRunner as OriginalRunner
REGISTRY["original"] = OriginalRunner