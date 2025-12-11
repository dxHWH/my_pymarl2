REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .episode_runner_CFKeyAgent import CFKeyAgentEpisodeRunner
REGISTRY["episode_CFKeyAgent"] = CFKeyAgentEpisodeRunner

from .parallel_runner_CFKeyAgent import CFKeyAgentParallelRunner
REGISTRY["parallel_CFKeyAgent"] = CFKeyAgentParallelRunner
