import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import numpy as np
from custom_wrappers import FrameStack, SaveEpisodeStatistics

def defaultEnvironment(video_save_path, video_file_prefix, episode_statistics_save_path, episode_statistics_file_prefix):
    env = gym.make('highway-v0', render_mode="rgb_array")

    # Observation space as grayscale images without stacking and with (128, 64) shape.
    config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 1,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    "action": {
        "type": "DiscreteAction"
    },
    "policy_frequency": 2
    }
    env.unwrapped.configure(config)
    env.reset()

    env = RecordVideo(env=env, video_folder=video_save_path,
                    name_prefix=video_file_prefix, 
                    episode_trigger=lambda e: e % 1000 == 0)
    env.unwrapped.set_record_video_wrapper(env)

    env = FrameStack(env, 4, np.uint8)
    env = RecordEpisodeStatistics(env, deque_size=1000)
    env = SaveEpisodeStatistics(env, episode_statistics_save_path, episode_statistics_file_prefix)

    return env