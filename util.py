import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from stable_baselines3 import DQN, PPO
import numpy as np
from custom_wrappers import FrameStack, SaveEpisodeStatistics
import os


def defaultEnvironment(video_save_path, video_file_prefix, video_episode_trigger=lambda e: e % 1000 == 0, episode_statistics_save_path="", episode_statistics_file_prefix=""):
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
        "policy_frequency": 2,
        "offroad_terminal": True
    }
    env.unwrapped.configure(config)
    env.reset()

    env = RecordVideo(env=env, video_folder=video_save_path,
                      name_prefix=video_file_prefix,
                      episode_trigger=video_episode_trigger)
    env.unwrapped.set_record_video_wrapper(env)

    env = FrameStack(env, 4, np.uint8)

    if episode_statistics_save_path != "" and episode_statistics_file_prefix != "":
        env = RecordEpisodeStatistics(env, deque_size=1000)
        env = SaveEpisodeStatistics(
            env, episode_statistics_save_path, episode_statistics_file_prefix)

    return env


def createModel(env, model_algorithm="dqn", tensorboard_log_path="./model/dqn/tensorboard", checkpoint_load_path="", checkpoint_file_prefix=""):

    model = defaultModel(env, model_algorithm, tensorboard_log_path)

    if checkpoint_load_path:
        assert checkpoint_file_prefix, "When passing checkpoint_load_path to load a model from, checkpoint_file_prefix must be passed too."

        os.makedirs(checkpoint_load_path, exist_ok=True) # Create directory and dont raise error if already exists

        files = os.listdir(checkpoint_load_path)
        files = list(filter(lambda f: f.startswith(checkpoint_file_prefix) and f.endswith("_steps.zip"), files))
        files = sorted(files, key=lambda f: int(f.split("_")[-2]))

        if len(files) == 0:
            print(f"There's no {model_algorithm} checkpoints files at {checkpoint_load_path} with a '{checkpoint_file_prefix}' prefix.")
            print(f"Creating new {model_algorithm} model.")
            return model

        last_checkpoint_file, _ = os.path.splitext(files[-1])


        print(f"Loading {model_algorithm} model from last checkpoint {files[-1]}")

        model.set_parameters(os.path.join(checkpoint_load_path, last_checkpoint_file))
        model.num_timesteps = int(last_checkpoint_file.split("_")[-2])

        return model


    print(f"Creating new {model_algorithm} model.")
    return model

def defaultModel(env, model_algorithm, tensorboard_log_path):
    if model_algorithm == "dqn":
        return DQN("CnnPolicy", env,
                   learning_rate=5e-4,
                   buffer_size=50_000,
                   learning_starts=200,
                   batch_size=32,
                   gamma=0.8,
                   train_freq=1,
                   gradient_steps=1,
                   target_update_interval=50,
                   exploration_fraction=0.9,
                   verbose=1,
                   tensorboard_log=tensorboard_log_path)
    elif model_algorithm == "ppo":
        return PPO("CnnPolicy",
                    env,
                    verbose=1,
                    tensorboard_log=tensorboard_log_path)