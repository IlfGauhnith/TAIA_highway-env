import gymnasium as gym
import numpy as np
import os

class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super().__init__(env)
        old_space = env.observation_space

        self.dtype = dtype
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

        self.buffer = np.zeros_like(
            self.observation_space.low, dtype=self.dtype)

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation[0]
        return self.buffer


class SaveEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, save_path, file_prefix):
        super().__init__(env)
        self.save_path = save_path
        self.file_prefix = file_prefix

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            self.log_episode_reward(info['episode']['r'][0], info['episode']['l'][0])

        return obs, reward, terminated, truncated, info

    def log_episode_reward(self, cumulative_reward, episode_length):
        cumulative_reward_file = os.path.join(self.save_path, f"{self.file_prefix}_cumulative_reward.txt")
        mean_reward_file = os.path.join(self.save_path, f"{self.file_prefix}_mean_reward.txt")

        mean_reward = cumulative_reward/episode_length

        self.append_or_create_file(cumulative_reward_file, str(cumulative_reward))
        self.append_or_create_file(mean_reward_file, str(mean_reward))
    
    def append_or_create_file(self, file_path, content):
        if not os.path.exists(file_path):
            directory_path = os.path.dirname(file_path)
            os.makedirs(directory_path, exist_ok=True)

            with open(file_path, "w") as file:
                file.write(content + "\n")
        else:
            with open(file_path, "a") as file:
                file.write(content + "\n")

