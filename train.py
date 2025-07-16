import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

import sys, os
maze_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maze')
if maze_path not in sys.path:
    sys.path.append(maze_path)
import gym_maze
#from gym_maze.utils.play_discrete import play_discrete

import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(6, 32, 8, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    # arguments
    seed = 42
    run_name = "maze_seed42_0"
    track = False
    cuda = False

    learning_rate = 1e-4
    buffer_size = 1000000
    total_timesteps = 10000000

    # arguments: Epsilon Greedy
    start_e = 1
    end_e = 0.01
    exploration_fraction = 0.10

    # Train Track
    if track:
        import wandb
        # TODO
    
    # Seed
    #random.seed(seed)
    np_rng = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    #torch.backends.cudnn.deterministic = ?

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # Setting: Env
    env = gym.make('gym_maze/Maze-v0', render_mode="rgb_array", height_range=[5, 20], width_range=[5, 20])
    env = RecordEpisodeStatistics(env)

    # Setting: Networks
    q_network = QNetwork(env).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    target_network = QNetwork(env).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # Setting: Replay Buffer
    rb = ReplayBuffer(
        buffer_size,                   # Buffer Size
        env.observation_space,      # Observation Space
        env.action_space,           # Action Space
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    # Start
    obs, _ = env.reset(seed=seed)       # return obs, info
    for global_step in range(total_timesteps):
        epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)      # 전체 학습 스텝의 10% 동안 start_e에서 end_e까지 선형으로 변한다.
        if np_rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(device)
            q_value = q_network(obs_tensor)
            action = torch.argmax(q_value, dim=1).item()     # return ex. [1]

        # Execute the game
        next_obs, reward, termination, trunction, info = env.step(action)   # np.ndarray, float, bool, bool, dict
        if termination:
            print("termination", global_step, info)

        if "final_info" in info:
            if info and "episode" in info:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                # TODO: torch.utils.tensorboard.SummaryWrieter
            assert False




    # TODO: 환경 상호작용, final_info쪽 구현, 학습 파트까지