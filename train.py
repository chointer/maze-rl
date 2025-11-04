import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.autoreset import AutoResetWrapper
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.buffers import ReplayBuffer

import time
from tqdm import tqdm
import sys, os
maze_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maze')
if maze_path not in sys.path:
    sys.path.append(maze_path)
import gym_maze

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(6, 32, 8, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2, stride=1), #nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 32),#nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(32, env.action_space.n),#nn.Linear(128, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":

    ### arguments ###
    run_name = "first_test"
    seed = 42
    run_name = "maze_seed42_1"
    track = False
    cuda = False

    learning_rate = 1e-4
    buffer_size = 100000
    total_timesteps = 10000000
    learning_starts = 80000
    train_frequency = 4
    trunctaion_limit = 1000

    # arguments: Epsilon Greedy
    start_e = 1
    end_e = 0.01
    exploration_fraction = 0.10

    # arguments: Training
    batch_size = 8
    gamma = 0.99
    target_network_frequency = 1000
    tau = 1.0

    ### Initialize ###
    # if track: import wandb
    
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Seed
    #random.seed(seed)
    np_rng = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    #torch.backends.cudnn.deterministic = ?

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # Initialize: Env
    env = gym.make('gym_maze/Maze-v0', render_mode="rgb_array", height_range=[5, 10], width_range=[5, 10])
    env = TimeLimit(env, trunctaion_limit)
    env = RecordEpisodeStatistics(env)
    env = AutoResetWrapper(env)

    # Initialize: Networks
    q_network = QNetwork(env).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    target_network = QNetwork(env).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # Initialize: Replay Buffer
    rb = ReplayBuffer(
        buffer_size,                   # Buffer Size
        env.observation_space,      # Observation Space
        env.action_space,           # Action Space
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    obs, _ = env.reset(seed=seed)       # return obs, info

    for global_step in tqdm(range(total_timesteps)):
        ### Env. Interaction ###
        epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)      # 전체 학습 스텝의 10% 동안 start_e에서 end_e까지 선형으로 변한다.
        if np_rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(device)
            q_value = q_network(obs_tensor)
            action = torch.argmax(q_value, dim=1).item()
            action -= 1
            # TODO. action은 -1부터 시작하는데, Qnetwork에서 행동 index는 0부터 시작한다. 지금은 index에 1을 빼줌으로써 맞춰줬지만, 나중에 환경에서 행동이 0부터 시작하게 수정하면 편할 것이다.

        # Execute the game
        next_obs, reward, termination, truncation, info = env.step(action)   # np.ndarray, float, bool, bool, dict

        # Episode End Handling
        if "final_info" in info:
            #print(f"global_step={global_step}, episodic_return={info['final_info']['episode']['r']}, moves={info['final_info']['move_count']}, newsize={env.maze_width}")
            writer.add_scalar("charts/episodic_return", info['final_info']["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info['final_info']["episode"]["l"], global_step)

        real_next_obs = next_obs.copy() if not truncation else info["final_observation"]
        # trunctation은 강제 종료한 상황이므로 next_obs를 사용하여 target Q를 계산해야한다. 때문에, 실제 final_observation을 가져온다. 
        # termination은 실제 환경이 종료된 상황으로, target Q 계산에 next_obs를 사용하지 않는다. 그래서 따로 final_observation으로 바꿔주지 않는 것 같다.

        # replay buffer
        if isinstance(action, int):
            action = np.array([[action]])
        if isinstance(reward, float):
            reward = np.array([[reward]])
        if isinstance(termination, bool):
            termination = np.array([[termination]])

        rb.add(obs, real_next_obs, action, reward, termination, info)       # add(obs, next_obs, action, reward, done, infos). truncation으로 끝나면, next_obs를 사용해서 target Q를 계산해야하므로, done에는 truncation만 반영하는 같다.
        
        obs = next_obs

        ### Train ###
        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                data = rb.sample(batch_size)
                mapped_actions_indices = (data.actions + 1).long()

                with torch.no_grad():
                    next_obs_tensor_batch = data.next_observations.float().permute(0, 3, 1, 2).to(device)
                    target_max, _ = target_network(next_obs_tensor_batch).max(dim=1)
                    td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
                obs_tensor_batch = data.observations.float().permute(0, 3, 1, 2).to(device)
                old_val = q_network(obs_tensor_batch).gather(1, mapped_actions_indices).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)))
                
                # Optimze
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update Target Network
            if global_step % target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        tau * q_network_param.data + (1.0 - tau) * target_network_param.data
                    )
    
    env.close()
    writer.close()
