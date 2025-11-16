import random
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path
import yaml
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer

import sys, os
maze_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maze')
sys.path.append(maze_path)
import gym_maze
from Evaluator import Evaluator


class QNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        if len(obs_shape) == 4:
            _, input_h, input_w, input_c = obs_shape
        elif len(obs_shape) == 3:
            input_h, input_w, input_c = obs_shape
        else:
            raise ValueError(f"Unsupported observation space shape dimension: {len(obs_shape)}.")
        input_wh = input_h * input_w
        
        self.network = nn.Sequential(
            nn.Conv2d(input_c, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(input_wh * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.float().permute(0, 3, 1, 2)
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def make_maze_env(env_id, seed, height_range, width_range, truncation_limit, rewards_dict):
    def thunk():
        env = gym.make(env_id, height_range=height_range, width_range=width_range)
        env = gym.wrappers.TimeLimit(env, truncation_limit)
        env = gym.wrappers.RecordEpisodeStatistics(env)     # info에 정보가 추가된다. 학습에 영향 없음.
        #env = gym.wrappers.AutoResetWrapper(env)           # SyncVectorEnv는 기본으로 auto-reset을 적용한다고 함.
        for r_name, r in rewards_dict.items():
            env.reward_manager.add(r_name, r[0], r[1])

        #env = NoopResetEnv(env, noop_max=30)               # Atari game이 결정론적이라 항상 동일 상황에서 시작한다. 그래서 무작위 횟수만큼 noop(no operation?)을 수행하여 환경이 랜덤하게 바뀌게 한다. 미로 환경은 매번 랜덤으로 미로를 생성 및 배치하고, 기다려도 환경이 변하지 않으므로 사용할 필요없다.
        #env = MaxAndSkipEnv(env, skip=4)                   # 깜빡이는 개체가 보이지 않는 순간으로 인한 문제를 해결하기위한 래퍼. 미로 찾기에는 깜빡이는 개체가 없고, 이미지를 넣지 않을 수도 있기 때문에 사용하지 않는다.
        #env = EpisodicLifeEnv(env)                         # 목숨이 여러 개인 게임에 대해, 한 번만 죽어도 패널티를 받도록 하는 래퍼. 미로 찾기는 죽는 개념이 없으므로 사용하지 않는다.
        #if "FIRE" in env.unwrapped.get_action_meanings():  # FIRE 버튼을 눌러야 시작하는 일부 atari game에 사용하는 래퍼
        #    env = FireResetEnv(env)
        #env = ClipRewardEnv(env)                           # 게임마다 보상 스케일이 달라서 모든 reward를 -1, 0, 1로 정규화. 미로 찾기는 보상을 설계할 것이므로 사용 X
        #env = gym.wrappers.ResizeObservation(env, (84, 84))
        #env = gym.wrappers.GrayScaleObservation(env)
        #env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


def dict_to_ns(d):
    ns = {}
    for k, v in d.items():
        if isinstance(v, dict):
            ns[k] = dict_to_ns(v)
        else:
            ns[k] = v
    return SimpleNamespace(**ns)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return dict_to_ns(config)


if __name__ == "__main__":
    # ====================
    #      Arguments
    # ====================
    config_path = "config.yaml"
    cfg = load_config(config_path)

    # ====================
    #      Initialize
    # ====================
    run_idx = 0
    run_name = f"{cfg.env_id}__{cfg.exp_name}__{run_idx:02}"
    save_dir = Path(f"runs/{run_name}")
    while save_dir.exists() and save_dir.is_dir():
        run_idx += 1
        run_name = f"{cfg.env_id}__{cfg.exp_name}__{run_idx:02}"
        save_dir = Path(f"runs/{run_name}")
    (save_dir / "weights").mkdir(parents=True)
    
    loss = None
    evaluation_result = None

    # ===== Tensorboard =====
    writer = SummaryWriter(save_dir)
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    # ===== Seed =====
    random.seed(cfg.seed)
    np_rng = np.random.default_rng(seed=cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic   # ?

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    # ===== Init.Env =====
    if hasattr(cfg, 'reward') and cfg.reward is not None:
        rewards = vars(cfg.reward)
        """rewards = {
            'goal': (5, None),
            'friction': (None, 0.01),
            'manhattan_dist': (0.03, None),
            'shortest_path': (0.1, None),
        }
        """
    else:
        rewards = {}

    envs = gym.vector.SyncVectorEnv([
        make_maze_env(
            cfg.env_id, 
            cfg.seed + i, 
            height_range=cfg.height_range, 
            width_range=cfg.width_range, 
            truncation_limit=cfg.truncation_limit, 
            rewards_dict=rewards
        ) for i in range(cfg.num_envs)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # ===== Init.Env_eval =====
    rewards_eval = {
        'goal': (5, None),
        'friction': (None, 0.01),
    }
    env_eval = gym.make(cfg.env_id, render_mode="rgb_array", height_range=cfg.height_range, width_range=cfg.width_range)
    env_eval = gym.wrappers.TimeLimit(env_eval, cfg.truncation_limit)
    for r_name, r in rewards_eval.items():
        env_eval.reward_manager.add(r_name, r[0], r[1])
    evaluator = Evaluator(env_eval)
    best_eval_return = -float('inf')

    # ===== Init.Networks =====
    obs_shape = envs.single_observation_space.shape     # Shape(H, W, 4 + 2)
    n_actions = envs.single_action_space.n              # if hasattr(envs, "single_action_space") else envs.action_space.n
    q_network = QNetwork(obs_shape, n_actions).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=cfg.learning_rate)
    target_network = QNetwork(obs_shape, n_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())
    q_network.train()

    # ===== Init.ReplayBuffer =====
    rb = ReplayBuffer(
        cfg.buffer_size,                   # Buffer Size
        envs.single_observation_space,      # Observation Space
        envs.single_action_space,           # Action Space
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )


    # ====================
    #         Run
    # ====================
    start_time = time.time()

    obs, _ = envs.reset(seed=cfg.seed)       # return obs, info

    for global_step in tqdm(range(cfg.total_timesteps)):
        # ===== Action (Epsilon Greedy) =====
        epsilon = linear_schedule(cfg.start_e, cfg.end_e, cfg.exploration_fraction * cfg.total_timesteps, global_step)      # 전체 학습 스텝의 10% 동안 start_e에서 end_e까지 선형으로 변한다.
        if np_rng.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_value = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_value, dim=1).cpu().numpy()
            #actions -= 1

        # ===== Step =====
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)   # np.ndarray, float, bool, bool, dict

        # ===== Record ===== Episode End Handling
        if "final_info" in infos:
            """ final_info가 있을 때의 info sample
            {
                'move_count': array([0]), '_move_count': array([ True]), 
                'agent_location': array([array([0, 3])], dtype=object), '_agent_location': array([ True]), 
                'target_location': array([array([0, 2])], dtype=object), '_target_location': array([ True]), 
                'final_observation': array([array([[[...]]], dtype=float32)], dtype=object), '_final_observation': array([ True]), 
                'final_info': array([
                    {
                        'move_count': 89,
                        'agent_location': array([2, 2]), 
                        'target_location': array([2, 2]), 	
                        'manual_mode': False, 
                        'rewards': {'goal': 5, 'friction': -0.01, 'manhattan_dist': 0.02, 'shortest_path': 0.05}, 
                        'episode': {'r': array([4.55], dtype=float32), 'l': array([89], dtype=int32), 't': array([0.011475], dtype=float32)}
                    }
                ], dtype=object), 
                '_final_info': array([ True])
            }
            """
            for info in infos["final_info"]:
                if info and "episode" in info:
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
                # truncation은 강제 종료한 상황이므로 next_obs를 사용하여 target Q를 계산해야한다. 때문에, 실제 final_observation을 가져온다. 
                # termination은 실제 환경이 종료된 상황으로, target Q 계산에 next_obs를 사용하지 않는다. 그래서 따로 final_observation으로 바꿔주지 않는 것 같다.

        # ===== replay buffer =====
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # ====================
        #       Train
        # ====================
        if global_step > cfg.learning_starts:
            if global_step % cfg.train_frequency == 0:
                data = rb.sample(cfg.batch_size)
                #mapped_actions_indices = (data.actions + 1).long()

                with torch.no_grad():
                    next_obs_tensor_batch = data.next_observations.float().to(device)
                    target_max, _ = target_network(next_obs_tensor_batch).max(dim=1)
                    td_target = data.rewards.flatten() + cfg.gamma * target_max * (1 - data.dones.flatten())
                obs_tensor_batch = data.observations.float().to(device)
                old_val = q_network(obs_tensor_batch).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                writer.add_scalar("losses/lr", optimizer.param_groups[0]['lr'], global_step)

                # Optimze
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update Target Network
            if global_step % cfg.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        cfg.tau * q_network_param.data + (1.0 - cfg.tau) * target_network_param.data
                    )
        
        # ====================
        #     Evaluation
        # ====================
        if global_step % cfg.eval_frequency == 0 and global_step > cfg.learning_starts:
            def agent_fn(observ):
                with torch.no_grad():
                    observ = torch.Tensor(observ).to(device)
                    if observ.ndim == 3:
                        observ = observ.unsqueeze(0)

                    q_value = q_network(observ)
                    action = torch.argmax(q_value, dim=1).item()
                    #action -= 1
                return action
            
            q_network.eval()
            evaluation_result = evaluator.evaluate(agent_fn, n_episodes=cfg.n_episode_eval)
            writer.add_scalar("eval/mean_ep_returns", evaluation_result['mean_ep_returns'], global_step)
            writer.add_scalar("eval/moves_per_shortest", np.mean(evaluation_result['agent_moves']/evaluation_result['ep_min_moves']), global_step)
            q_network.train()

            if evaluation_result['mean_ep_returns'] > best_eval_return:
                best_eval_return = evaluation_result['mean_ep_returns']
                torch.save({
                    'global_step': global_step,
                    'q_network_state_dict': q_network.state_dict(),
                }, f"{save_dir}/weights/best.pt")
                print(f"New best.pt Saved: {save_dir}/weights/best.pt")


        # ===== print status =====
        if global_step % cfg.eval_frequency == 0:
            print_txt = f"[{global_step}/{cfg.total_timesteps}]\teps:{epsilon:.2f}\t"
            if global_step > cfg.learning_starts:
                if loss is not None:
                    print_txt += f"tr_l:{loss.item():.5f}\t"
                if evaluation_result is not None:
                    print_txt += f"val_r:{evaluation_result['mean_ep_returns']:.2f}\t uneff.:{np.mean(evaluation_result['agent_moves']/evaluation_result['ep_min_moves']):.2f}"
            print(print_txt)

        # ====================
        #     Freq. Save
        # ====================
        if global_step % cfg.save_frequency == 0 and global_step > cfg.learning_starts:
            checkpoint_path = f"{save_dir}/weights/checkpoint_{global_step}.pt"
            torch.save({
                'global_step': global_step,
                'q_network_state_dict': q_network.state_dict(),
                #'target_network_state_dict': target_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
            }, checkpoint_path)
            print(f"Checkpoint Saved: {checkpoint_path}")


    envs.close()
    writer.close()
