import gymnasium as gym
from typing import Callable
import numpy as np

"""
Evaluator class
1. env 생성: **kwargs로 env 생성에 필요한 것 받기; height_range, width_range 등
2. 실행: agent 함수 받기. (미로에서는 model 돌리고 그 결과에 -1 한 것을 함수로 만들어서 줘야할 듯.) 그 함수로 환경 돌리고 결과를 도출.
TODO: Batch 처리할 수 있으면 더 좋긴할 듯
"""
class Evaluator:
    """
    에피소드를 여러 번 실행 후 평균 보상을 반환한다.
        __init__: Evaluation을 위한 gym env를 받는다. 자유롭게 env를 설정할 수 있도록, 그 자체를 받도록 했다.
        evaluate: 매개변수로 받은 agent로 행동을 선택하여 에피소드를 수행한다.
    """
    def __init__(self, env):
        self.env = env       # 미리 생성한 env 인스턴스를 전달받는다.
    
    def evaluate(self, agent_fn:Callable, n_episodes:int=10, seed=None, max_steps=10000):
        ep_returns = np.zeros(n_episodes)
        min_moves = np.zeros(n_episodes)
        agent_moves = np.zeros(n_episodes)

        for i in range(n_episodes):
            # === Reset ===
            if seed is not None:
                obs, info = self.env.reset(seed=seed + i)
            else:
                obs, info = self.env.reset()
            min_moves[i] = self.env.distance_map[*info['agent_location']]

            # === Run ===
            termination, truncation = False, False

            steps = 0
            while not (termination or truncation) and steps < max_steps:
                action = agent_fn(obs)
                next_obs, reward, termination, truncation, info = self.env.step(action)
                ep_returns[i] += reward
                obs = next_obs
                steps += 1

            if steps >= max_steps:
                print(f"[Evaluator Warning] Episode {i} reached max_step ({max_steps})")
            
            agent_moves[i] = info['move_count']

        return {
            'mean_ep_returns': np.mean(ep_returns),
            'all_ep_returns': ep_returns,
            'ep_min_moves': min_moves,
            'agent_moves': agent_moves,
        }


