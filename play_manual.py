import sys, os
maze_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maze')
if maze_path not in sys.path:
    sys.path.append(maze_path)

import gymnasium as gym
import gym_maze
from gym_maze.utils.play_discrete import play_discrete

env = gym.make('gym_maze/Maze-v0', render_mode="rgb_array", height_range=[5, 20], width_range=[5, 20], manual_mode=True)

rewards = {
    'goal': (5, None),
    'friction': (None, 0.01),
    'manhattan_dist': (0.03, None),
    'shortest_path': (0.1, None),
}

for r_name, r in rewards.items():
    env.reward_manager.add(r_name, r[0], r[1])

play_discrete(env, noop=-1)