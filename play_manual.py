import gymnasium as gym

import sys, os
maze_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maze')
if maze_path not in sys.path:
    sys.path.append(maze_path)

import gym_maze
from gym_maze.utils.play_discrete import play_discrete

env = gym.make('gym_maze/Maze-v0', render_mode="rgb_array", height_range=[5, 20], width_range=[5, 20])
play_discrete(env, noop=-1)