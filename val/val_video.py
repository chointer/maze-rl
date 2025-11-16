import gymnasium as gym
import torch
import torch.nn as nn
import sys
import cv2
import numpy as np

maze_path = "/home/sangho/workspace/maze-rl/maze"
if maze_path not in sys.path:
    sys.path.append(maze_path)
import gym_maze


model_path = "/home/sangho/workspace/maze-rl/runs/gym_maze/Maze-v0__dqn_00__00/weights/best.pt"
height_range = [3, 4]
width_range = [3, 5]

rewards = {
    'goal': (5, None),
    'friction': (None, 0.01),
    'manhattan_dist': (0.03, None),
    'shortest_path': (0.1, None),
}

output_filename = "validation_00.mp4"
truncation_limit = 100
n_episodes = 5
fps = 4

# =======================================================================

class QNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        if len(obs_shape) == 4:
            _, input_h, input_w, input_c = obs_shape
        elif len(obs_shape) == 3:
            input_h, input_w, input_c = obs_shape
        else:
            raise ValueError(f"Unsupported observation space shape dimension: {len(env.observation_space.shape)}.")
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


def draw_text(frame, text, tx, ty, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.32, color=(0, 0, 0), thickness=1):
    (txt_w, txt_h), _ = cv2.getTextSize(text, fontFace, fontScale, thickness)     # W, H
    txt_cx = int(tx - txt_w / 2)
    txt_cy = int(ty + txt_h / 2)
    cv2.putText(frame, text, (txt_cx, txt_cy), fontFace, fontScale, color=(255, 255, 255), thickness=thickness+1, lineType=cv2.LINE_AA)
    cv2.putText(frame, text, (txt_cx, txt_cy), fontFace, fontScale, color=color, thickness=thickness, lineType=cv2.LINE_AA)


action_to_direction = {
    0: np.array((-1, 0), dtype=int),        # Up
    1: np.array((1, 0), dtype=int),         # Down
    2: np.array((0, -1), dtype=int),        # Left
    3: np.array((0, 1), dtype=int),         # Right
    4: np.array((0, 0), dtype=int),         # No action
}

# =======================================================================
# ===== Init. Env =====
env = gym.make("gym_maze/Maze-v0", render_mode="rgb_array", height_range=height_range, width_range=width_range)
env = gym.wrappers.TimeLimit(env, truncation_limit)

for r_name, r in rewards.items():
    env.reward_manager.add(r_name, r[0], r[1])

obs_shape = env.observation_space.shape     # Shape(H, W, 4 + 2)
n_actions = env.action_space.n              # if hasattr(envs, "single_action_space") else envs.action_space.n


# ===== Init. Agent =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(model_path)

q_network = QNetwork(obs_shape, n_actions)
q_network.load_state_dict(ckpt['q_network_state_dict'])
#q_network.load_state_dict(ckpt)
q_network.eval()


# ===== Run =====
obs, info = env.reset()

frame = env.render()
h, w, c = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (h, w))

for i in range(n_episodes):
    obs, info = env.reset()
    draw_infos = env.get_draw_infos()
    sq_size = draw_infos['pix_square_size']
    startx = draw_infos['startx']
    starty = draw_infos['starty']

    termination, truncation = False, False
    steps = 0
    
    while not (termination or truncation):
        with torch.no_grad():
            observ = torch.Tensor(obs).to(device).unsqueeze(0)
            q_values = q_network(observ)
            action = torch.argmax(q_values, dim=1).item()
        
        # ===== Frame Write =====
        frame = env.render()
        ay, ax = info['agent_location']
        for i, q_val in enumerate(q_values[0].cpu().numpy()):
            dy, dx = action_to_direction[i]
            text_x = startx + (ax + dx + 0.5) * sq_size
            text_y = starty + (ay + dy + 0.5) * sq_size
            if i == action:
                draw_text(frame, f"{q_val:.2f}", text_x, text_y, color=(50, 255, 50))
            else:
                draw_text(frame, f"{q_val:.2f}", text_x, text_y, color=(0, 0, 0))
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

        
        next_obs, reward, termination, truncation, info = env.step(action)
        obs = next_obs

out.release()