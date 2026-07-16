import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import copy
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

random.seed(0)

# ==========================================
# PART 1: NETWORKS
# ==========================================
# Critic J = COST-TO-GO (positive = bad). Capture = strong negative terminal.
# Actor MINIMISES J(s, actor(s)). This is the lecture's ADHDP actor-critic,
# stabilised with a target critic + experience replay (i.e. DDPG).

class Actor(nn.Module):
    def __init__(self, state_dim=6, hidden=64, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim=6, action_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# ==========================================
# PART 2: ENVIRONMENT
# ==========================================
class PursuitEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.dt = 0.1
        self.max_steps = 400
        self.capture_radius = 5.0

        self.v_evader_max = 8.0
        self.v_tracker_max = 12.0
        self.v_tracker_min = 4.0
        self.g_max = 5.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.p_tracker = np.array([0.0, 0.0, 0.0])
        self.v_tracker = np.array([self.v_tracker_max, 0.0, 0.0])

        self.p_evader = self.np_random.uniform(50, 100, size=3)
        ang = self.np_random.uniform(0, 2 * np.pi)
        self.v_evader = self.v_evader_max * np.array([np.cos(ang), np.sin(ang), 0.0])

        self.previous_distance = np.linalg.norm(self.p_evader - self.p_tracker)
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        self.p_evader += self.v_evader * self.dt

        forward = self.v_tracker / np.linalg.norm(self.v_tracker)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        a_lat = action[0] * self.g_max
        a_vert = action[1] * self.g_max
        delta_v = (a_lat * right + a_vert * up) * self.dt
        self.v_tracker += delta_v
        speed_cmd = self.v_tracker_min + (action[2] + 1.0) * 0.5 * (self.v_tracker_max - self.v_tracker_min)
        self.v_tracker = speed_cmd * (self.v_tracker / np.linalg.norm(self.v_tracker))
        self.p_tracker += self.v_tracker * self.dt

        current_distance = np.linalg.norm(self.p_evader - self.p_tracker)
        distance_closed = self.previous_distance - current_distance
        self.previous_distance = current_distance

        cost = -distance_closed
        cost += 0.05 * (abs(action[0]) + abs(action[1]))

        if current_distance < 25.0:
            speed_now = np.linalg.norm(self.v_tracker)
            cost += 0.1 *speed_now *self.dt


        terminated = False
        truncated = False
        if current_distance < self.capture_radius:
            terminated = True
            cost = -200.0
        elif self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), cost, terminated, truncated, {}

    def _get_obs(self):
        rel_pos = (self.p_evader - self.p_tracker) / 200.0
        rel_vel = (self.v_evader - self.v_tracker) / 25.0
        return np.concatenate([rel_pos, rel_vel]).astype(np.float32)


# ==========================================
# PART 3: TRAINING (ADHDP + target critic + replay buffer = DDPG)
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    env = PursuitEnv()
    actor = Actor()
    critic = Critic()
    actor_target = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)
    for p in actor_target.parameters():
        p.requires_grad_(False)
    for p in critic_target.parameters():
        p.requires_grad_(False)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    episodes = 2000
    gamma = 0.95
    MAX_STEPS = 400
    tau = 0.005
    batch_size = 256
    warmup = 1000                       # random steps before learning begins
    buffer = deque(maxlen=100_000)      # the replay buffer ("memory")

    explore_std = 0.3
    explore_decay = 0.998

    captures = 0
    capture_history = []
    reward_history = []
    total_steps = 0

    def soft_update(target, source):
        with torch.no_grad():
            for tp, p in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    for i in range(episodes):
        obs, _ = env.reset(seed=i)
        episode_cost = 0.0

        for j in range(MAX_STEPS):
            total_steps += 1

            # ---- choose action (random during warmup, else actor + noise) ----
            if total_steps < warmup:
                a = np.random.uniform(-1, 1, size=3)
            else:
                with torch.no_grad():
                    a = actor(torch.FloatTensor(obs)).numpy()
                a = (a + np.random.normal(0, explore_std, size=3)).clip(-1, 1)

            obs_next, c, terminated, truncated, _ = env.step(a.astype(np.float32))
            episode_cost += c

            # ---- store transition in replay buffer ----
            buffer.append((obs, a, c, obs_next, float(terminated)))
            obs = obs_next

            # ---- learn from a random minibatch ----
            if len(buffer) >= batch_size and total_steps >= warmup:
                batch = random.sample(buffer, batch_size)
                s, ac, r, s2, d = zip(*batch)
                s = torch.FloatTensor(np.array(s))
                ac = torch.FloatTensor(np.array(ac))
                r = torch.FloatTensor(np.array(r)).unsqueeze(1)
                s2 = torch.FloatTensor(np.array(s2))
                d = torch.FloatTensor(np.array(d)).unsqueeze(1)

                # critic update (target uses frozen target nets)
                with torch.no_grad():
                    a2 = actor_target(s2)
                    target = r + gamma * critic_target(s2, a2) * (1 - d)
                critic_loss = ((critic(s, ac) - target) ** 2).mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # actor update: minimise cost-to-go (+ small anti-saturation term)
                u = actor(s)
                actor_loss = critic(s, u).mean() + 0.01 * (u ** 2).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                soft_update(actor_target, actor)
                soft_update(critic_target, critic)

            if terminated:
                captures += 1
                break
            if truncated:
                break

        explore_std = max(0.02, explore_std * explore_decay)
        capture_history.append(captures)
        reward_history.append(episode_cost)

        if i % 50 == 0:
            print(f"Episode {i}/{episodes} | Captures: {captures} "
                  f"| Cost: {episode_cost:.1f} | final_dist: {env.previous_distance:.1f} "
                  f"| explore_std: {explore_std:.3f}")

    # ---- learning curves ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.plot(capture_history)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative captures")
    ax1.set_title("Learning curve - ADHDP + replay (DDPG) Tracker")
    ax2.plot(reward_history, alpha=0.4, label="Per episode")
    window = 20
    if len(reward_history) >= window:
        smoothed = np.convolve(reward_history, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(reward_history)), smoothed, label=f"{window}-ep average")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total cost")
    ax2.set_title("Cost per episode")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=120)
    plt.show()

    # ==========================================
    # PART 4: GREEDY EVALUATION + ANIMATE BEST EPISODE
    # ==========================================
    print("\nEvaluating greedy policy over 20 episodes (no exploration noise)...")

    episodes_data = []
    eval_caps = 0

    for k in range(20):
        obs, _ = env.reset()
        t_hist, e_hist = [], []
        captured = False
        for step in range(MAX_STEPS):
            t_hist.append(env.p_tracker.copy())
            e_hist.append(env.p_evader.copy())
            with torch.no_grad():
                action = actor(torch.FloatTensor(obs)).numpy()
            if k == 0 and step < 10:
                print(f"step {step}: action={action}, dist={env.previous_distance:.1f}")
            obs, c, terminated, truncated, _ = env.step(action)
            if terminated:
                captured = True
                break
            if truncated:
                break
        if captured:
            eval_caps += 1
        episodes_data.append({
            "tracker": np.array(t_hist),
            "evader": np.array(e_hist),
            "captured": captured,
            "final_dist": env.previous_distance,
            "steps": len(t_hist),
        })

    dists = [d["final_dist"] for d in episodes_data]
    print(f"Greedy capture rate: {eval_caps}/20 ({100*eval_caps/20:.0f}%) | "
          f"mean final distance: {np.mean(dists):.1f} m")
    print("per-episode final dists:", [f"{x:.0f}" for x in dists])

    captures_list = [d for d in episodes_data if d["captured"]]
    if captures_list:
        best = min(captures_list, key=lambda d: d["steps"])
        print(f"Animating a captured episode ({best['steps']} steps to capture).")
    else:
        best = min(episodes_data, key=lambda d: d["final_dist"])
        print(f"No captures in eval; animating closest miss (final dist {best['final_dist']:.1f} m).")

    tracker_history = best["tracker"]
    evader_history = best["evader"]

    print("\nGenerating 3D animation. Close the window to end the script.")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Trained ADHDP + replay: 3D Pursuit-Evasion')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

    all_x = np.concatenate((tracker_history[:, 0], evader_history[:, 0]))
    all_y = np.concatenate((tracker_history[:, 1], evader_history[:, 1]))
    all_z = np.concatenate((tracker_history[:, 2], evader_history[:, 2]))
    ax.set_xlim([all_x.min(), all_x.max()])
    ax.set_ylim([all_y.min(), all_y.max()])
    ax.set_zlim([all_z.min(), all_z.max()])

    tracker_line, = ax.plot([], [], [], color='blue', lw=2, label='Tracker')
    evader_line, = ax.plot([], [], [], color='red', ls='--', lw=2, label='Evader')
    tracker_point, = ax.plot([], [], [], 'o', color='blue', ms=8)
    evader_point, = ax.plot([], [], [], 'o', color='red', ms=8)
    ax.legend()

    def update(frame):
        tracker_line.set_data(tracker_history[:frame, 0], tracker_history[:frame, 1])
        tracker_line.set_3d_properties(tracker_history[:frame, 2])
        evader_line.set_data(evader_history[:frame, 0], evader_history[:frame, 1])
        evader_line.set_3d_properties(evader_history[:frame, 2])
        tracker_point.set_data([tracker_history[frame, 0]], [tracker_history[frame, 1]])
        tracker_point.set_3d_properties([tracker_history[frame, 2]])
        evader_point.set_data([evader_history[frame, 0]], [evader_history[frame, 1]])
        evader_point.set_3d_properties([evader_history[frame, 2]])
        return tracker_line, evader_line, tracker_point, evader_point

    ani = animation.FuncAnimation(fig, update, frames=len(tracker_history),
                                  interval=60, blit=False)
    plt.show()