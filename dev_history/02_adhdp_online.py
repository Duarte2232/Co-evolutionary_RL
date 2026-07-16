import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random;  random.seed(0)
# ==========================================
# PART 1: NETWORKS

#basic adhdp
# ==========================================
# Convention used throughout:
#   - The critic J approximates a COST-TO-GO (cost positive = bad).
#   - Capture is the goal, where J* = 0 (cost floor).
#   - The actor MINIMISES J(s, actor(s)).
# This matches the lecture's "actor reaches goal value J* = 0" statement,
# and keeps the reward/critic-target/actor-loss signs mutually consistent.

class Actor(nn.Module):
    def __init__(self, state_dim=6, hidden=64, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()  # bound action to [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    # input = state (6) + action (2) = 8
    def __init__(self, state_dim=6, action_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)  # linear output: cost-to-go can be large
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# PART 2: ENVIRONMENT
# ==========================================
class PursuitEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.dt = 0.1
        self.max_steps = 400
        self.capture_radius = 5.0

        self.v_evader_max = 10.0
        self.v_tracker_max = 12.0
        self.g_max = 5.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # tracker
        self.p_tracker = np.array([0.0, 0.0, 0.0])
        self.v_tracker = np.array([self.v_tracker_max, 0.0, 0.0])

        # evader: random point 50-100 m away, flying radially outward
        self.p_evader = self.np_random.uniform(50, 100, size=3)
        direction = self.p_evader / np.linalg.norm(self.p_evader)
        ang = self.np_random.uniform(0, 2*np.pi)
        self.v_evader = self.v_evader_max * np.array([np.cos(ang), np.sin(ang), 0.0])

        self.previous_distance = np.linalg.norm(self.p_evader - self.p_tracker)
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # 1. Move evader
        self.p_evader += self.v_evader * self.dt

        # 2. Steer tracker (constant-speed, lateral + vertical accel only)
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
        self.v_tracker = self.v_tracker_max * (self.v_tracker / np.linalg.norm(self.v_tracker))
        self.p_tracker += self.v_tracker * self.dt

        # 3. Cost-to-go formulation (positive cost = bad; J* = 0 at capture)
        current_distance = np.linalg.norm(self.p_evader - self.p_tracker)
        distance_closed = self.previous_distance - current_distance
        self.previous_distance = current_distance

        cost = -distance_closed                            # closing distance reduces cost
        cost += 0.05 * (abs(action[0]) + abs(action[1]))   # control effort penalty

        terminated = False
        truncated = False
        if current_distance < self.capture_radius:
            terminated = True
            cost = -200.0                                  # strong capture reward
        elif self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), cost, terminated, truncated, {}

    def _get_obs(self):
        rel_pos = (self.p_evader - self.p_tracker) / 200.0
        rel_vel = (self.v_evader - self.v_tracker) / 25.0
        return np.concatenate([rel_pos, rel_vel]).astype(np.float32)


# ==========================================
# PART 3: TRAINING (ADHDP)
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    

    env = PursuitEnv()
    actor = Actor()
    critic = Critic()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    episodes = 2000
    gamma = 0.95
    MAX_STEPS = 400
    explore_std = 0.3        # exploration noise (decays over training)
    explore_decay = 0.998

    captures = 0
    capture_history = []
    reward_history = []      # really cost history

    for i in range(episodes):
        obs, _ = env.reset(seed=i)
        episode_cost = 0.0

        for j in range(MAX_STEPS):
            obs_tensor = torch.FloatTensor(obs)

            # actor action + exploration noise
            u = actor(obs_tensor)
            noisy = (u.detach().numpy()
                     + np.random.normal(0, explore_std, size=2)).clip(-1, 1)
            obs_next, c, terminated, truncated, _ = env.step(noisy.astype(np.float32))
            episode_cost += c
            obs_next_tensor = torch.FloatTensor(obs_next)
            a_taken = torch.FloatTensor(noisy)

            # ----- Critic update: target = c + gamma * J(s', a') -----
            J_now = critic(torch.cat([obs_tensor, a_taken]))
            with torch.no_grad():
                u_next = actor(obs_next_tensor)
                J_next = critic(torch.cat([obs_next_tensor, u_next]))
                mask = 0.0 if terminated else 1.0
                target = c + gamma * J_next * mask
            critic_loss = 0.5 * (J_now - target) ** 2
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # ----- Actor update: minimise cost-to-go J(s, actor(s)) -----
            u_for_actor = actor(obs_tensor)
            J_for_actor = critic(torch.cat([obs_tensor, u_for_actor]))
            actor_loss = J_for_actor.mean() + 0.01 * (u_for_actor ** 2).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            obs = obs_next
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

    # ---- Learning curves ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.plot(capture_history)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative captures")
    ax1.set_title("Learning curve - ADHDP Tracker")

    ax2.plot(reward_history, alpha=0.4, label="Per episode")
    window = 20
    if len(reward_history) >= window:
        smoothed = np.convolve(reward_history, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(reward_history)), smoothed, label=f"{window}-ep avg")
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

    episodes_data = []   # one entry per eval episode
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
            if k == 0 and step < 10:          # <-- ADD THIS
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

    # pick the best episode to animate:
    #   prefer captures (shortest capture = cleanest intercept);
    #   if no captures, pick the one that got closest.
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
    ax.set_title('Trained ADHDP: 3D Pursuit-Evasion')
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