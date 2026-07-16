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
torch.manual_seed(0)
np.random.seed(0)

# ==========================================
# NETWORKS (shared definitions, used for both tracker and evader)
# ==========================================
class Actor(nn.Module):
    def __init__(self, state_dim=6, hidden=64, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# ==========================================
# ENVIRONMENT
# Now both tracker AND evader can steer. step() takes both actions.
# If evader_action is None, the evader flies a fixed straight heading
# (this is the Phase-0 setting used to train the tracker).
# ==========================================
class PursuitEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.dt = 0.1
        self.max_steps = 400
        self.capture_radius = 5.0
        self.v_evader_max = 8.0
        self.v_tracker_max = 12.0
        self.g_max = 5.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
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
        return self._get_obs_tracker(), {}

    def _steer(self, v, action):
        """Apply lateral+vertical steering to a constant-speed velocity vector."""
        speed = np.linalg.norm(v)
        forward = v / speed
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        delta_v = (action[0] * self.g_max * right + action[1] * self.g_max * up) * self.dt
        v_new = v + delta_v
        return speed * (v_new / np.linalg.norm(v_new))

    def step(self, tracker_action, evader_action=None):
        self.current_step += 1

        # --- move evader ---
        if evader_action is None:
            self.p_evader += self.v_evader * self.dt           # straight flight
        else:
            self.v_evader = self._steer(self.v_evader, evader_action)
            self.p_evader += self.v_evader * self.dt

        # --- move tracker ---
        self.v_tracker = self._steer(self.v_tracker, tracker_action)
        self.p_tracker += self.v_tracker * self.dt

        current_distance = np.linalg.norm(self.p_evader - self.p_tracker)
        distance_closed = self.previous_distance - current_distance
        self.previous_distance = current_distance

        # --- tracker reward (cost-to-go: low = good for tracker) ---
        tracker_cost = -distance_closed
        tracker_cost += 0.05 * (abs(tracker_action[0]) + abs(tracker_action[1]))

        # --- evader reward (cost-to-go: low = good for EVADER) ---
        # mirror: evader wants distance to GROW and to survive to max_steps
        evader_cost = distance_closed     # if tracker closes, evader cost rises (bad for evader)
        if evader_action is not None:
            evader_cost += 0.05 * (abs(evader_action[0]) + abs(evader_action[1]))

        terminated = False
        truncated = False
        if current_distance < self.capture_radius:
            terminated = True
            tracker_cost = -200.0          # capture: great for tracker
            evader_cost = +200.0           # capture: terrible for evader
        elif self.current_step >= self.max_steps:
            truncated = True
            evader_cost = -200.0           # survived the full episode: great for evader

        return (self._get_obs_tracker(), self._get_obs_evader(),
                tracker_cost, evader_cost, terminated, truncated, {})

    def _get_obs_tracker(self):
        rel_pos = (self.p_evader - self.p_tracker) / 200.0
        rel_vel = (self.v_evader - self.v_tracker) / 25.0
        return np.concatenate([rel_pos, rel_vel]).astype(np.float32)

    def _get_obs_evader(self):
        # evader sees the threat from its own perspective (signs flipped)
        rel_pos = (self.p_tracker - self.p_evader) / 200.0
        rel_vel = (self.v_tracker - self.v_evader) / 25.0
        return np.concatenate([rel_pos, rel_vel]).astype(np.float32)


# ==========================================
# A small DDPG "agent" bundle so we don't duplicate code for tracker & evader
# ==========================================
class DDPGAgent:
    def __init__(self, lr=3e-4):
        self.actor = Actor()
        self.critic = Critic()
        self.actor_t = copy.deepcopy(self.actor)
        self.critic_t = copy.deepcopy(self.critic)
        for p in self.actor_t.parameters(): p.requires_grad_(False)
        for p in self.critic_t.parameters(): p.requires_grad_(False)
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer = deque(maxlen=100_000)
        self.gamma = 0.95
        self.tau = 0.005
        self.batch_size = 256

    def act(self, obs, noise=0.0):
        with torch.no_grad():
            a = self.actor(torch.FloatTensor(obs)).numpy()
        if noise > 0:
            a = a + np.random.normal(0, noise, size=a.shape)
        return a.clip(-1, 1)

    def store(self, s, a, c, s2, done):
        self.buffer.append((s, a, c, s2, float(done)))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s, ac, r, s2, d = zip(*batch)
        s = torch.FloatTensor(np.array(s))
        ac = torch.FloatTensor(np.array(ac))
        r = torch.FloatTensor(np.array(r)).unsqueeze(1)
        s2 = torch.FloatTensor(np.array(s2))
        d = torch.FloatTensor(np.array(d)).unsqueeze(1)

        with torch.no_grad():
            a2 = self.actor_t(s2)
            target = r + self.gamma * self.critic_t(s2, a2) * (1 - d)
        c_loss = ((self.critic(s, ac) - target) ** 2).mean()
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()

        u = self.actor(s)
        a_loss = self.critic(s, u).mean() + 0.01 * (u ** 2).mean()
        self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()

        with torch.no_grad():
            for tp, p in zip(self.actor_t.parameters(), self.actor.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for tp, p in zip(self.critic_t.parameters(), self.critic.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)


# ==========================================
# PHASE 0: TRAIN THE TRACKER (vs straight-flying evader)
# ==========================================
def train_tracker(env, episodes=2000, warmup=1000):
    print("=== PHASE 0: training tracker vs straight-flying evader ===")
    agent = DDPGAgent()
    explore = 0.3
    total_steps = 0
    captures = 0
    for i in range(episodes):
        obs_t, _ = env.reset(seed=i)
        for j in range(env.max_steps):
            total_steps += 1
            if total_steps < warmup:
                a = np.random.uniform(-1, 1, size=2)
            else:
                a = agent.act(obs_t, noise=explore)
            obs_t2, _, c_t, _, term, trunc, _ = env.step(a, evader_action=None)
            agent.store(obs_t, a, c_t, obs_t2, term)
            obs_t = obs_t2
            if total_steps >= warmup:
                agent.learn()
            if term:
                captures += 1; break
            if trunc:
                break
        #explore = max(0.02, explore * 0.998)
        explore = max(0.0, explore * 0.997)
        if i % 100 == 0:
            print(f"  tracker ep {i}/{episodes} | captures {captures}")
    print(f"  tracker training done. total captures during training: {captures}")
    return agent


# ==========================================
# PHASE 1: FREEZE TRACKER, TRAIN EVADER
# ==========================================
def train_evader(env, tracker, episodes=2000, warmup=1000):
    print("\n=== PHASE 1: tracker FROZEN, training evader (survival) ===")
    # freeze tracker
    for p in tracker.actor.parameters():
        p.requires_grad_(False)

    evader = DDPGAgent()
    explore = 0.3
    total_steps = 0
    escapes = 0
    survival_history = []
    escape_history = []

    for i in range(episodes):
        obs_t, _ = env.reset(seed=10_000 + i)
        obs_e = env._get_obs_evader()
        steps_survived = 0
        for j in range(env.max_steps):
            total_steps += 1
            # tracker acts greedily (frozen, no noise)
            a_t = tracker.act(obs_t, noise=0.0)
            # evader acts + explores (learning)
            if total_steps < warmup:
                a_e = np.random.uniform(-1, 1, size=2)
            else:
                a_e = evader.act(obs_e, noise=explore)

            obs_t2, obs_e2, _, c_e, term, trunc, _ = env.step(a_t, evader_action=a_e)
            # evader stores its own transition. done = episode ended either way,
            # but we only bootstrap-terminate on capture (the evader's "absorbing" bad state)
            evader.store(obs_e, a_e, c_e, obs_e2, term)
            obs_t, obs_e = obs_t2, obs_e2
            steps_survived += 1
            if total_steps >= warmup:
                evader.learn()
            if term:
                break          # captured
            if trunc:
                escapes += 1
                break          # survived to max_steps = escape
        #explore = max(0.02, explore * 0.998)
        explore = max(0.0, explore * 0.997)
        survival_history.append(steps_survived)
        escape_history.append(escapes)
        if i % 100 == 0:
            recent = survival_history[-50:]
            print(f"  evader ep {i}/{episodes} | escapes {escapes} "
                  f"| avg survival (last50) {np.mean(recent):.0f} steps")
    print(f"  evader training done. total escapes: {escapes}/{episodes}")
    return evader, survival_history, escape_history


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    env = PursuitEnv()

    tracker = train_tracker(env, episodes=2000)
    evader, survival_history, escape_history = train_evader(env, tracker, episodes=2000)
    torch.save(tracker.actor.state_dict(), "tracker_actor.pt")
    torch.save(evader.actor.state_dict(), "evader_actor.pt")
    print("Saved tracker_actor.pt and evader_actor.pt")

    # ---- evader learning curves ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.plot(survival_history, alpha=0.3, label="per episode")
    w = 25
    if len(survival_history) >= w:
        sm = np.convolve(survival_history, np.ones(w)/w, mode='valid')
        ax1.plot(range(w-1, len(survival_history)), sm, label=f"{w}-ep avg")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Steps survived")
    ax1.set_title("Evader learning: survival time vs episode"); ax1.legend()
    ax2.plot(escape_history)
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Cumulative escapes")
    ax2.set_title("Evader: cumulative full-episode escapes")
    plt.tight_layout(); plt.savefig("evader_learning.png", dpi=120); plt.show()

    # ---- evaluate + animate one late episode (both greedy) ----
    print("\nEvaluating: tracker vs trained evader (both greedy)...")
    n_eval = 20
    eval_escapes = 0
    best = None
    for k in range(n_eval):
        obs_t, _ = env.reset(seed=50_000 + k)
        obs_e = env._get_obs_evader()
        t_hist, e_hist = [], []
        survived = True
        for step in range(env.max_steps):
            t_hist.append(env.p_tracker.copy())
            e_hist.append(env.p_evader.copy())
            a_t = tracker.act(obs_t, noise=0.0)
            a_e = evader.act(obs_e, noise=0.0)
            obs_t, obs_e, _, _, term, trunc, _ = env.step(a_t, evader_action=a_e)
            if term:
                survived = False; break
            if trunc:
                break
        if survived:
            eval_escapes += 1
        # animate the longest episode (most interesting juking)
        if best is None or len(t_hist) > best["len"]:
            best = {"t": np.array(t_hist), "e": np.array(e_hist),
                    "len": len(t_hist), "survived": survived}
    print(f"Evader escape rate (greedy): {eval_escapes}/{n_eval} "
          f"({100*eval_escapes/n_eval:.0f}%)")

    tracker_history, evader_history = best["t"], best["e"]
    print(f"Animating episode of {best['len']} steps "
          f"({'evader escaped' if best['survived'] else 'captured'}).")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Phase 1: Tracker (frozen) vs Trained Evader')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    all_x = np.concatenate((tracker_history[:, 0], evader_history[:, 0]))
    all_y = np.concatenate((tracker_history[:, 1], evader_history[:, 1]))
    all_z = np.concatenate((tracker_history[:, 2], evader_history[:, 2]))
    ax.set_xlim([all_x.min(), all_x.max()]); ax.set_ylim([all_y.min(), all_y.max()])
    ax.set_zlim([all_z.min(), all_z.max()])
    tl, = ax.plot([], [], [], color='blue', lw=2, label='Tracker')
    el, = ax.plot([], [], [], color='red', ls='--', lw=2, label='Evader')
    tp, = ax.plot([], [], [], 'o', color='blue', ms=8)
    ep, = ax.plot([], [], [], 'o', color='red', ms=8)
    ax.legend()
    def update(frame):
        tl.set_data(tracker_history[:frame, 0], tracker_history[:frame, 1]); tl.set_3d_properties(tracker_history[:frame, 2])
        el.set_data(evader_history[:frame, 0], evader_history[:frame, 1]); el.set_3d_properties(evader_history[:frame, 2])
        tp.set_data([tracker_history[frame, 0]], [tracker_history[frame, 1]]); tp.set_3d_properties([tracker_history[frame, 2]])
        ep.set_data([evader_history[frame, 0]], [evader_history[frame, 1]]); ep.set_3d_properties([evader_history[frame, 2]])
        return tl, el, tp, ep
    ani = animation.FuncAnimation(fig, update, frames=len(tracker_history), interval=60, blit=False)
    ani.save("phase1_evader.gif", writer="pillow", fps=20)
    print("Saved animation to phase1_evader.gif")
    plt.show()