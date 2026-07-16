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
import os

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


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
        speed = np.linalg.norm(v)
        forward = v / speed
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])  # fallback near vertical to avoid gimbal lock
        right = np.cross(forward, world_up); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        delta_v = (action[0] * self.g_max * right + action[1] * self.g_max * up) * self.dt
        v_new = v + delta_v
        return speed * (v_new / np.linalg.norm(v_new))

    def step(self, tracker_action, evader_action=None):
        self.current_step += 1
        if evader_action is None:
            self.p_evader += self.v_evader * self.dt
        else:
            self.v_evader = self._steer(self.v_evader, evader_action)
            self.p_evader += self.v_evader * self.dt

        self.v_tracker = self._steer(self.v_tracker, tracker_action)
        self.p_tracker += self.v_tracker * self.dt

        current_distance = np.linalg.norm(self.p_evader - self.p_tracker)
        distance_closed = self.previous_distance - current_distance  # positive when closing
        self.previous_distance = current_distance

        tracker_cost = -distance_closed + 0.05 * (abs(tracker_action[0]) + abs(tracker_action[1]))
        evader_cost = distance_closed
        if evader_action is not None:
            evader_cost += 0.05 * (abs(evader_action[0]) + abs(evader_action[1]))

        terminated = False
        truncated = False
        if current_distance < self.capture_radius:
            terminated = True
            tracker_cost = -200.0  # best outcome in cost-to-go (lower = better)
            evader_cost  = +200.0
        elif self.current_step >= self.max_steps:
            truncated = True
            evader_cost = -200.0

        return (self._get_obs_tracker(), self._get_obs_evader(),
                tracker_cost, evader_cost, terminated, truncated, {})

    def _get_obs_tracker(self):
        # 200 ≈ max engagement range, 25 ≈ max relative speed; keeps inputs O(1)
        rel_pos = (self.p_evader - self.p_tracker) / 200.0
        rel_vel = (self.v_evader - self.v_tracker) / 25.0
        return np.concatenate([rel_pos, rel_vel]).astype(np.float32)

    def _get_obs_evader(self):
        rel_pos = (self.p_tracker - self.p_evader) / 200.0
        rel_vel = (self.v_tracker - self.v_evader) / 25.0
        return np.concatenate([rel_pos, rel_vel]).astype(np.float32)


class DDPGAgent:
    def __init__(self, lr=3e-4):
        self.actor   = Actor()
        self.critic  = Critic()
        self.actor_t = copy.deepcopy(self.actor)
        self.critic_t = copy.deepcopy(self.critic)
        for p in self.actor_t.parameters():  p.requires_grad_(False)
        for p in self.critic_t.parameters(): p.requires_grad_(False)
        self.a_opt = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer     = deque(maxlen=100_000)
        self.gamma      = 0.95
        self.tau        = 0.005
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
        s  = torch.FloatTensor(np.array(s))
        ac = torch.FloatTensor(np.array(ac))
        r  = torch.FloatTensor(np.array(r)).unsqueeze(1)
        s2 = torch.FloatTensor(np.array(s2))
        d  = torch.FloatTensor(np.array(d)).unsqueeze(1)
        with torch.no_grad():
            a2     = self.actor_t(s2)
            target = r + self.gamma * self.critic_t(s2, a2) * (1 - d)  # no bootstrap on terminal
        c_loss = ((self.critic(s, ac) - target) ** 2).mean()
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()
        u      = self.actor(s)
        a_loss = self.critic(s, u).mean() + 0.01 * (u ** 2).mean()  # L2 to prevent action saturation
        self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
        with torch.no_grad():
            for tp, p in zip(self.actor_t.parameters(),  self.actor.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for tp, p in zip(self.critic_t.parameters(), self.critic.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def freeze(self):
        for p in self.actor.parameters():
            p.requires_grad_(False)


def train_tracker(env, episodes=2000, warmup=1000, label="tracker"):
    print(f"\n=== Training {label} vs straight evader ===")
    agent = DDPGAgent()
    explore = 0.3
    total_steps = 0
    captures = 0
    capture_history = []

    for i in range(episodes):
        obs_t, _ = env.reset(seed=i)
        for _ in range(env.max_steps):
            total_steps += 1
            # random actions during warmup to populate the buffer before any gradient step
            a = np.random.uniform(-1, 1, size=2) if total_steps < warmup else agent.act(obs_t, noise=explore)
            obs_t2, _, c_t, _, term, trunc, _ = env.step(a, evader_action=None)
            agent.store(obs_t, a, c_t, obs_t2, term)
            obs_t = obs_t2
            if total_steps >= warmup:
                agent.learn()
            if term:
                captures += 1; break
            if trunc:
                break
        explore = max(0.02, explore * 0.997)
        capture_history.append(captures)
        if i % 200 == 0:
            print(f"  ep {i}/{episodes} | captures {captures}")

    print(f"  Done. captures: {captures}/{episodes}")
    return agent, capture_history


def train_evader(env, tracker_agent, episodes=2000, warmup=1000, label="evader"):
    print(f"\n=== Training {label} vs frozen tracker ===")
    tracker_agent.freeze()
    agent = DDPGAgent()
    explore = 0.3
    total_steps = 0
    escapes = 0
    survival_history = []
    escape_history   = []

    for i in range(episodes):
        obs_t, _ = env.reset(seed=10_000 + i)  # offset so evader sees different episodes than tracker
        obs_e = env._get_obs_evader()
        steps_survived = 0
        for _ in range(env.max_steps):
            total_steps += 1
            a_t = tracker_agent.act(obs_t, noise=0.0)
            a_e = np.random.uniform(-1, 1, size=2) if total_steps < warmup else agent.act(obs_e, noise=explore)
            obs_t2, obs_e2, _, c_e, term, trunc, _ = env.step(a_t, evader_action=a_e)
            agent.store(obs_e, a_e, c_e, obs_e2, term)
            obs_t, obs_e = obs_t2, obs_e2
            steps_survived += 1
            if total_steps >= warmup:
                agent.learn()
            if term:
                break
            if trunc:
                escapes += 1; break
        explore = max(0.02, explore * 0.997)
        survival_history.append(steps_survived)
        escape_history.append(escapes)
        if i % 200 == 0:
            recent = survival_history[-50:] if len(survival_history) >= 50 else survival_history
            print(f"  ep {i}/{episodes} | escapes {escapes} | avg survival (last 50) {np.mean(recent):.0f}")

    print(f"  Done. escapes: {escapes}/{episodes}")
    return agent, survival_history, escape_history


def train_tracker_vs_evader(env, evader_agent, episodes=2000, warmup=1000, label="tracker"):
    print(f"\n=== Training {label} vs frozen maneuvering evader ===")
    evader_agent.freeze()
    agent = DDPGAgent()
    explore = 0.3
    total_steps = 0
    captures = 0
    capture_history = []

    for i in range(episodes):
        obs_t, _ = env.reset(seed=i)
        obs_e = env._get_obs_evader()
        for _ in range(env.max_steps):
            total_steps += 1
            a_t = np.random.uniform(-1, 1, size=2) if total_steps < warmup else agent.act(obs_t, noise=explore)
            a_e = evader_agent.act(obs_e, noise=0.0)
            obs_t2, obs_e2, c_t, _, term, trunc, _ = env.step(a_t, evader_action=a_e)
            agent.store(obs_t, a_t, c_t, obs_t2, term)
            obs_t, obs_e = obs_t2, obs_e2
            if total_steps >= warmup:
                agent.learn()
            if term:
                captures += 1; break
            if trunc:
                break
        explore = max(0.02, explore * 0.998)  # slower decay: harder opponent needs more exploration
        capture_history.append(captures)
        if i % 200 == 0:
            print(f"  ep {i}/{episodes} | captures {captures}")

    print(f"  Done. captures: {captures}/{episodes}")
    return agent, capture_history


def evaluate_pair(env, tracker_agent, evader_agent, n_eval=50, seed_offset=50_000):
    # seed_offset keeps eval episodes out of the training seed range
    captures = 0
    episodes_data = []

    for k in range(n_eval):
        obs_t, _ = env.reset(seed=seed_offset + k)
        obs_e    = env._get_obs_evader()
        t_hist, e_hist = [], []
        captured = False

        for _ in range(env.max_steps):
            t_hist.append(env.p_tracker.copy())
            e_hist.append(env.p_evader.copy())
            a_t = tracker_agent.act(obs_t, noise=0.0)
            a_e = evader_agent.act(obs_e, noise=0.0) if evader_agent is not None else None
            obs_t, obs_e, _, _, term, trunc, _ = env.step(a_t, evader_action=a_e)
            if term:
                captured = True; break
            if trunc:
                break

        if captured:
            captures += 1
        episodes_data.append({
            "tracker":    np.array(t_hist),
            "evader":     np.array(e_hist),
            "captured":   captured,
            "steps":      len(t_hist),
            "final_dist": env.previous_distance,
        })

    survival_steps = [e["steps"] for e in episodes_data]
    return {
        "capture_rate":  captures / n_eval,
        "captures":      captures,
        "n_eval":        n_eval,
        "mean_survival": np.mean(survival_steps),
        "episodes":      episodes_data,
    }


def print_eval(label, result):
    print(f"  {label:35s} | capture {result['captures']:2d}/{result['n_eval']} "
          f"({result['capture_rate']:5.1%}) | mean steps {result['mean_survival']:.0f}")


def pick_best_episode(result):
    # prefer shortest capture; fall back to closest miss if none
    caps = [e for e in result["episodes"] if e["captured"]]
    if caps:
        return min(caps, key=lambda e: e["steps"])
    return min(result["episodes"], key=lambda e: e["final_dist"])


def animate_episode(episode_data, title, filename):
    th = episode_data["tracker"]
    eh = episode_data["evader"]
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    all_x = np.concatenate([th[:, 0], eh[:, 0]])
    all_y = np.concatenate([th[:, 1], eh[:, 1]])
    all_z = np.concatenate([th[:, 2], eh[:, 2]])
    ax.set_xlim([all_x.min(), all_x.max()])
    ax.set_ylim([all_y.min(), all_y.max()])
    ax.set_zlim([all_z.min(), all_z.max()])
    tl,  = ax.plot([], [], [], color='blue', lw=2, label='Tracker')
    el,  = ax.plot([], [], [], color='red',  ls='--', lw=2, label='Evader')
    tp,  = ax.plot([], [], [], 'o', color='blue', ms=8)
    ep,  = ax.plot([], [], [], 'o', color='red',  ms=8)
    ax.legend()

    def update(frame):
        tl.set_data(th[:frame, 0], th[:frame, 1]); tl.set_3d_properties(th[:frame, 2])
        el.set_data(eh[:frame, 0], eh[:frame, 1]); el.set_3d_properties(eh[:frame, 2])
        tp.set_data([th[frame, 0]], [th[frame, 1]]); tp.set_3d_properties([th[frame, 2]])
        ep.set_data([eh[frame, 0]], [eh[frame, 1]]); ep.set_3d_properties([eh[frame, 2]])
        return tl, el, tp, ep

    ani = animation.FuncAnimation(fig, update, frames=len(th), interval=60, blit=False)
    ani.save(filename, writer="pillow", fps=20)
    print(f"  Saved {filename}")
    plt.close(fig)


def plot_distance_over_time(results_dict, filename):
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, len(results_dict), figsize=(6 * len(results_dict), 5), sharey=True)
    if len(results_dict) == 1:
        axes = [axes]

    for ax, (label, result) in zip(axes, results_dict.items()):
        for ep in result["episodes"]:
            th = ep["tracker"]
            eh = ep["evader"]
            dists = np.linalg.norm(eh - th, axis=1)
            color = "steelblue" if ep["captured"] else "firebrick"
            ax.plot(np.arange(len(dists)) * 0.1, dists, alpha=0.35, color=color, lw=1)
        ax.axhline(5.0, color="green", ls="--", lw=2)
        ax.set_xlabel("Time (s)")
        ax.set_title(label)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Relative distance (m)")

    legend_handles = [
        Line2D([0], [0], color="steelblue", label="Captured"),
        Line2D([0], [0], color="firebrick", label="Escaped"),
        Line2D([0], [0], color="green", ls="--", label="Capture radius (5 m)"),
    ]
    axes[-1].legend(handles=legend_handles, loc="upper right")
    fig.suptitle("Relative distance over time", fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.show()


def _action_vs_bearing(agent, dist_m=60.0):
    angles = np.linspace(0, 2 * np.pi, 360)
    lat_out, vrt_out = [], []
    for ang in angles:
        rel_pos = np.array([np.cos(ang), np.sin(ang), 0.0]) * dist_m / 200.0
        obs = torch.FloatTensor(np.concatenate([rel_pos, np.zeros(3)]))
        with torch.no_grad():
            a = agent.actor(obs).numpy()
        lat_out.append(a[0])
        vrt_out.append(a[1])
    return np.degrees(angles), np.array(lat_out), np.array(vrt_out)


def _value_heatmap(agent, grid_range=120, n=50):
    grid = np.linspace(-grid_range, grid_range, n)
    V = np.zeros((n, n))
    for i, y in enumerate(grid):
        for j, x in enumerate(grid):
            rel_pos = np.array([x, y, 0.0]) / 200.0
            obs = torch.FloatTensor(np.concatenate([rel_pos, np.zeros(3)]))
            with torch.no_grad():
                a = agent.actor(obs)
                V[i, j] = agent.critic(obs, a).item()
    return grid, V


def plot_agent_analysis(env, tracker_agents, evader_agents, filename):
    n_t = len(tracker_agents)
    n_e = len(evader_agents)
    fig, axes = plt.subplots(2, max(n_t, n_e) * 2, figsize=(7 * max(n_t, n_e), 10))

    dist_m = 60.0

    def fill_row(row_axes, agents, role, xy_label):
        for col, (label, agent) in enumerate(agents):
            ax_act = row_axes[col * 2]
            degs, lat, vrt = _action_vs_bearing(agent, dist_m)
            ax_act.plot(degs, lat, label="Lateral $a_{lat}$")
            ax_act.plot(degs, vrt, label="Vertical $a_{vert}$")
            ax_act.axhline(0, color="gray", ls="--", lw=0.8)
            ax_act.set_xlabel("Bearing angle (deg)")
            ax_act.set_ylabel("Action [-1, 1]")
            ax_act.set_title(f"{label}\nAction vs bearing (dist={dist_m:.0f} m, zero rel. vel)")
            ax_act.legend(fontsize=8)

            ax_val = row_axes[col * 2 + 1]
            grid, V = _value_heatmap(agent)
            cf = ax_val.contourf(grid, grid, V, levels=30, cmap="RdYlGn_r")
            plt.colorbar(cf, ax=ax_val, label="J(s, π(s))")
            circle = plt.Circle((0, 0), env.capture_radius,
                                 color="lime", fill=False, lw=2, label="Capture r.")
            ax_val.add_patch(circle)
            ax_val.plot(0, 0, "bs" if role == "tracker" else "rs", ms=8, label=role.capitalize())
            ax_val.set_xlabel(xy_label[0]); ax_val.set_ylabel(xy_label[1])
            ax_val.set_title(f"{label}\nCost-to-go heatmap (zero rel. vel)")
            ax_val.set_aspect("equal"); ax_val.legend(fontsize=8)

        for col in range(len(agents), len(row_axes) // 2):
            row_axes[col * 2].set_visible(False)
            row_axes[col * 2 + 1].set_visible(False)

    fill_row(axes[0], tracker_agents, "tracker",
             ("Evader X relative to tracker (m)", "Evader Y relative to tracker (m)"))
    fill_row(axes[1], evader_agents,  "evader",
             ("Tracker X relative to evader (m)", "Tracker Y relative to evader (m)"))

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.show()


def save_eval_table(results_dict, filename):
    rows = [[label,
             f"{res['captures']}/{res['n_eval']}",
             f"{res['capture_rate']:.0%}",
             f"{res['mean_survival']:.0f} steps"]
            for label, res in results_dict.items()]

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Matchup", "Captures", "Capture rate", "Mean survival"],
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.9)
    for col in range(4):
        tbl[0, col].set_facecolor("#d0e4f7")
    ax.set_title("Cross-phase evaluation summary", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    env = PursuitEnv()
    os.makedirs("models",  exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    N_EPISODES = 2000
    WARMUP     = 1000
    N_EVAL     = 50

    # Phase 0
    tracker0, cap_hist0 = train_tracker(
        env, episodes=N_EPISODES, warmup=WARMUP, label="tracker0")
    torch.save(tracker0.actor.state_dict(), "models/tracker0_actor.pt")

    # Phase 1
    evader1, surv1, esc1 = train_evader(
        env, tracker0, episodes=N_EPISODES, warmup=WARMUP, label="evader1")
    torch.save(evader1.actor.state_dict(), "models/evader1_actor.pt")

    # Phase 2
    tracker2, cap_hist2 = train_tracker_vs_evader(
        env, evader1, episodes=N_EPISODES, warmup=WARMUP, label="tracker2")
    torch.save(tracker2.actor.state_dict(), "models/tracker2_actor.pt")

    # Phase 3
    evader3, surv3, esc3 = train_evader(
        env, tracker2, episodes=N_EPISODES, warmup=WARMUP, label="evader3")
    torch.save(evader3.actor.state_dict(), "models/evader3_actor.pt")

    print("\nAll models saved.")

    print("\n--- Cross-phase evaluation (greedy, no noise) ---")
    r_t0_straight = evaluate_pair(env, tracker0, None,    n_eval=N_EVAL)
    r_t0_e1       = evaluate_pair(env, tracker0, evader1, n_eval=N_EVAL)
    r_t0_e3       = evaluate_pair(env, tracker0, evader3, n_eval=N_EVAL)
    r_t2_straight = evaluate_pair(env, tracker2, None,    n_eval=N_EVAL)
    r_t2_e1       = evaluate_pair(env, tracker2, evader1, n_eval=N_EVAL)
    r_t2_e3       = evaluate_pair(env, tracker2, evader3, n_eval=N_EVAL)

    print_eval("tracker0 vs straight evader", r_t0_straight)
    print_eval("tracker0 vs evader1",          r_t0_e1)
    print_eval("tracker0 vs evader3",          r_t0_e3)
    print_eval("tracker2 vs straight evader", r_t2_straight)
    print_eval("tracker2 vs evader1",          r_t2_e1)
    print_eval("tracker2 vs evader3",          r_t2_e3)

    arms_labels = [
        "Ph0 tracker\nvs straight",
        "Ph0 tracker\nvs Ph1 evader",
        "Ph2 tracker\nvs Ph1 evader",
        "Ph2 tracker\nvs Ph3 evader",
    ]
    arms_rates = [
        r_t0_straight["capture_rate"],
        r_t0_e1["capture_rate"],
        r_t2_e1["capture_rate"],
        r_t2_e3["capture_rate"],
    ]
    bar_colors = ["steelblue", "firebrick", "steelblue", "firebrick"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(arms_labels, [r * 100 for r in arms_rates],
                  color=bar_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Tracker capture rate (%)")
    ax.set_title("Co-evolutionary arms race: capture rate by phase")
    ax.set_ylim(0, 115)
    ax.axhline(50, color="gray", ls="--", lw=1, label="50 % line")
    for bar, rate in zip(bars, arms_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2, f"{rate:.0%}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/arms_race.png", dpi=120)
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    w = 25

    def smooth(hist, w):
        if len(hist) >= w:
            return range(w - 1, len(hist)), np.convolve(hist, np.ones(w) / w, mode='valid')
        return range(len(hist)), np.array(hist, dtype=float)

    for ax, hist, title in [
        (axes[0, 0], cap_hist0, "Phase 0 tracker: cumulative captures (vs straight)"),
        (axes[0, 1], cap_hist2, "Phase 2 tracker: cumulative captures (vs evader1)"),
    ]:
        ax.plot(hist)
        ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative captures")
        ax.set_title(title)

    for ax, hist, title in [
        (axes[1, 0], surv1, "Phase 1 evader: survival time (vs tracker0)"),
        (axes[1, 1], surv3, "Phase 3 evader: survival time (vs tracker2)"),
    ]:
        ax.plot(hist, alpha=0.3, label="per episode")
        xs, ys = smooth(hist, w)
        ax.plot(xs, ys, label=f"{w}-ep avg")
        ax.set_xlabel("Episode"); ax.set_ylabel("Steps survived")
        ax.set_title(title); ax.legend()

    for ax, hist, title in [
        (axes[0, 2], esc1, "Phase 1 evader: cumulative escapes"),
        (axes[1, 2], esc3, "Phase 3 evader: cumulative escapes"),
    ]:
        ax.plot(hist)
        ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative escapes")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("figures/learning_curves.png", dpi=120)
    plt.show()

    animate_episode(
        pick_best_episode(r_t0_e1),
        "Phase 0 tracker vs Phase 1 evader",
        "figures/phase1_evader.gif")
    animate_episode(
        pick_best_episode(r_t2_e1),
        "Phase 2 tracker vs Phase 1 evader",
        "figures/phase2_chase.gif")
    animate_episode(
        pick_best_episode(r_t2_e3),
        "Phase 2 tracker vs Phase 3 evader",
        "figures/phase3_evader.gif")

    print("\nDone.")
