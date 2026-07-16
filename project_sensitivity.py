"""
Sensitivity analysis for the co-evolutionary DDPG pursuit-evasion pipeline.

Sweeps three parameters independently:
  - Evader max speed  (v_evader_max): 6, 8*, 10, 12  m/s
  - Discount factor   (gamma):        0.90, 0.95*, 0.99
  - Capture radius    (capture_radius): 2.5, 5.0*, 10.0  m
  (* = baseline value used in project_pipeline.py)

Results are saved progressively to sensitivity/results.json after each config,
so the script can be interrupted and resumed without losing completed runs.


Classes are duplicated from project_pipeline.py (no shared import) so this script
can run standalone without touching the main pipeline's saved models or state.
"""

import os
import json
import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces


class Actor(nn.Module):
    def __init__(self, state_dim=6, hidden=64, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
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
    def __init__(self, capture_radius=5.0, v_evader_max=8.0,
                 v_tracker_max=12.0, g_max=5.0):
        super().__init__()
        self.dt             = 0.1
        self.max_steps      = 400
        self.capture_radius = capture_radius
        self.v_evader_max   = v_evader_max
        self.v_tracker_max  = v_tracker_max
        self.g_max          = g_max
        self.action_space      = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1e3, 1e3, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step  = 0
        self.p_tracker     = np.array([0., 0., 0.])
        self.v_tracker     = np.array([self.v_tracker_max, 0., 0.])
        self.p_evader      = self.np_random.uniform(50, 100, size=3)
        ang                = self.np_random.uniform(0, 2 * np.pi)
        self.v_evader      = self.v_evader_max * np.array([np.cos(ang), np.sin(ang), 0.])
        self.previous_distance = np.linalg.norm(self.p_evader - self.p_tracker)
        return self._obs_tracker(), {}

    def _steer(self, v, action):
        speed   = np.linalg.norm(v)
        forward = v / speed
        up_w    = np.array([0., 0., 1.])
        if abs(np.dot(forward, up_w)) > 0.99:
            up_w = np.array([0., 1., 0.])  # fallback near vertical to avoid gimbal lock
        right = np.cross(forward, up_w); right /= np.linalg.norm(right)
        up    = np.cross(right, forward)
        v_new = v + (action[0] * self.g_max * right +
                     action[1] * self.g_max * up) * self.dt
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

        d  = np.linalg.norm(self.p_evader - self.p_tracker)
        dc = self.previous_distance - d
        self.previous_distance = d

        tc = -dc + 0.05 * (abs(tracker_action[0]) + abs(tracker_action[1]))
        ec = dc
        if evader_action is not None:
            ec += 0.05 * (abs(evader_action[0]) + abs(evader_action[1]))

        term = trunc = False
        if d < self.capture_radius:
            term, tc, ec = True, -200., +200.
        elif self.current_step >= self.max_steps:
            trunc, ec = True, -200.

        return self._obs_tracker(), self._obs_evader(), tc, ec, term, trunc, {}

    def _obs_tracker(self):
        return np.concatenate([(self.p_evader - self.p_tracker) / 200.,
                                (self.v_evader - self.v_tracker) / 25.]).astype(np.float32)

    def _obs_evader(self):
        return np.concatenate([(self.p_tracker - self.p_evader) / 200.,
                                (self.v_tracker - self.v_evader) / 25.]).astype(np.float32)


class DDPGAgent:
    def __init__(self, gamma=0.95, lr=3e-4):
        self.actor    = Actor()
        self.critic   = Critic()
        self.actor_t  = copy.deepcopy(self.actor)
        self.critic_t = copy.deepcopy(self.critic)
        for p in self.actor_t.parameters():  p.requires_grad_(False)
        for p in self.critic_t.parameters(): p.requires_grad_(False)
        self.a_opt      = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.c_opt      = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer     = deque(maxlen=100_000)
        self.gamma      = gamma
        self.tau        = 0.005
        self.batch_size = 256

    def act(self, obs, noise=0.):
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
            target = r + self.gamma * self.critic_t(s2, self.actor_t(s2)) * (1 - d)
        c_loss = ((self.critic(s, ac) - target) ** 2).mean()
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()
        u      = self.actor(s)
        a_loss = self.critic(s, u).mean() + 0.01 * (u ** 2).mean()
        self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
        with torch.no_grad():
            for tp, p in zip(self.actor_t.parameters(), self.actor.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for tp, p in zip(self.critic_t.parameters(), self.critic.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def freeze(self):
        for p in self.actor.parameters():
            p.requires_grad_(False)


def train_tracker(env, gamma, episodes, warmup):
    agent = DDPGAgent(gamma=gamma)
    explore, total_steps, captures = 0.3, 0, 0
    for i in range(episodes):
        obs_t, _ = env.reset(seed=i)
        for _ in range(env.max_steps):
            total_steps += 1
            a = (np.random.uniform(-1, 1, size=2) if total_steps < warmup
                 else agent.act(obs_t, explore))
            obs_t2, _, ct, _, term, trunc, _ = env.step(a, evader_action=None)
            agent.store(obs_t, a, ct, obs_t2, term)
            obs_t = obs_t2
            if total_steps >= warmup:
                agent.learn()
            if term:
                captures += 1; break
            if trunc:
                break
        explore = max(0.02, explore * 0.997)
        if i % 500 == 0:
            print(f"      ph0  ep {i:4d}/{episodes} | captures {captures}")
    return agent


def train_evader(env, tracker_agent, gamma, episodes, warmup):
    tracker_agent.freeze()
    agent = DDPGAgent(gamma=gamma)
    explore, total_steps, escapes = 0.3, 0, 0
    for i in range(episodes):
        obs_t, _ = env.reset(seed=10_000 + i)
        obs_e    = env._obs_evader()
        for _ in range(env.max_steps):
            total_steps += 1
            a_t = tracker_agent.act(obs_t, 0.)
            a_e = (np.random.uniform(-1, 1, size=2) if total_steps < warmup
                   else agent.act(obs_e, explore))
            obs_t2, obs_e2, _, ce, term, trunc, _ = env.step(a_t, a_e)
            agent.store(obs_e, a_e, ce, obs_e2, term)
            obs_t, obs_e = obs_t2, obs_e2
            if total_steps >= warmup:
                agent.learn()
            if term:
                break
            if trunc:
                escapes += 1; break
        explore = max(0.02, explore * 0.997)
        if i % 500 == 0:
            print(f"      evd  ep {i:4d}/{episodes} | escapes {escapes}")
    return agent


def train_tracker_vs_evader(env, evader_agent, gamma, episodes, warmup):
    evader_agent.freeze()
    agent = DDPGAgent(gamma=gamma)
    explore, total_steps, captures = 0.3, 0, 0
    for i in range(episodes):
        obs_t, _ = env.reset(seed=i)
        obs_e    = env._obs_evader()
        for _ in range(env.max_steps):
            total_steps += 1
            a_t = (np.random.uniform(-1, 1, size=2) if total_steps < warmup
                   else agent.act(obs_t, explore))
            a_e = evader_agent.act(obs_e, 0.)
            obs_t2, obs_e2, ct, _, term, trunc, _ = env.step(a_t, a_e)
            agent.store(obs_t, a_t, ct, obs_t2, term)
            obs_t, obs_e = obs_t2, obs_e2
            if total_steps >= warmup:
                agent.learn()
            if term:
                captures += 1; break
            if trunc:
                break
        explore = max(0.02, explore * 0.998)
        if i % 500 == 0:
            print(f"      ph2  ep {i:4d}/{episodes} | captures {captures}")
    return agent


def evaluate_pair(env, tracker_agent, evader_agent, n_eval=50, seed_offset=50_000):
    captures = 0
    for k in range(n_eval):
        obs_t, _ = env.reset(seed=seed_offset + k)
        obs_e    = env._obs_evader()
        for _ in range(env.max_steps):
            a_t = tracker_agent.act(obs_t, 0.)
            a_e = evader_agent.act(obs_e, 0.) if evader_agent is not None else None
            obs_t, obs_e, _, _, term, trunc, _ = env.step(a_t, a_e)
            if term:
                captures += 1; break
            if trunc:
                break
    return captures / n_eval


def run_config(label, env_kwargs, gamma, episodes, warmup, n_eval, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    env = PursuitEnv(**env_kwargs)

    print(f"  [Phase 0] tracker vs straight evader")
    tracker0 = train_tracker(env, gamma, episodes, warmup)
    torch.save(tracker0.actor.state_dict(), f"{save_dir}/tracker0.pt")

    print(f"  [Phase 1] evader vs tracker0")
    evader1 = train_evader(env, tracker0, gamma, episodes, warmup)
    torch.save(evader1.actor.state_dict(), f"{save_dir}/evader1.pt")

    print(f"  [Phase 2] tracker2 vs evader1")
    tracker2 = train_tracker_vs_evader(env, evader1, gamma, episodes, warmup)
    torch.save(tracker2.actor.state_dict(), f"{save_dir}/tracker2.pt")

    print(f"  [Phase 3] evader3 vs tracker2")
    evader3 = train_evader(env, tracker2, gamma, episodes, warmup)
    torch.save(evader3.actor.state_dict(), f"{save_dir}/evader3.pt")

    print(f"  Evaluating ({n_eval} episodes per matchup)...")
    cross = {
        "t0_straight": evaluate_pair(env, tracker0, None,    n_eval),
        "t0_e1":       evaluate_pair(env, tracker0, evader1, n_eval),
        "t0_e3":       evaluate_pair(env, tracker0, evader3, n_eval),
        "t2_straight": evaluate_pair(env, tracker2, None,    n_eval),
        "t2_e1":       evaluate_pair(env, tracker2, evader1, n_eval),
        "t2_e3":       evaluate_pair(env, tracker2, evader3, n_eval),
    }
    arms_race = [cross["t0_straight"], cross["t0_e1"],
                 cross["t2_e1"],       cross["t2_e3"]]
    print(f"  Arms race: {[f'{x:.0%}' for x in arms_race]}")

    return {
        "config":      {"gamma": gamma, **env_kwargs},
        "arms_race":   arms_race,
        "cross_phase": cross,
    }


PHASE_LABELS = ["Ph0 trk\nvs straight",
                 "Ph0 trk\nvs Ph1 evd",
                 "Ph2 trk\nvs Ph1 evd",
                 "Ph2 trk\nvs Ph3 evd"]
PHASE_COLORS = ["steelblue", "firebrick", "steelblue", "firebrick"]


def plot_sensitivity(sweep_label, param_label, param_values, sweep_results, out_prefix):
    n_phases = 4
    n_vals   = len(param_values)
    x        = np.arange(n_phases)
    width    = 0.8 / n_vals
    cmap     = plt.cm.plasma(np.linspace(0.15, 0.85, n_vals))

    fig, ax = plt.subplots(figsize=(10, 5))
    for vi, (val, res, col) in enumerate(zip(param_values, sweep_results, cmap)):
        rates = [r * 100 for r in res["arms_race"]]
        errs  = [s * 100 for s in res["arms_race_std"]]
        offset = (vi - (n_vals - 1) / 2) * width
        ax.bar(x + offset, rates, width=width * 0.9, color=col, alpha=0.85,
               edgecolor="black", label=f"{param_label} = {val}",
               yerr=errs if any(e > 0 for e in errs) else None,
               capsize=3, error_kw={"ecolor": "black", "lw": 1.5})
    ax.set_xticks(x); ax.set_xticklabels(PHASE_LABELS, fontsize=9)
    ax.set_ylabel("Tracker capture rate (%)")
    ax.set_title(f"Sensitivity to {sweep_label}")
    ax.set_ylim(0, 115)
    ax.axhline(50, color="gray", ls="--", lw=1)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_bars.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for pi, (label, color) in enumerate(zip(PHASE_LABELS, PHASE_COLORS)):
        rates = [r["arms_race"][pi] * 100    for r in sweep_results]
        stds  = [r["arms_race_std"][pi] * 100 for r in sweep_results]
        ax.plot(param_values, rates, marker="o", color=color,
                label=label.replace("\n", " "), lw=2, ms=7)
        for vi, (pv, rate, std) in enumerate(zip(param_values, rates, stds)):
            if std > 0:  # error bars only on baseline points (multi-seed)
                ax.errorbar(pv, rate, yerr=std, fmt="none",
                            ecolor=color, capsize=4, lw=2)
    ax.set_xlabel(f"{sweep_label}")
    ax.set_ylabel("Tracker capture rate (%)")
    ax.set_title(f"Arms race — sensitivity to {sweep_label}")
    ax.set_ylim(0, 110)
    ax.axhline(50, color="gray", ls="--", lw=1)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_lines.png", dpi=120)
    plt.close(fig)

    print(f"  Saved {out_prefix}_bars.png and {out_prefix}_lines.png")


# Baseline (v_evader=8, radius=5, gamma=0.95) is repeated across 3 seeds to
# provide variance estimates; all other configs use seed 0 only.
BASELINE_ENV    = {"capture_radius": 5.0, "v_evader_max": 8.0}
BASELINE_GAMMA  = 0.95
BASELINE_SEEDS  = [0, 1, 2]

SWEEPS = [
    {
        "sweep_label": "Evader max speed (m/s)",
        "param_label": "v_evader_max",
        "values":      [6.0, 8.0, 10.0, 12.0],
        "configs": [
            ("speed_6",    {"capture_radius": 5.0, "v_evader_max":  6.0}, 0.95, 0),
            ("baseline_s0",  BASELINE_ENV, BASELINE_GAMMA, 0),
            ("baseline_s1",  BASELINE_ENV, BASELINE_GAMMA, 1),
            ("baseline_s2",  BASELINE_ENV, BASELINE_GAMMA, 2),
            ("speed_10",   {"capture_radius": 5.0, "v_evader_max": 10.0}, 0.95, 0),
            ("speed_12",   {"capture_radius": 5.0, "v_evader_max": 12.0}, 0.95, 0),
        ],
        "plot_values":  [6.0, 8.0, 10.0, 12.0],
        "plot_labels":  ["speed_6", "baseline", "speed_10", "speed_12"],
    },
    {
        "sweep_label": "Discount factor γ",
        "param_label": "gamma",
        "values":      [0.90, 0.95, 0.99],
        "configs": [
            ("gamma_090",  {"capture_radius": 5.0, "v_evader_max": 8.0}, 0.90, 0),
            ("baseline_s0", BASELINE_ENV, BASELINE_GAMMA, 0),
            ("baseline_s1", BASELINE_ENV, BASELINE_GAMMA, 1),
            ("baseline_s2", BASELINE_ENV, BASELINE_GAMMA, 2),
            ("gamma_099",  {"capture_radius": 5.0, "v_evader_max": 8.0}, 0.99, 0),
        ],
        "plot_values":  [0.90, 0.95, 0.99],
        "plot_labels":  ["gamma_090", "baseline", "gamma_099"],
    },
    {
        "sweep_label": "Capture radius (m)",
        "param_label": "capture_radius",
        "values":      [2.5, 5.0, 10.0],
        "configs": [
            ("radius_2",   {"capture_radius":  2.5, "v_evader_max": 8.0}, 0.95, 0),
            ("baseline_s0", BASELINE_ENV, BASELINE_GAMMA, 0),
            ("baseline_s1", BASELINE_ENV, BASELINE_GAMMA, 1),
            ("baseline_s2", BASELINE_ENV, BASELINE_GAMMA, 2),
            ("radius_10",  {"capture_radius": 10.0, "v_evader_max": 8.0}, 0.95, 0),
        ],
        "plot_values":  [2.5, 5.0, 10.0],
        "plot_labels":  ["radius_2", "baseline", "radius_10"],
    },
]


if __name__ == "__main__":
    EPISODES     = 1000
    WARMUP       = 1000
    N_EVAL       = 50
    RESULTS_FILE = "sensitivity/results.json"

    os.makedirs("sensitivity", exist_ok=True)

    # resume from checkpoint if interrupted
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
        print(f"Resumed: {len(all_results)} config(s) already complete.")
    else:
        all_results = {}

    for sweep in SWEEPS:
        for label, env_kwargs, gamma, seed in sweep["configs"]:
            if label in all_results:
                print(f"Skipping {label} (already done).")
                continue
            print(f"\n{'='*60}")
            print(f"CONFIG: {label}  |  {env_kwargs}  gamma={gamma}  seed={seed}")
            print(f"{'='*60}")
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            result = run_config(label, env_kwargs, gamma,
                                EPISODES, WARMUP, N_EVAL,
                                save_dir=f"sensitivity/{label}")
            result["seed"] = seed
            all_results[label] = result
            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved checkpoint → {RESULTS_FILE}")

    # average the 3 baseline seeds
    baseline_runs = [all_results[f"baseline_s{s}"]["arms_race"]
                     for s in BASELINE_SEEDS if f"baseline_s{s}" in all_results]
    if baseline_runs:
        baseline_mean = list(np.mean(baseline_runs, axis=0))
        baseline_std  = list(np.std(baseline_runs,  axis=0))
    else:
        baseline_mean = [None] * 4
        baseline_std  = [0.0]  * 4
    print(f"\nBaseline (3 seeds) — mean arms race: "
          f"{[f'{x:.0%}' for x in baseline_mean]}")
    print(f"Baseline std:  {[f'{x:.2f}' for x in baseline_std]}")

    print("\n--- Generating sensitivity plots ---")
    for sweep in SWEEPS:
        sweep_res = []
        for lbl in sweep["plot_labels"]:
            if lbl == "baseline":
                sweep_res.append({"arms_race": baseline_mean,
                                   "arms_race_std": baseline_std})
            else:
                r = all_results[lbl]
                sweep_res.append({"arms_race": r["arms_race"],
                                   "arms_race_std": [0.0] * 4})

        plot_sensitivity(
            sweep["sweep_label"],
            sweep["param_label"],
            sweep["plot_values"],
            sweep_res,
            out_prefix=f"sensitivity/sensitivity_{sweep['param_label']}",
        )

    print("\nDone. All plots saved to sensitivity/")
