import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Stable Baselines 3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

# Matplotlib imports for 3D animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# ======================================
#correu mal a reward tava uma merda

#=========================================
# ==========================================
# PART 1: ENVIRONMENT SETUP
# ==========================================
class PursuitEnv(gym.Env):
    def __init__(self):
        super(PursuitEnv, self).__init__()
        self.dt = 0.1
        self.max_steps = 1000
        self.capture_radius = 1.0

        self.v_evader_max = 10.0
        self.v_tracker_max = 12.0
        self.g_max = 5.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # spawn tracker
        self.p_tracker = np.array([0.0, 0.0, 0.0])
        self.v_tracker = np.array([self.v_tracker_max, 0.0, 0.0])

        # spawn evader (randomly between 50 and 100 meters away)
        self.p_evader = self.np_random.uniform(50, 100, size=3)
        direction = self.p_evader / np.linalg.norm(self.p_evader)
        self.v_evader = direction * self.v_evader_max

        return self._get_obs(), {}
    
    def step(self, action):
        self.current_step += 1

        # 1. Move Evader
        self.p_evader += self.v_evader * self.dt

        # 2. Steer Tracker
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
        
        # 3. Check Status and Rewards
        obs = self._get_obs()
        distance = np.linalg.norm(self.p_evader - self.p_tracker)
        
        terminated = False
        truncated = False
        reward = 0.0
        
        if distance < self.capture_radius:
            terminated = True
            reward = 100.0  # Big reward for catching!
        elif self.current_step >= self.max_steps:
            truncated = True
            reward = -10.0  # Penalty for taking too long
        else:
            reward = -0.1   # Small penalty every step to encourage speed
            
        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self):
        rel_pos = self.p_evader - self.p_tracker
        rel_vel = self.v_evader - self.v_tracker
        return np.concatenate([rel_pos, rel_vel]).astype(np.float32)

# ==========================================
# PART 2: INITIALIZATION & TRAINING
# ==========================================
if __name__ == "__main__":
    env = PursuitEnv()

    print("Checking environment...")
    check_env(env)
    print("Environment check passed!\n")

    print("Initializing PPO AI Agent...")
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=0.0003, 
        gamma=0.99, 
        verbose=1
    )

    TOTAL_TIMESTEPS = 100000
    print(f"\n--- Starting Training for {TOTAL_TIMESTEPS} steps ---")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("ppo_tracker_model")
    print("\nTraining complete and model saved as 'ppo_tracker_model.zip'!")

    # ==========================================
    # PART 3: EVALUATION
    # ==========================================
    print("\nEvaluating trained policy...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Average Reward over 10 test flights: {mean_reward:.2f} +/- {std_reward:.2f}")

    # ==========================================
    # PART 4: SINGLE SIMULATION & 3D ANIMATION
    # ==========================================
    print("\nRunning the final simulation for 3D animation...")
    obs, info = env.reset()

    tracker_history = []
    evader_history = []

    for step in range(1000):
        # Save positions
        tracker_history.append(env.p_tracker.copy())
        evader_history.append(env.p_evader.copy())
        
        # Agent decides and environment steps
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"--> Target Captured at step {step}!")
            break
        elif truncated:
            print("--> Time ran out! Target escaped.")
            break

    tracker_history = np.array(tracker_history)
    evader_history = np.array(evader_history)

    print("\nGenerating 3D Animation. Close the plot window to end the script.")
    
    # Setup Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Trained AI: 3D Pursuit-Evasion Animation', fontsize=16)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')

    # Lock camera axes
    all_x = np.concatenate((tracker_history[:, 0], evader_history[:, 0]))
    all_y = np.concatenate((tracker_history[:, 1], evader_history[:, 1]))
    all_z = np.concatenate((tracker_history[:, 2], evader_history[:, 2]))
    ax.set_xlim([np.min(all_x), np.max(all_x)])
    ax.set_ylim([np.min(all_y), np.max(all_y)])
    ax.set_zlim([np.min(all_z), np.max(all_z)])

    # Empty plot elements
    tracker_line, = ax.plot([], [], [], color='blue', linewidth=2, label='Tracker Path')
    evader_line, = ax.plot([], [], [], color='red', linestyle='--', linewidth=2, label='Evader Path')
    tracker_point, = ax.plot([], [], [], marker='o', color='blue', markersize=8)
    evader_point, = ax.plot([], [], [], marker='o', color='red', markersize=8)
    
    ax.legend()

    # Animation update loop
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

    ani = animation.FuncAnimation(fig, update, frames=len(tracker_history), interval=30, blit=False)
    plt.show()