import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your custom environment
from simulation.sumo_env import SumoEnv

# --- CONFIGURATION ---
MAP_CONFIG = "maps/TestMap/osm.sumocfg"
LOG_DIR = "./reports"
MODEL_DIR = "./models"
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "highest_reward")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# Hyperparameters
TOTAL_TIMESTEPS = 100_000
LEARNING_RATE = 0.0003
N_STEPS = 2048
BATCH_SIZE = 64

# Create directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class DetailedLogCallback(BaseCallback):
    """
    Custom callback to print detailed stats and episode summaries.
    """
    def __init__(self, verbose=0):
        super(DetailedLogCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_ep_reward = 0

    def _on_step(self) -> bool:
        # 1. GET DATA FROM LOCAL VARIABLES
        # SB3 stores current variables in self.locals
        
        # 'dones' indicates if the episode finished in this step
        dones = self.locals['dones'][0]
        # 'rewards' is the reward for the LAST action taken
        reward = self.locals['rewards'][0]
        # 'actions' is the action the agent JUST took
        action = self.locals['actions'][0]
        # 'new_obs' is the state AFTER the action
        obs = self.locals['new_obs'][0]
        
        self.current_ep_reward += reward

        # 2. CHECK IF EPISODE FINISHED
        if dones:
            # When done is True, 'infos' usually contains the terminal info
            infos = self.locals['infos'][0]
            
            # SB3 Monitor wrapper adds 'episode' key with {r: reward, l: length, t: time}
            if 'episode' in infos:
                ep_r = infos['episode']['r']
                ep_l = infos['episode']['l']
                print(f"\n>>> EPISODE FINISHED <<< | Total Reward: {ep_r:.2f} | Steps: {ep_l}")
                print("-" * 60)
            
            # Reset tracker
            self.current_ep_reward = 0

        # 3. PRINT DETAILED STATS (Every 10 steps)
        if self.n_calls % 10 == 0:
            # Extract specific features from your 20-dim Observation
            # Index 0: Speed (Normalized)
            # Index 1: Accel
            # Index 2: Energy
            # Index 3: Lane
            speed = obs[0]
            energy = obs[2]
            
            # Un-normalize for readability if you want (Optional)
            # real_speed = speed * 30.0 
            
            print(f"Step {self.n_calls:06d} | Act: {action} | Rew: {reward:6.2f} | "
                  f"Spd: {speed:.2f} | Egy: {energy:.2f}")

        return True


def make_env(rank: int, seed: int = 0):
    def _init():
        env = SumoEnv(
            render=False, # Make sure this is False for training speed!
            map_config=MAP_CONFIG,
            TRAFFIC_SCALE=5.0
        )
        env = Monitor(env, filename=os.path.join(LOG_DIR, str(rank)))
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    print("--- STARTING TRAINING SCRIPT ---")
    print(f"Device: CPU (Forced)")

    # 1. Environment
    env = DummyVecEnv([make_env(rank=0)])
    eval_env = DummyVecEnv([make_env(rank=1)])

    # 2. Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=2000, # Check best model every 2000 steps
        deterministic=True,
        render=False
    )
    
    log_callback = DetailedLogCallback()
    
    callback_list = CallbackList([eval_callback, log_callback])

    # 3. Model (Force CPU for efficiency on small inputs)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0, # Set to 0 to suppress default SB3 prints, we use ours
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        device='cpu' 
    )

    print(f"Training for {TOTAL_TIMESTEPS} steps...")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list)
        model.save(os.path.join(MODEL_DIR, "ppo_sumo_final"))
        print("Training Finished.")

    except KeyboardInterrupt:
        print("\n\n!!! INTERRUPTED !!!")
        model.save(os.path.join(CHECKPOINT_DIR, "ppo_sumo_interrupted"))
        print("Model Saved. Exiting.")
        env.close()

if __name__ == "__main__":
    main()