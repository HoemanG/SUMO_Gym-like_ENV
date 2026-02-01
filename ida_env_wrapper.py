# ida_env_wrapper.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO


class SwitchableWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.mode = 'STEER' 
        self.helper_model = None
        self.current_obs = None
        
        # Keep your original Action Space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def set_mode(self, mode, helper_model=None):
        self.mode = mode
        self.helper_model = helper_model
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        return obs, info

    def step(self, action):
        student_val = action[0]
        
        # Get Teacher prediction
        if self.helper_model is not None:
            teacher_val, _ = self.helper_model.predict(self.current_obs, deterministic=True)
            if isinstance(teacher_val, np.ndarray): teacher_val = teacher_val.item()
        else:
            teacher_val = 0.3 if self.mode == 'STEER' else 0.5 # Default forward speed

        # --- ORIGINAL LOGIC (RESTORED) ---
        if self.mode == 'STEER':
            steer = student_val
            # Use teacher's throttle directly (no forced mapping)
            throttle = float(teacher_val) 
            
        else: # THROTTLE
            # Use student's throttle directly
            throttle = student_val
            steer = float(teacher_val)

        # DEBUG: Print what is actually being sent to SUMO
        # (This helps us see if the agent is just hitting the brakes)
        if np.random.rand() < 0.05: # Print only 5% of the time to avoid spam
            print(f"[{self.mode}] Student Output: {student_val:.3f} -> SUMO Throttle: {throttle:.3f}")

        full_action = np.array([steer, throttle], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = self.env.step(full_action)
        
        self.current_obs = next_obs
        return next_obs, reward, terminated, truncated, info
    
    def load_helper(self, model_path, algo_class=PPO):
        """
        Loads a helper model from a file path inside the subprocess.
        """
        if model_path is None:
            self.helper_model = None
            return
            
        try:
            # We load the model onto the CPU to avoid GPU conflicts in subprocesses
            self.helper_model = algo_class.load(model_path, device='cpu')
        except Exception as e:
            print(f"Error loading helper in subprocess: {e}")
            self.helper_model = None

    def set_mode(self, mode, helper_model=None):
        """
        Updated set_mode. 
        Note: We ignore helper_model arg here if we are using the 
        parallel 'load_helper' method, but keep it for backward compatibility.
        """
        self.mode = mode
        if helper_model is not None:
            self.helper_model = helper_model
        # If helper_model is None, we assume it was already loaded via load_helper()