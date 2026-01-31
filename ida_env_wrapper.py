# ida_env_wrapper.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SwitchableWrapper(gym.Wrapper):
    """
    A wrapper that allows hot-swapping between Steering Training and Throttle Training
    without resetting the environment.
    """
    def __init__(self, env):
        super().__init__(env)
        self.mode = 'STEER' # Options: 'STEER' or 'THROTTLE'
        self.helper_model = None
        self.current_obs = None
        
        # Both agents see the same observation (24,), but output only 1 action (Steer OR Throttle)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def set_mode(self, mode, helper_model=None):
        """Switch the active agent. Pass helper_model=None to use constant fallback."""
        self.mode = mode
        self.helper_model = helper_model
        
    def reset(self, **kwargs):
        """Standard reset, tracks the observation."""
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        return obs, info

    def step(self, action):
        # 'action' comes from the Student
        student_val = action[0]
        
        # Get Teacher prediction
        if self.helper_model is not None:
            teacher_val, _ = self.helper_model.predict(self.current_obs, deterministic=True)
            if isinstance(teacher_val, np.ndarray): teacher_val = teacher_val.item()
        else:
            teacher_val = 0.3 if self.mode == 'STEER' else 0.0

        # --- FIX: PREVENT THE "PARKING STRATEGY" ---
        if self.mode == 'STEER':
            steer = student_val
            # Even if the Throttle Agent wants to stop (-1.0), force it to 0.1
            # This ensures the Steering Agent always has forward momentum to learn from.
            throttle = max(float(teacher_val), 0.1) 
            
        else: # THROTTLE
            throttle = student_val
            steer = float(teacher_val)

        # Send to SUMO
        full_action = np.array([steer, throttle], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = self.env.step(full_action)
        
        self.current_obs = next_obs
        return next_obs, reward, terminated, truncated, info