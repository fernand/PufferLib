import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.td import binding

# Observation dimension: 6 + 2*TD_MAX_TOWERS + 3*TD_MAX_AGENTS = 46
OBS_DIM = 6 + 2 * 5 + 3 * 10

class TD(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, num_agents=10, width=30, height=30,
                 render_mode=None, log_interval=1, buf=None, seed=42):
        # Define native spaces
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        # Total number of agents across all envs
        self.num_agents = num_envs * num_agents
        self.log_interval = log_interval

        super().__init__(buf)
        # Initialize C environments
        c_envs = []
        for i in range(num_envs):
            offset = i * num_agents
            c_env = binding.env_init(
                self.observations[offset:offset + num_agents],
                self.actions[offset:offset + num_agents],
                self.rewards[offset:offset + num_agents],
                self.terminals[offset:offset + num_agents],
                self.truncations[offset:offset + num_agents],
                seed,
                width=width,
                height=height,
                num_agents=num_agents,
            )
            c_envs.append(c_env)
        # Combine into a vectorized environment
        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=42):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        # Copy actions and step
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == "__main__":
    # Simple performance test
    TIME = 10
    env = TD(num_envs=512)
    actions = np.random.randint(0, env.single_action_space.n, env.num_agents)
    env.reset()
    import time
    steps = 0
    start = time.time()
    while time.time() - start < TIME:
        env.step(actions)
        steps += env.num_agents
    print("C M SPS:", steps / (1e6 * (time.time() - start)))
    env.close()