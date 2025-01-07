import os
import gymnasium as gym
import ale_py
from ray.tune import register_env
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog
from sympy.abc import epsilon

# Custom model
from models import DQNModel
from image_processing import ImageWrapper  # Custom wrapper for image processing

# 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT, 4: DOWNRIGHT, 5: DOWNLEFT

# Set up the game
GAME = 'ale_py:ALE/Pong-v5'
ale = ale_py.ALEInterface()

# Register the custom wrapper
def create_wrapped_env(config):
    env = gym.make(GAME)
    return ImageWrapper(env)

register_env(GAME, create_wrapped_env)

# Register model in rllib
ModelCatalog.register_custom_model('DQN_Model', DQNModel)

# Policy configuration
config = (
    DQNConfig() # Policy: DQN
    .environment(GAME)
    .framework('torch')
    .api_stack( # Using old ray API stack, so I can use custom model
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
    .env_runners(num_env_runners=2)
    .training(
        model={'custom_model': 'DQN_Model'},
        gamma=0.99,
        lr=3e-4,
        dueling=True,
        double_q=True, # Use double Q-learning
        epsilon=[[0, 1.0], [50000, 0.1], [100000, 0.05]], # Epsilon decay
        train_batch_size=32,
        replay_buffer_config={
            "type": "MultiAgentPrioritizedReplayBuffer",  # Compatible replay buffer
            "capacity": 50000, # buffer capacity
            "prioritized_replay_alpha": 0.6,  # Alpha for prioritization
            "prioritized_replay_beta": 0.4,  # Beta for importance-sampling weights
            "prioritized_replay_eps": 1e-6,  # Small epsilon to prevent zero probability
        },
    )
)

# Build the actual agent
agent = config.build()

for epoch in range(100):
    result = agent.train()
    print(pretty_print(result))

    # Log rewards and losses
    avg_reward = result['env_runners']['episode_reward_mean']
    avg_loss = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("loss",
                                                                                                                "N/A")
    print(f"Epoch {epoch}, Average Reward: {avg_reward}, Loss: {avg_loss}")
    print(f"Replay Buffer Size: {agent.local_replay_buffer._num_added}")

