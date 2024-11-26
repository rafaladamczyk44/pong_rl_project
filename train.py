import gymnasium as gym
import ale_py
from ray.tune import register_env
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog

from models import DQNModel
# 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT, 4: DOWNRIGHT, 5: DOWNLEFT

# TODO: Make wrapper for image processing
# TODO: Write DQN Model

# Set up the game
GAME = 'Pong-v4'
ale = ale_py.ALEInterface()

# Register model in rllib
ModelCatalog.register_custom_model('DQN_Model', DQNModel)

# Register env in rllib
register_env(GAME, lambda config: gym.make(GAME))

# Model configuration
config = (
    DQNConfig() # Policy: DQN
    .environment(GAME)
    .framework('torch')
    .env_runners(num_env_runners=2)
    .training(
        model={'custom_model': 'DQN_Model'},
        gamma=0.99,
        lr=3e-4,
        dueling=True,
        double_q=True, # Use double Q-learning
        replay_buffer_config={'capacity': 50000},
    )
)
