import os
from datetime import datetime
import gymnasium as gym
import ale_py
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.tune.logger import pretty_print

# Custom model
from models import DQNModel
from image_processing import ImageWrapper  # Custom wrapper for image processing

# Register custom model in rllib
ModelCatalog.register_custom_model('DQN_Model', DQNModel)

# Set up the game
GAME = 'ale_py:ALE/Pong-v5'
ale = ale_py.ALEInterface()

# Register the custom wrapper
def create_wrapped_env(config):
    env = gym.make(GAME)
    return ImageWrapper(env)

register_env(GAME, create_wrapped_env)

'''
Actions: 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT, 4: DOWNRIGHT, 5: DOWNLEFT
DQN Policy configuration
    gamma: discount factor
    lr: learning rate
    dueling: enable dueling network (separate value and advantage streams)
    double_q: use double Q-learning to improve q-value estimation
    epsilon: epsilon-greedy exploration - decay from 1.0 to 0.1 over 100k steps, then to 0.01 over 400k steps
    type: MultiAgentPrioritizedReplayBuffer: prioritized replay buffer
    capacity: max capacity of transitions
    prioritized_replay_alpha: how much prioritization is used (0: uniform sampling, 1: full prioritization)
    prioritized_replay_beta: importance sampling exponent
    prioritized_replay_eps: small value to avoid division by zero
    target_network_update_freq: how often to update the target network
    grad_clip: clip gradients to this value
    num_steps_sampled_before_learning_starts: buffer before it starts learning - counting loss value
'''

# Env config
BATCH_SIZE = 64
ENV_RUNNERS = 4
BUFFER_CAPACTIY = 100000

# DQN config
discount_factor = 0.99
learning_rate = 1e-4
epsilon_decay_rate = [[0, 1.0], [100000, 0.1], [500000, 0.01]]

config = (
    DQNConfig()
        .environment(GAME)
        .framework('torch')
        .api_stack( # Using old ray API stack, so I can use custom model
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False
            )
        .env_runners(num_env_runners=ENV_RUNNERS)
        .training(
            model={'custom_model': 'DQN_Model'},
            gamma=discount_factor,
            lr=learning_rate,
            dueling=True,
            double_q=True,
            epsilon=epsilon_decay_rate,
            train_batch_size=BATCH_SIZE,
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": BUFFER_CAPACTIY,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
            },
            target_network_update_freq=1000,
            grad_clip=1.0,
            num_steps_sampled_before_learning_starts=10000
        )
)

agent = config.build()

# For tracking progress
best_reward = float('-inf')
running_rewards = []

for epoch in range(1000):
    result = agent.train()

    # Extract key metrics
    avg_reward = result['env_runners']['episode_reward_mean']
    avg_loss = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("loss",
                                                                                                                "N/A")

    # Get epsilon - fixed method
    # epsilon = agent.get_policy().exploration.epsilon
    q_values = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get(
        "mean_q", "N/A")

    # Keep track of last 10 rewards for running average
    running_rewards.append(avg_reward)
    if len(running_rewards) > 10:
        running_rewards.pop(0)
    running_avg = sum(running_rewards) / len(running_rewards)

    # Track best performance
    if avg_reward > best_reward:
        best_reward = avg_reward
        print(f"\n>>> New best reward: {best_reward:.2f} <<<\n")

    if epoch % 100 == 0:
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        checkpoint_path = f"checkpoints/chkpt_e{epoch}_{timestamp}"
        try:
            agent.save(checkpoint_path)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    # Print metrics
    print(f"\nEpoch {epoch}")
    print(f"Avg. reward: {avg_reward:.2f}")
    print(f"Running Avg (10 epochs): {running_avg:.2f}")
    print(f"Loss: {avg_loss}")
    print(f"Mean Q-Value: {q_values}")
    print(f"Buffer Size: {agent.local_replay_buffer._num_added}")
    print(f"Best Reward So Far: {best_reward:.2f}")
    print("---------------")


# Saving the model
try:
    final_model_path = f"models/final_model_{datetime.now().strftime('%d%m%Y_%H%M%S')}"
    agent.save(final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
except Exception as e:
    print(f"Error saving final model: {e}")