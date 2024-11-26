import gymnasium as gym
import ale_py

env = gym.make('Pong-v4', render_mode='human')

state, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Choose a random action
    state, reward, done, truncated, info = env.step(action)

env.close()