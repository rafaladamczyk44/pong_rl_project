import gymnasium as gym
import ale_py

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://medium.com/@joachimiak.krzysztof/learning-to-play-pong-with-pytorch-tianshou-a9b8d2f1b8bd
# https://www.youtube.com/watch?v=tsWnOt2OKx8
# http://karpathy.github.io/2016/05/31/rl/


# 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT, 4: DOWNRIGHT, 5: DOWNLEFT

ale = ale_py.ALEInterface()
env = gym.make('Pong-v4', render_mode='human')
state, info = env.reset()


done = False
while not done:
    action = env.action_space.sample()  # Choose a random action
    state, reward, done, truncated, info = env.step(action)
    print(info)

env.close()