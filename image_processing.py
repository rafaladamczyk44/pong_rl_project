# TODO: Image processing class
import cv2
from gymnasium import Wrapper
from gymnasium.spaces import Box
import numpy as np
import collections

class ImageWrapper(Wrapper):
    """
    Customer image wrapper for pong
    """
    def __init__(self, env, shape=(84, 84), frame_stack=4, skip_frames=4):
        super(ImageWrapper, self).__init__(env)
        self.shape = shape
        self.frame_stack = frame_stack
        # Skipping frames implementation to save comp power
        self.skip_frames = skip_frames
        self.frames = collections.deque(maxlen=frame_stack)

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.shape[0], self.shape[1], frame_stack),
            dtype=np.uint8
        )

    def step(self, action):
        """
        Execute action and skip frames while accumulating rewards
        """
        total_reward = 0
        terminated = False
        truncated = False
        info = {}

        # Execute the same action for skip_frames steps
        for _ in range(self.skip_frames):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            # Break if the episode ended either way
            if terminated or truncated:
                break

        # Process the last observed frame
        processed_obs = self.observation(obs)
        return processed_obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment to start a new episode
        Also needed to be updated for new Gymnasium API
        """
        obs, info = self.env.reset(**kwargs)
        processed_obs = self.observation(obs)
        return processed_obs, info

    def observation(self, obs):

        """
        Process observation and change to correct form
        :param obs: Take a single observation from the game
        :return: Processed observation
        """
        # Convert to grayscale
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize the image
        resized_obs = cv2.resize(gray_obs, self.shape, interpolation=cv2.INTER_AREA)
        # Add the frame to the deque
        self.frames.append(resized_obs)

        # If we don't have enough frames yet, duplicate the current frame
        while len(self.frames) < self.frame_stack:
            self.frames.append(resized_obs)

        # Stack frames and return the observation
        stacked_obs = np.stack(list(self.frames), axis=-1)  # Shape: (84, 84, frame_stack)
        return stacked_obs