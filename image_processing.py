# TODO: Image processing class
import cv2
from gymnasium import Wrapper
from gymnasium.spaces import Box
import numpy as np
import collections

class ImageWrapper(Wrapper):
    def __init__(self, env, shape=(84, 84), frame_stack=4):
        super(ImageWrapper, self).__init__(env)
        self.shape = shape
        self.frame_stack = frame_stack
        self.frames = collections.deque(maxlen=frame_stack)

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.shape[0], self.shape[1], frame_stack),
            dtype=np.uint8
        )

    def observation(self, obs):
        # Convert to grayscale
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize the image
        resized_obs = cv2.resize(gray_obs, self.shape, interpolation=cv2.INTER_AREA)
        # Add the frame to the deque
        self.frames.append(resized_obs)
        # Stack frames and return the observation
        stacked_obs = np.stack(list(self.frames), axis=-1)  # Shape: (84, 84, frame_stack)
        return stacked_obs