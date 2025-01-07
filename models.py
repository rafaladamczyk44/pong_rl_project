import torch
import torch.nn as nn
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class DQNModel(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_config:dict, name:str):
        """
        My custom model for DQN learning of playing Pong

        :param observation_space: Current state of the game at each timestamp
        :param action_space: Available actions to take by the agent (6 in Pong)
        :param num_outputs: Number of output neurons
        :param model_config: Model configuration, policy, number of workers, etc.
        :param name: Model's name used by rllib
        """
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Activation for all layers
        self.relu = nn.ReLU()

        # Define CNN layers
        input_shape = observation_space.shape  # 84 x 84 x 4
        print(input_shape[-1])
        input_shape = input_shape[-1]  # Only taking channel value
        self.conv1 = nn.Conv2d(input_shape, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        # FC Network
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, out_features=num_outputs)

    def forward(self, input_dict, state, seq_lens):
        """
        Passing the observations
        Observation dict is of shape: torch.Size([32, 84, 84, 4])
        Need to permute it to batch, channel, height, width for Conv layers
        """
        # print(input_dict['obs'].size())
        x = input_dict['obs'].permute(0, 3, 1, 2).float() # Permuting for correct shape: batch, channels, height, width
        # print(x.size())

        # Convolution layers pass
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # print(f'Shape before reshape: {x.size()}')
        x = x.reshape(x.size(0), -1) # Flatten
        # print(f'Shape after reshape: {x.size()}')

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x, state
