import torch
import torch.nn as nn
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from noisy_net import NoisyLinear

class DQNModel(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_config:dict, name:str):
        """
        My custom model for DQN learning of playing Pong.
        Implemented NoisyNet paper into last 2 layers of the net for enhanced training

        :param observation_space: Current state of the game at each timestamp
        :param action_space: Available actions to take by the agent (6 in Pong)
        :param num_outputs: Number of output neurons - 6 discrete actions
        :param model_config: Model configuration, policy, number of workers, etc.
        :param name: Model's name used by rllib
        """
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        input_shape = observation_space.shape  # 84 x 84 x 4
        input_shape = input_shape[-1]  # Only taking channel value

        # ReLu activation
        self.relu = nn.ReLU()

        # Conv layers + batch norm
        self.conv1 = nn.Conv2d(input_shape, 32, 8, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.bn3 = nn.BatchNorm2d(64)

        # FC Network
        self.fc1 = NoisyLinear(7 * 7 * 64, 512)
        self.fc_bn = nn.BatchNorm1d(512)
        self.fc2 = NoisyLinear(512, num_outputs)

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )


    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()

    def forward(self, input_dict, state, seq_lens):
        """
        Passing the observations
        """
        # Observation dict is of shape: torch.Size([32, 84, 84, 4])
        # Permute the matrix, add normalization
        x = input_dict['obs'].permute(0, 3, 1, 2).float() / 255.0

        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))

        x = x.reshape(x.size(0), -1)
        x = self.fc_bn(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x, state
