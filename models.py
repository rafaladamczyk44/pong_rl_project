import torch
import torch.nn as nn
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class DQNModel(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Activation for all layers
        self.relu = nn.ReLU()
        # TODO: Define CNN layers

        # TODO: FC Network

    def forward(self, input_dict, state, seq_lens):
        """Passing the observations"""
        pass
