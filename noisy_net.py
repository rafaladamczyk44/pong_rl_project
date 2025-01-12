import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    """
    Implementation of Noisy Net from https://arxiv.org/abs/1706.10295v3

    Replaces a standard linear layer and the end of the main model by adding noise to weights and biases.
    Enhances exploration by allowing net to learn the noise needed for exploration instead of random exploration
    with epsilon greedy algorithm.

    :param in_features: Size of the input
    :param out_features: Size of the output
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable mean params
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))

        # Sigma parameter for noise scaling
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

        # Buffers for noise layers
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Initializing the net params.
        _mu params are initialized with uniform distrubution
        _sigma params are initialized as constant - std_init/√in_features
        """

        # -1/√n, 1/√n
        uniform_dist = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-uniform_dist, uniform_dist)
        self.bias_mu.data.uniform_(-uniform_dist, uniform_dist)

        # std_init / √in_features
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))


    def _scale_noise(self, size:int):
        """
        Generate scaled noise
        Transformation ensures no zero-mean noise.

        :param size:
        :return:
        """
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """
        Generate noise: a product between two noise vectors instead of full noise matrix
        """
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)

        # Combine by using outer prod
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        # For biases - use output noise directly
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        """
        Forward pass method of the layer.
        From the paper:

        During training:
            weight = μw + σw × εw
            bias = μb + σb × εb

        During evaluation:
            weight = μw
            bias = μb

        :param x: Input tensor
        :return: output tensor after noisy linear layer transformation
        """

        if self.training:
            # During training, combine mean parameters with scaled noise
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # During evaluation, use only mean parameters
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)
