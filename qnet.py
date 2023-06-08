# TODO: Make the default fully connected DNN

# TODO: Experiment with an RNN

# Input should have a size of (batch_size, num_letters_in_alphabet, num_states, ((num_guesses * word_length) + 1))
# which is ... ->             (batch_size,                      26,          4,                                31)

#
# Keyboard: [a, b, c, ... x, y, z] (one hot encoded vector of size 26x4)
# First Guess: [a, b, c, ... x, y, z] (one hot encoded vector of size 26x6x4)
# Second Guess: [a, b, c, ... x, y, z] (one hot encoded vector of size 26x6x4)
# Third Guess: [a, b, c, ... x, y, z] (one hot encoded vector of size 26x6x4)
# Fourth Guess: [a, b, c, ... x, y, z] (one hot encoded vector of size 26x6x4)
# Fifth Guess: [a, b, c, ... x, y, z] (one hot encoded vector of size 26x6x4)
# Sixth Guess: [a, b, c, ... x, y, z] (one hot encoded vector of size 26x6x4)
#
#
# [[a: 0, 0, 0, 1], b, c, ... x, y, z]
#


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(
        self,
        device="cpu",
        num_letters=26,
        num_state=4,
        num_guesses=6,
        word_length=5,
        layer1_size=2048,
        layer2_size=1024,
        layer3_size=512,
    ):
        super(DQN, self).__init__()

        self.device = device

        self.layer1 = nn.Linear(
            num_letters * num_state * ((num_guesses * word_length) + 1), layer1_size
        )
        self.batch_norm1 = nn.BatchNorm1d(layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.batch_norm2 = nn.BatchNorm1d(layer2_size)
        self.layer3 = nn.Linear(layer2_size, layer3_size)
        self.batch_norm3 = nn.BatchNorm1d(layer3_size)
        self.output_layer = nn.Linear(layer3_size, num_letters * word_length)

    def forward(self, state: np.ndarray):
        if len(state.shape) == 1:
            state = np.expand_dims(state, 0)
        # state = self.encode_state(board)
        state_tensor = torch.from_numpy(state).float()
        layer1_output = F.relu(self.batch_norm1(self.layer1(state_tensor)))
        layer2_output = F.relu(self.batch_norm2(self.layer2(layer1_output)))
        layer3_output = F.relu(self.batch_norm3(self.layer3(layer2_output)))
        action_output = F.relu(self.output_layer(layer3_output))
        return action_output


class DQRNN(nn.Module):
    def __init__(self):
        super(DQRNN, self).__init__()

    def encode_state(self, batch: np.ndarray):
        pass

    def forward(self, game: np.ndarray):
        pass
