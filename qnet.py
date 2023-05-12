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
        self
    ):
        super(DQN, self).__init__()

    def encode_state(self, batch: np.ndarray):
        pass

    def forward(self, board: np.ndarray):
        pass


class DQRNN(nn.Module):
    def __init__(
        self
    ):
        super(DQRNN, self).__init__()

    def encode_state(self, batch: np.ndarray):
        pass

    def forward(self, game: np.ndarray):
        pass
