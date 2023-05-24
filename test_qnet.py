import unittest
import game
import qnet
import numpy as np


class TestQNetFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def test_forward(self):
        self.test_game = game.Game(visualize=False)
        self.test_game.step("flame")
        test_encode_before = self.test_game.game_data.get_game_state()
        self.test_game.step("brick")
        test_encode_batch_before = np.concatenate(
            (
                test_encode_before[np.newaxis, :],
                self.test_game.game_data.get_game_state()[np.newaxis, :],
            ),
            axis=0,
        )

        print(test_encode_before.shape, test_encode_batch_before.shape)

        net = qnet.DQN()

        net.eval()
        net(test_encode_before)

        net.train()
        net(test_encode_batch_before)

        total_params = sum(p.numel() for p in net.parameters())
        print("Total parameters: ", total_params)

        # rnn_net = qnet.DQRNN()

        # rnn_net.eval()
        # rnn_net(test_encode_before)

        # rnn_net.train()
        # rnn_net(test_encode_batch_before)

        # total_params = sum(p.numel() for p in rnn_net.parameters())
        # print("Total parameters: ", total_params)


if __name__ == "__main__":
    unittest.main()
