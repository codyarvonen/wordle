import unittest
import game


class TestGameFunctions(unittest.TestCase):
    def setUp(self):
        # self.test_game = game.Game()
        pass

    def test_win(self):
        pass

    def test_loss(self):
        pass

    def test_invalid_word(self):
        pass

    def test_repear_letters(self):
        # Test double letters
        test_wordles = [
            "llama",
            "allow",
            "polls",
            "spill",
            "lulus",
            "slily",
            "salol",
            "ladle",
            "algal",
            "label",
        ]
        for wordle in test_wordles:
            self.test_game = game.Game()

        # Test triple letters
        test_wordles = ["lulls", "added", "abaca"]
        for wordle in test_wordles:
            self.test_game = game.Game()

    def test_step(self):
        pass


if __name__ == "__main__":
    unittest.main()
