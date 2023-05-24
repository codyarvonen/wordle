import unittest
import game

from letter_state import LetterState


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

    def test_repeat_letters(self):
        # Test double letters
        test_wordles = [
            "llama",
            "llama",
            "llama",
            "allow",
            "allow",
            "allow",
            "polls",
            "polls",
            "polls",
            "spill",
            "spill",
            "spill",
            "lulus",
            "lulus",
            "lulus",
            "slily",
            "slily",
            "slily",
            "salol",
            "salol",
            "salol",
            "ladle",
            "ladle",
            "ladle",
            "algal",
            "algal",
            "algal",
            "label",
            "label",
            "label",
        ]
        test_words = [
            "falls",
            "stall",
            "salol",
            "stall",
            "ladle",
            "label",
            "llama",
            "algal",
            "label",
            "llama",
            "allow",
            "lulus",
            "stall",
            "algal",
            "slily",
            "lulus",
            "salol",
            "label",
            "llama",
            "ladle",
            "slily",
            "allow",
            "algal",
            "salol",
            "lulus",
            "ladle",
            "falls",
            "allow",
            "falls",
            "slily",
        ]
        test_words_2 = [
            "allow",
            "lulus",
            "slily",
            "llama",
            "lulus",
            "slily",
            "allow",
            "stall",
            "ladle",
            "falls",
            "salol",
            "label",
            "llama",
            "allow",
            "falls",
            "llama",
            "stall",
            "falls",
            "allow",
            "label",
            "lulus",
            "llama",
            "stall",
            "slily",
            "llama",
            "allow",
            "salol",
            "llama",
            "stall",
            "ladle",
        ]
        for i in range(len(test_wordles)):
            self.test_game = game.Game(visualize=False, wordle_word=test_wordles[i])
            self.test_game.step(test_words[i])
            num_yellows = 0
            for letter in self.test_game.game_data.guesses[0]:
                if letter.get("l") is LetterState.YELLOW:
                    # if LetterState.YELLOW in letter.values():
                    num_yellows += 1
            self.assertTrue(
                num_yellows == 2,
                f"Guess {test_words[i]} for wordle {test_wordles[i]} had {num_yellows} yellows instead of 2",
            )
            num_yellows = 0

            self.test_game.step(test_words_2[i])
            for letter in self.test_game.game_data.guesses[1]:
                if letter.get("l") is LetterState.YELLOW:
                    # if LetterState.YELLOW in letter.values():
                    num_yellows += 1
            self.assertTrue(
                num_yellows == 1,
                f"Guess {test_words_2[i]} for wordle {test_wordles[i]} had {num_yellows} yellows instead of 1",
            )
            num_yellows = 0

        # Test triple letters
        # test_wordles = ["lulls", "added", "abaca"]
        # for wordle in test_wordles:
        #     self.test_game = game.Game()

    def test_step(self):
        pass


if __name__ == "__main__":
    unittest.main()
