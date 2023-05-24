import enum
import time
import string
import pickle
import pygame
import numpy as np
from typing import Tuple

from constants import *
from letter_state import LetterState
from word_data import is_valid, select_random_word
from reward import ActionReward


class GameStatus(enum.Enum):
    GAME_OVER = 0
    INVALID_WORD = 1
    INVALID_LENGTH = 2
    VALID_WORD = 3
    GAME_WON = 4


class GameData:
    def __init__(self, word_length: int, num_guesses: int, wordle_word: str = None):
        self.word_length = word_length
        self.num_guesses = num_guesses
        self.wordle_word = (
            wordle_word
            if wordle_word is not None
            else select_random_word(WORD_DATA_FILE)
        )
        self.keyboard = dict.fromkeys(string.ascii_lowercase, LetterState.EMPTY)
        self.guesses = [
            [
                dict.fromkeys(string.ascii_lowercase, LetterState.EMPTY)
                for _ in range(word_length)
            ]
            for _ in range(num_guesses)
        ]

    def get_game_state(self) -> np.ndarray:
        encoded_guesses = np.array(
            [
                [
                    [
                        [1 if letter_state == state else 0 for state in LetterState]
                        for letter_state in letter_dict.values()
                    ]
                    for letter_dict in guess
                ]
                for guess in self.guesses
            ]
        )
        encoded_keyboard = np.array(
            [
                [1 if letter_state == state else 0 for state in LetterState]
                for letter_state in self.keyboard.values()
            ]
        )
        # return np.concatenate(
        #     [
        #         encoded_keyboard[np.newaxis, :],
        #         encoded_guesses.reshape(
        #             -1, len(string.ascii_lowercase), len(LetterState)
        #         ),
        #     ],
        #     axis=0,
        # )
        return np.concatenate(
            [
                encoded_keyboard.flatten(),
                encoded_guesses.flatten(),
            ],
            axis=0,
        )

    def make_guess(self, guess: str) -> GameStatus:
        if len(guess) != self.word_length:
            return GameStatus.INVALID_LENGTH
        if not is_valid(WORD_DATA_FILE, guess):
            return GameStatus.INVALID_WORD

        # TODO: Write test functions to verify edge cases
        # TODO: Fix yellow tile bug

        # guess:  hello
        # wordle: world

        word_dict = {}
        unmatched_guess = []

        # Build a dictionary of letter counts for the actual word
        for i, letter in enumerate(self.wordle_word):
            word_dict[letter] = word_dict.get(letter, 0) + 1

        # Iterate over the guess and determine the color of each letter
        for i, letter in enumerate(guess):
            if letter == self.wordle_word[i]:
                # Correct letter in the correct position
                state = LetterState.GREEN
                word_dict[letter] -= 1
                self.guesses[-self.num_guesses][i][guess[i]] = state
                self.keyboard[guess[i]] = state
                unmatched_guess.append("_")
            else:
                unmatched_guess.append(letter)

        # Iterate over the unmatched letters in the guess
        for i, letter in enumerate(guess):
            if letter == unmatched_guess[i]:
                if word_dict.get(letter, 0) > 0:
                    # Correct letter in the wrong position
                    state = LetterState.YELLOW
                    word_dict[letter] -= 1
                else:
                    # Incorrect letter
                    state = LetterState.GREY

                self.guesses[-self.num_guesses][i][guess[i]] = state

                if self.keyboard[guess[i]] not in (
                    LetterState.GREEN,
                    LetterState.GREY,
                ) and (
                    self.keyboard[guess[i]] == LetterState.EMPTY
                    or (
                        self.keyboard[guess[i]] == LetterState.YELLOW
                        and state == LetterState.GREEN
                    )
                ):
                    self.keyboard[guess[i]] = state

        self.num_guesses -= 1

        # np.set_printoptions(threshold=np.inf)
        # test_state = self.get_game_state()
        # print(test_state.shape)
        # print(test_state)

        if guess == self.wordle_word:
            return GameStatus.GAME_WON
        elif self.num_guesses > 0:
            return GameStatus.VALID_WORD
        else:
            return GameStatus.GAME_OVER


class Game:
    # Initialize game parameters
    def __init__(
        self,
        num_guesses: int = NUM_GUESSES,
        word_length: int = WORD_LENGTH,
        visualize: bool = True,
        save_game: bool = False,
        wordle_word: str = None,
    ):
        self.num_guesses = num_guesses
        self.word_length = word_length
        self.wordle_word = wordle_word
        self.visualize = visualize
        self.save_game = save_game
        self.tile_size = (WINDOW_SIZE[0] / self.word_length) - (TILE_SPACING * 2)
        self.key_size = (
            WINDOW_SIZE[1] - ((self.tile_size + TILE_SPACING) * num_guesses)
        ) / 3

        self.history = []

        if wordle_word is not None:
            assert (
                len(wordle_word) == word_length
            ), "target word must have length of word_length"
            assert is_valid(WORD_DATA_FILE, wordle_word), "target word must be valid"

        self.game_data = GameData(word_length, num_guesses, wordle_word)

    # Display the typed word
    def update_type(self, typed_word: str):
        self.font = pygame.font.SysFont("Arial", TILE_FONT_SIZE, bold=True)
        i = self.num_guesses - self.game_data.num_guesses
        for j in range(self.word_length):
            x, y = (
                j * self.tile_size + (TILE_SPACING * j) + TILE_SPACING,
                i * self.tile_size + (TILE_SPACING * i) + TILE_SPACING,
            )

            if j < len(typed_word):
                # Draw Letter
                letter = self.font.render(
                    str(typed_word[j]), True, KEY_FONT_COLOR_BLACK
                )
                text_rect = letter.get_rect(
                    center=(x + self.tile_size / 2, y + self.tile_size / 2)
                )
                self.window.blit(letter, text_rect)
            else:
                # Draw Blank Tile
                pygame.draw.rect(
                    self.window,
                    TILE_COLORS[LetterState.EMPTY],
                    pygame.Rect(
                        x + BORDER_WIDTH,
                        y + BORDER_WIDTH,
                        self.tile_size - (BORDER_WIDTH * 2),
                        self.tile_size - (BORDER_WIDTH * 2),
                    ),
                )

        # Update the display
        pygame.display.update()

    # Redraw the board
    def update_screen(self):
        self.window.fill(BACKGROUND_COLOR)
        self.font = pygame.font.SysFont("Arial", TILE_FONT_SIZE, bold=True)
        # Update the guesses
        for i in range(self.num_guesses):
            for j in range(self.word_length):
                x, y = (
                    j * self.tile_size + (TILE_SPACING * j) + TILE_SPACING,
                    i * self.tile_size + (TILE_SPACING * i) + TILE_SPACING,
                )
                tile = [
                    (key, value)
                    for key, value in self.game_data.guesses[i][j].items()
                    if value != LetterState.EMPTY
                ]
                if len(tile) == 0:
                    color_state = LetterState.EMPTY
                elif len(tile) == 1:
                    color_state = tile[0][1]
                else:
                    raise Exception("Invalid tile state")
                color = TILE_COLORS[color_state]

                # Draw Border
                pygame.draw.rect(
                    self.window,
                    BORDER_COLOR,
                    pygame.Rect(x, y, self.tile_size, self.tile_size),
                )
                # Draw Tile
                pygame.draw.rect(
                    self.window,
                    color,
                    pygame.Rect(
                        x + BORDER_WIDTH,
                        y + BORDER_WIDTH,
                        self.tile_size - (BORDER_WIDTH * 2),
                        self.tile_size - (BORDER_WIDTH * 2),
                    ),
                )
                # Draw Letter
                if color_state != LetterState.EMPTY:
                    letter = self.font.render(str(tile[0][0]), True, TILE_FONT_COLOR)
                    text_rect = letter.get_rect(
                        center=(x + self.tile_size / 2, y + self.tile_size / 2)
                    )
                    self.window.blit(letter, text_rect)

        x, y = (0, (BORDER_WIDTH + self.tile_size) * self.num_guesses)
        # Update the keyboard
        count = 0
        self.font = pygame.font.SysFont("Arial", KEY_FONT_SIZE, bold=True)
        for key in self.game_data.keyboard:
            # Draw Tile
            if count % 9 == 8:
                y += self.key_size
                x = 0
            pygame.draw.rect(
                self.window,
                TILE_COLORS[self.game_data.keyboard[key]],
                pygame.Rect(x, y, self.key_size, self.key_size),
            )
            # Draw Letter
            if self.game_data.keyboard[key] is LetterState.EMPTY:
                text_color = KEY_FONT_COLOR_BLACK
            else:
                text_color = TILE_FONT_COLOR
            letter = self.font.render(str(key), True, text_color)
            text_rect = letter.get_rect(
                center=(x + self.key_size / 2, y + self.key_size / 2)
            )
            self.window.blit(letter, text_rect)
            x += self.key_size
            count += 1

        # Update the display
        pygame.display.update()

    def init_screen(self):
        # Initialize Pygame
        pygame.init()
        self.window = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("wordle")

        # Load the font
        self.font = pygame.font.SysFont("Arial", TILE_FONT_SIZE, bold=True)

    def store_history(self, directory: str):
        with open(f"{directory}/game-{self.game_data.wordle_word}.pkl", "wb") as f:
            pickle.dump(self.history, f)

    def step(self, guess: str) -> Tuple[ActionReward, GameStatus]:
        # TODO: Convert one hot encoded vector to str???
        status = self.game_data.make_guess(guess)
        if (
            status is not GameStatus.INVALID_LENGTH
            and status is not GameStatus.INVALID_WORD
        ):
            self.history.append(guess)
            valid = True
        else:
            valid = False
        reward = ActionReward(
            self.history, self.game_data.wordle_word, self.num_guesses, valid
        )

        return reward, status

    def run(self):
        if self.visualize:
            self.init_screen()

            self.update_screen()

        quit_game = False
        game_complete = False
        game_over = quit_game or game_complete
        make_step = False
        # Start the game loop
        current_word = ""
        while not game_over:
            event = pygame.event.wait()
            while event.type not in [pygame.KEYDOWN, pygame.QUIT]:
                event = pygame.event.wait()
            if event.type == pygame.QUIT:
                quit_game = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    current_word = current_word[:-1]
                    self.update_type(current_word)
                elif (
                    event.key == pygame.K_RETURN
                    and len(current_word) == self.word_length
                ):
                    make_step = True
                elif len(current_word) < self.word_length:
                    current_word += chr(event.key)
                    self.update_type(current_word)

            game_won = False
            if make_step:
                _, status = self.step(current_word)
                game_complete = (
                    status is GameStatus.GAME_OVER or status is GameStatus.GAME_WON
                )
                game_won = status is GameStatus.GAME_WON
                current_word = ""

            game_over = quit_game or game_complete

            if self.visualize and make_step:
                self.update_screen()

            make_step = False

            if game_over and self.visualize:
                if game_won:
                    game_over_text = self.font.render(
                        "You Won!", True, GAME_OVER_FONT_COLOR
                    )
                    game_over_color = GAME_WON_COLOR
                else:
                    game_over_text = self.font.render(
                        self.game_data.wordle_word, True, GAME_OVER_FONT_COLOR
                    )
                    game_over_color = GAME_OVER_COLOR
                game_over_rect = game_over_text.get_rect(
                    center=(WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] / 2)
                )
                pygame.draw.rect(
                    self.window, game_over_color, game_over_rect.inflate(20, 20)
                )
                self.window.blit(game_over_text, game_over_rect)

                # Update the display
                pygame.display.update()

                break

        if self.visualize:
            # Quit Pygame
            time.sleep(2)
            pygame.quit()

        if self.save_game:
            self.store_history("saved_games")


if __name__ == "__main__":
    # Game(wordle_word="crepe").run()
    Game().run()
