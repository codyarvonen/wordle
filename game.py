import enum
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
    def __init__(self, word_length: int, num_guesses: int):
        self.word_length = word_length
        self.num_guesses = num_guesses
        self.wordle_word = select_random_word(WORD_DATA_FILE)
        self.keyboard = dict.fromkeys(string.ascii_lowercase, LetterState.EMPTY)
        self.guesses = [[dict.fromkeys(string.ascii_lowercase, LetterState.EMPTY) for _ in range(word_length)] for _ in range(num_guesses)]

    def make_guess(self, guess: str) -> GameStatus:            
        if len(guess) != self.word_length:
            return GameStatus.INVALID_LENGTH
        if not is_valid(WORD_DATA_FILE, guess):
            return GameStatus.INVALID_WORD
        
        # TODO: Make it so that you can input the wordle word
        
        # TODO: Fix Yellow letter bug
        
        # TODO: Make sure that the keyboard is accurate

        # TODO: Write test functions to verify edge cases

        # TODO: Fix the UI
                        
        # Update the game data
        for i in range(self.word_length):
            if guess[i] == self.wordle_word[i]:
                state = LetterState.GREEN
            elif guess[i] in self.wordle_word and guess[:i].count(guess[i]) <= self.wordle_word.count(guess[i]):
                state = LetterState.YELLOW
            else:
                state = LetterState.GREY
            self.guesses[-self.num_guesses][i][guess[i]] = state
            self.keyboard[guess[i]] = state

        self.num_guesses -= 1

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
        seed: int = None,
        visualize: bool = True,
        save_game: bool = False
    ):
        self.num_guesses = num_guesses
        self.word_length = word_length
        self.seed = seed
        self.visualize = visualize
        self.save_game = save_game
        self.tile_size = (WINDOW_SIZE[0] / self.word_length) - TILE_SPACING
        self.key_size = (WINDOW_SIZE[1] - ((self.tile_size + TILE_SPACING) * num_guesses)) / 3
        
        self.history = []

        self.game_data = GameData(word_length, num_guesses)

        if self.seed is not None:
            np.random.seed(seed)

    
    # Redraw the board
    def update_screen(self):
        self.window.fill(BACKGROUND_COLOR)
        self.font = pygame.font.SysFont("Arial", TILE_FONT_SIZE, bold=True)
        # Update the guesses
        for i in range(self.num_guesses):
            for j in range(self.word_length):
                x, y = (
                    BORDER_WIDTH + j * self.tile_size,
                    BORDER_WIDTH + i * self.tile_size,
                )
                tile = [(key, value) for key, value in self.game_data.guesses[i][j].items() if value != LetterState.EMPTY]
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
                    pygame.Rect(x + BORDER_WIDTH, y + BORDER_WIDTH, self.tile_size - BORDER_WIDTH, self.tile_size - BORDER_WIDTH),
                )
                # Draw Letter
                if color_state != LetterState.EMPTY:
                    letter = self.font.render(str(tile[0][0]), True, TILE_FONT_COLOR)
                    text_rect = letter.get_rect(
                        center=(x + self.tile_size / 2, y + self.tile_size / 2)
                    )
                    self.window.blit(letter, text_rect)
        
        x, y = (
            0,
            (BORDER_WIDTH + self.tile_size) * self.num_guesses
        )
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
        if status is not GameStatus.INVALID_LENGTH and status is not GameStatus.INVALID_WORD:
            self.history.append(guess)
            valid = True
        else: 
            valid = False
        reward = ActionReward(self.history, self.game_data.wordle_word, self.num_guesses, valid)

        return reward, status

    def run(self):
        
        if self.visualize:
            self.init_screen()

            self.update_screen()

            # if self.command_list is not None:
            #     pygame.event.get()

        quit_game = False
        game_complete = False
        game_over = quit_game or game_complete
        make_step = False
        # Start the game loop
        current_word = ""
        while not game_over:
            # if self.command_list is not None:
            #     # Take a step through command list
            #     # command = self.command_list.pop(0)
            #     # quit_game = len(self.command_list) == 0
            #     # if self.visualize:
            #     #     time.sleep(0.75)
            #     pass
            # else:
            # Handle events
            event = pygame.event.wait()
            while event.type not in [pygame.KEYDOWN, pygame.QUIT]:
                event = pygame.event.wait()
            if event.type == pygame.QUIT:
                quit_game = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    current_word = current_word[:-1]
                elif event.key == pygame.K_RETURN and len(current_word) == self.word_length:
                    # self.step(current_word)
                    # self.update_screen()
                    make_step = True
                elif len(current_word) < self.word_length:
                    current_word += chr(event.key)
                print(current_word)
                        
            if make_step:
                _, status = self.step(current_word)
                game_complete = status is GameStatus.GAME_OVER or status is GameStatus.GAME_WON
                current_word = ""
            
            game_over = quit_game or game_complete

            if self.visualize and make_step:
                self.update_screen()

            make_step = False

            if game_over and self.visualize:
                game_over_text = self.font.render(
                    "Game Over!", True, GAME_OVER_FONT_COLOR
                )
                game_over_rect = game_over_text.get_rect(
                    center=(WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] / 2)
                )
                pygame.draw.rect(
                    self.window, GAME_OVER_COLOR, game_over_rect.inflate(20, 20)
                )
                self.window.blit(game_over_text, game_over_rect)

                # Update the display
                pygame.display.update()

                break

        if self.visualize:
            # Quit Pygame
            pygame.quit()

        if self.save_game:
            self.store_history("saved_games")


if __name__ == "__main__":
    Game().run()
