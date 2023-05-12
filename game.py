import enum
import pickle
import time
from typing import Tuple
import pygame
import numpy as np
# from history import GameHistory
# from reward import ActionReward
# from direction import Direction
from constants import *

import string
from letter_state import LetterState


class GameStatus(enum.Enum):
    GAME_OVER = 0
    INVALID_WORD = 1
    INVALID_LENGTH = 2
    VALID_WORD = 3

class GameData:
    def __init__(self, word_length: int, num_guesses: int):
        self._keyboard = dict.fromkeys(string.ascii_lowercase, LetterState.EMPTY)
        self._guesses = [[dict.fromkeys(string.ascii_lowercase, LetterState.EMPTY)] * word_length] * num_guesses
        # self._first_guess = [dict.fromkeys(string.ascii_lowercase, LetterState.EMPTY)] * word_length
        # self._second_guess = [dict.fromkeys(string.ascii_lowercase, LetterState.EMPTY)] * word_length
        # self._third_guess = [dict.fromkeys(string.ascii_lowercase, LetterState.EMPTY)] * word_length
        # self._fourth_guess = [dict.fromkeys(string.ascii_lowercase, LetterState.EMPTY)] * word_length
        # self._fifth_guess = [dict.fromkeys(string.ascii_lowercase, LetterState.EMPTY)] * word_length
        # self._sixth_guess = [dict.fromkeys(string.ascii_lowercase, LetterState.EMPTY)] * word_length

        self.num_guesses = num_guesses

    def make_guess(self, guess: str) -> GameStatus:
        pass

    


class Game:
    # Initialize game parameters
    def __init__(
        self,
        num_guesses: int = NUM_GUESSES,
        seed: int = None,
        # initial_board: np.ndarray = None,
        # iterative_mode: bool = False,
        # visualize: bool = True,
        # save_game: bool = False,
        # command_list: list[Direction] = None,
    ):
        self.num_guesses = num_guesses
        self.seed = seed
        # self.initial_board = initial_board
        # self.tile_size = BOARD_SIZE[0] / board_size
        # self.visualize = visualize
        # self.save_game = save_game
        # self.command_list = command_list
        # self.total_score = 0

        if self.seed is not None:
            np.random.seed(seed)

    
    # Redraw the board
    def update_screen(self, board: np.ndarray):
        # self.window.fill(BACKGROUND_COLOR)
        # pygame.draw.rect(
        #     self.window, TILE_COLORS[0], pygame.Rect(*BOARD_POS, *BOARD_SIZE)
        # )
        # for i in range(self.board_size):
        #     for j in range(self.board_size):
        #         x, y = (
        #             BOARD_POS[0] + j * self.tile_size,
        #             BOARD_POS[1] + i * self.tile_size,
        #         )
        #         value = board[i, j]
        #         color = TILE_COLORS[value]
        #         pygame.draw.rect(
        #             self.window,
        #             color,
        #             pygame.Rect(x, y, self.tile_size, self.tile_size),
        #         )
        #         if value > 0:
        #             text = self.font.render(str(value), True, TILE_FONT_COLOR)
        #             text_rect = text.get_rect(
        #                 center=(x + self.tile_size / 2, y + self.tile_size / 2)
        #             )
        #             self.window.blit(text, text_rect)
        # # Update the display
        # pygame.display.update()

    def init_screen(self):
        # Initialize Pygame
        pygame.init()
        self.window = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("wordle")

        # Load the font
        self.font = pygame.font.SysFont("Arial", TILE_FONT_SIZE, bold=True)

    def store_history(self, directory: str):
        with open(f"{directory}/game-{self.seed}.pkl", "wb") as f:
            pickle.dump(self.history, f)

    def step(
        self, board: np.ndarray, command: Direction
    ) -> Tuple[np.ndarray, ActionReward, bool]:
        game_over = False
        prev_board = board.copy()
        current_board = board.copy()
        collapsed_board, score = self.move_tiles(current_board, command)

        if self.save_game:
            self.history.add_action(command)

        # Check if the game is over
        if self.is_game_over(collapsed_board):
            game_over = True
            new_board = collapsed_board
        elif np.array_equal(prev_board, collapsed_board):
            new_board = collapsed_board
        else:
            new_board = self.add_tile(collapsed_board)

        reward = ActionReward(score, prev_board, new_board)

        return new_board, reward, game_over

    def run(self):
        # Initialize the game board
        if self.initial_board is None:
            board = self.init_board()
        else:
            board = self.initial_board

        prev_board = board.copy()

        if self.visualize:
            self.init_screen()

            self.update_screen(board)

            if self.command_list is not None:
                pygame.event.get()

        quit_game = False
        board_complete = False
        game_over = quit_game or board_complete

        # Start the game loop
        while not game_over:
            if self.command_list is not None:
                # Take a step through command list
                command = self.command_list.pop(0)
                quit_game = len(self.command_list) == 0
                if self.visualize:
                    time.sleep(0.75)
            else:
                # Handle events
                event = pygame.event.wait()
                while event.type not in [pygame.KEYDOWN, pygame.QUIT]:
                    event = pygame.event.wait()
                if event.type == pygame.QUIT:
                    quit_game = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        command = Direction.UP
                    elif event.key == pygame.K_DOWN:
                        command = Direction.DOWN
                    elif event.key == pygame.K_LEFT:
                        command = Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        command = Direction.RIGHT

            if self.save_game:
                self.history.add_action(command)

            board, reward, board_complete = self.step(board, command)
            self.total_score += reward.action_score
            game_over = quit_game or board_complete

            if self.visualize and not np.array_equal(prev_board, board):
                self.update_screen(board)

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

            prev_board = board.copy()

        if self.visualize:
            # Quit Pygame
            pygame.quit()

        if self.save_game:
            self.history.final_board = board
            self.store_history("saved_games")


if __name__ == "__main__":
    Game().run()
