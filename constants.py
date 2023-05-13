from letter_state import LetterState

NUM_GUESSES = 6
WORD_LENGTH = 5

WINDOW_SIZE = (500, 700)
BOARD_POS = (10, 10)
BACKGROUND_COLOR = (255, 255, 255)
BORDER_COLOR = (210, 214, 218)
TILE_FONT_COLOR = (255, 255, 255)
KEY_FONT_COLOR_BLACK = (0, 0, 0)
GAME_OVER_COLOR = (255, 0, 0)
GAME_OVER_FONT_SIZE = 48
GAME_OVER_FONT_COLOR = (255, 255, 255)
TILE_FONT_SIZE = 32
KEY_FONT_SIZE = 12
TILE_SPACING = 5
BORDER_WIDTH = 5

TILE_COLORS = {
    LetterState.EMPTY: (255, 255, 255),
    LetterState.GREY: (120, 124, 126),
    LetterState.YELLOW: (201, 180, 88),
    LetterState.GREEN: (106, 170, 100),
}


WORD_DATA_FILE = 'words.csv'