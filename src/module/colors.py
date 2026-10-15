import os
import sys

reset = '\033[0m'

CYAN = (150, 225, 235)
MAUVE = (200, 170, 235)
PINK = (245, 175, 205)
MINT = (170, 235, 190)

_ENABLED = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def rgb(r: int, g: int, b: int):
    return f'\033[38;2;{str(r)};{str(g)};{str(b)}m' if _ENABLED else ''


def fade(text: str):
    return ''.join(f'{rgb(0, 150 + 4 * i, 255)}{char}' for i, char in enumerate(text)) + (reset if _ENABLED else '')


def paint(text: str, color: tuple):
    return f'{rgb(*color)}{text}{reset}' if _ENABLED else text


TAGS = {
    "conv": CYAN,
    "gauss": MAUVE,
    "sobel": PINK,
    "canny": MINT,
}


def tag(name: str):
    return paint(f"[{name}]", TAGS[name])
