from __future__ import annotations

import functools
import logging
import mimetypes
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING, NamedTuple

import appdirs
from natsort import natsorted

from . import NAME

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

log = logging.getLogger(NAME)
log.setLevel(logging.INFO)

TEMP = Path(gettempdir()) / NAME
DATA_DIR = Path(appdirs.user_data_dir(NAME, appauthor=False, roaming=True))


class Point(NamedTuple):
    x: float
    y: float


def get_sorted_dir[T: Path](folder: T) -> list[T]:
    return natsorted(folder.iterdir(), key=lambda p: p.name)


def flatten[T](groups: Iterable[Iterable[T]]) -> list[T]:
    return [item for group in groups for item in group]


def catch_log_exceptions[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            fn(*args, **kwargs)
        except Exception:
            log.exception("Error caught")
    return wrapper


def is_supported_archive(path: Path) -> bool:
    return (mimetypes.guess_type(path)[0] or "") in {
        "image/cbr",
        "application/x-zip-compressed",
    }


def is_web_image(path: Path | str) -> bool:
    return (mimetypes.guess_type(path)[0] or "") in {
        "image/apng",
        "image/avif",
        "image/bmp",
        "image/gif",
        "image/vnd.microsoft.icon",
        "image/jpeg",
        "image/png",
        "image/svg+xml",
        "image/tiff",
        "image/webp",
    }


def is_hiragana(char: str) -> bool:
    return 0x3040 <= ord(char) <= 0x309F


def is_katakana(char: str) -> bool:
    return 0x30A0 <= ord(char) <= 0x30FF


def is_kanji(char: str) -> bool:
    return (
        (0x3400 <= (code := ord(char)) <= 0x4DBF) or
        (0x4E00 <= code <= 0x9FFF) or
        (0xF900 <= code <= 0xFAFF)
    )


def script_difficulty(word: str) -> float:
    def score_char(char: str) -> float:
        if is_hiragana(char):
            return 0.3
        if is_katakana(char):
            return 0.4
        if is_kanji(char):
            return 1
        return 0

    scores = [score_char(char) for char in word]
    factor = len(word) ** 1.2

    if set(scores) == {1}:  # all kanji
        factor *= 1.3
    elif set(scores) == {0.3}:  # all hiragana
        factor *= 0.5
    elif set(scores) == {0.4}:  # all katakana, probably loanword or SFX
        factor *= 0.1

    return sum(scores) / len(word) * factor
