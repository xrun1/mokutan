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
