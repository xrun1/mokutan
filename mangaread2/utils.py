from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, NamedTuple

from natsort import natsorted

from . import NAME

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

log = logging.getLogger(NAME)
log.setLevel(logging.INFO)


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
