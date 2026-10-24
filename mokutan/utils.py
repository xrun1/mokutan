from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import appdirs
from fastapi import Request, status
from fastapi.datastructures import URL
from fastapi.responses import RedirectResponse
from natsort import natsorted

from . import NAME

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

log = logging.getLogger("uvicorn")

DATA_DIR = Path(appdirs.user_data_dir(NAME, appauthor=False, roaming=True))
CACHE_DIR = Path(appdirs.user_cache_dir(NAME, appauthor=False))


class Point(NamedTuple):
    x: float
    y: float


class ReferrerRedirect(RedirectResponse):
    def __init__(self, request: Request) -> None:
        super().__init__(
            URL(request.headers.get("referer") or "/")
                .replace(fragment=request.query_params.get("focus")),
            status.HTTP_303_SEE_OTHER,
        )


def ellide(text: str, max_length: int) -> str:
    return text[:max_length] + "…" if len(text) > max_length else text


def get_sorted_dir[T: Path](folder: T) -> list[T]:
    return natsorted(folder.iterdir(), key=lambda p: p.name)


def flatten[T](groups: Iterable[Iterable[T]]) -> Iterator[T]:
    for group in groups:
        yield from group


def catch_log_exceptions[**P](fn: Callable[P, None]) -> Callable[P, None]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception:
            log.exception("Error caught")
    return wrapper


def is_supported_archive(path: Path) -> bool:
    return path.suffix in {".cbz", ".zip"} and path.is_file()


def is_web_image(path: Path | str) -> bool:
    return Path(path).suffix in {
        ".avif", ".bmp", ".gif", ".ico", ".jpg", ".jpeg",
        ".png", ".svg", ".tif", ".tiff", ".webp",
    }


def winsorized_mean(
    *nums: float, trim_start: float = 0.1, trim_end: float = 0.1,
) -> list[float]:
    # e.g. start/end 0.1: [12, 15, 18, 22, 90] → [15, 15, 18, 22, 22]
    if not nums:
        return []
    ordered = sorted(nums)
    lower = ordered[int(len(nums) * trim_start)]
    upper = ordered[len(nums) - 1 - int(len(nums) * trim_end)]
    return [min(max(n, lower), upper) for n in nums]
