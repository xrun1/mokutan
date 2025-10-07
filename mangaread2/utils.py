from __future__ import annotations

import asyncio
import functools
import logging
import mimetypes
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING, NamedTuple
from uuid import uuid4
from zipfile import ZipFile

from natsort import natsorted

from . import NAME

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from threading import Event

log = logging.getLogger(NAME)
log.setLevel(logging.INFO)

TEMP = Path(gettempdir()) / NAME
LOCKS: defaultdict[Path, asyncio.Lock] = defaultdict(asyncio.Lock)
IGNORED_ARCHIVES: set[Path] = set()
LAST_ARCHIVE_ACCESSES: dict[Path, datetime] = {}


class Point(NamedTuple):
    x: float
    y: float


def get_sorted_dir[T: Path](folder: T) -> list[T]:
    return natsorted(folder.iterdir(), key=lambda p: p.name)


def trim_archive_cache(stop: Event) -> None:
    if TEMP.exists():
        shutil.rmtree(TEMP, ignore_errors=True)

    while not stop.is_set():
        long_ago = datetime.now() - timedelta(hours=2)
        for folder, date in LAST_ARCHIVE_ACCESSES.items():
            if date < long_ago:
                shutil.rmtree(folder, ignore_errors=True)
                del LAST_ARCHIVE_ACCESSES[folder]
        time.sleep(1)


async def get_auto_extracted_path(path: Path) -> Path | None:
    if path in IGNORED_ARCHIVES or not (archive := next((
        p for p in (path, *path.parents)
        if is_supported_archive(p) and p.is_file()
    ), None)):
        return None

    def fix_end(path: Path | str) -> Path:
        if os.name == "nt":
            assert Path(path).is_absolute()
            return Path(str(path).replace(":", "", 1))
        return Path(path)

    if (base := TEMP / fix_end(archive)).exists():
        LAST_ARCHIVE_ACCESSES[archive] = datetime.now()
        return TEMP / fix_end(path)

    async with LOCKS[base]:
        with ZipFile(archive) as arc:
            renderable = [
                f for f in arc.filelist
                if f.is_dir() or is_web_image(f.orig_filename)
            ]
            if len(renderable) < len(arc.filelist) * 0.8:
                IGNORED_ARCHIVES.add(path)
                return None

            base.mkdir(parents=True)

            await asyncio.to_thread(arc.extractall, base)

        while len(list(base.iterdir())) == 1 and next(base.iterdir()).is_dir():
            new = base.parent / str(uuid4())
            shutil.move(next(base.iterdir()), new)
            shutil.rmtree(base)
            new.rename(base)

    LAST_ARCHIVE_ACCESSES[archive] = datetime.now()
    return TEMP / fix_end(path)


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
