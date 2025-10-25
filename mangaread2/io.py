from __future__ import annotations

import asyncio
import json
import math
import multiprocessing
import os
import shutil
import time
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4
from zipfile import ZipFile

import wakepy
from fastapi.datastructures import URL
from fastapi.responses import RedirectResponse
from natsort import natsorted

from mangaread2.difficulty import Difficulty, anki

from .utils import (
    CACHE_DIR,
    DATA_DIR,
    is_supported_archive,
    is_web_image,
    log,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from threading import Event

from fastapi import APIRouter, Response, status

from .utils import Point, flatten, get_sorted_dir

router = APIRouter(prefix="/ocr")

EXTRACT_DIR = CACHE_DIR / "Extracted"
OCR_QUEUE: deque[MPath] = deque()
IGNORED_ARCHIVES: set[Path] = set()
LAST_ARCHIVE_ACCESSES: dict[Path, datetime] = {}
LOCKS: defaultdict[Path, asyncio.Lock] = defaultdict(asyncio.Lock)
pause_queue: bool = False


@dataclass(slots=True)
class OCRBox:
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0
    image_w: int = 0
    image_h: int = 0
    vertical: bool = False
    font_size: int = 0
    lines: list[str] = field(default_factory=list)


class MPath(Path):
    @property
    async def extracted(self) -> Self:
        if self in IGNORED_ARCHIVES or not (archive := next((
            p for p in (self, *self.parents)
            if is_supported_archive(p) and p.is_file()
        ), None)):
            return self

        def fix_end(path: Self) -> Self:
            if os.name == "nt":
                assert path.is_absolute()
                return type(self)(str(path).replace(":", "", 1))
            return path

        if (base := EXTRACT_DIR / fix_end(archive)).exists():
            LAST_ARCHIVE_ACCESSES[archive] = datetime.now()
            return type(self)(EXTRACT_DIR) / fix_end(self)

        async with LOCKS[base]:
            with ZipFile(archive) as arc:
                renderable = [
                    f for f in arc.filelist
                    if f.is_dir() or is_web_image(f.orig_filename)
                ]
                if len(renderable) < len(arc.filelist) * 0.8:
                    IGNORED_ARCHIVES.add(self)
                    return self

                base.mkdir(parents=True)

                await asyncio.to_thread(arc.extractall, base)

            while True:
                inside = list(base.iterdir())
                if not (len(inside) == 1 and inside[0].is_dir()):
                    break
                new = base.parent / str(uuid4())
                shutil.move(next(base.iterdir()), new)
                shutil.rmtree(base)
                new.rename(base)

        LAST_ARCHIVE_ACCESSES[archive] = datetime.now()
        return type(self)(EXTRACT_DIR) / fix_end(self)

    @property
    def unextracted(self) -> Self:
        if EXTRACT_DIR not in self.parents:
            return self

        path = str(self).removeprefix(str(EXTRACT_DIR) + os.sep)
        if os.name == "nt":
            path = path[0] + ":" + path[1:]

        return type(self)(path) if Path(path).exists() else self

    @property
    def as_anchor(self) -> str:
        return self.stem.replace(" ", "-")

    @property
    def images(self) -> list[Self]:
        if not self.is_dir():
            return []
        return [p for p in get_sorted_dir(self) if is_web_image(p)]

    def non_images(self, sort: str = "") -> list[Self]:
        return self._sort(sort, (
            p for p in self.iterdir() if not
            (is_web_image(p) or p.suffix == ".mokuro" or p.name == "_ocr")
        ))

    @property
    def difficulty(self) -> Difficulty:
        return Difficulty.calculate(self.ocr_json_file)

    @property
    def ocr_wip_dir(self) -> Self:
        name = self.name if self.unextracted.is_dir() else self.stem
        return self.unextracted.parent / "_ocr" / name

    @property
    def ocr_json_file(self) -> Self:
        name = self.name if self.unextracted.is_dir() else self.stem
        return self.unextracted.parent / (name + ".mokuro")

    @property
    def ocr_queue_position(self) -> int | None:
        try:
            return OCR_QUEUE.index(self)
        except ValueError:
            return None

    @property
    def ocr_progress(self) -> tuple[int, int]:
        try:
            if is_supported_archive(self):
                with ZipFile(self) as zf:
                    total = len([
                        f for f in zf.filelist if is_web_image(f.orig_filename)
                    ])
            else:
                total = len(self.images)
        except OSError:
            log.exception("Error trying to gauge OCR progress for %s", self)
            return (0, 0)

        if self.ocr_json_file.exists():
            return (total, total)
        if not self.ocr_wip_dir.exists():
            return (0, total)
        return (len(list(self.ocr_wip_dir.glob("*.json"))), total)

    @property
    def ocr_progress_text(self) -> str:
        return "/".join(map(str, self.ocr_progress))

    @property
    def ocr_can_begin(self) -> bool:
        return not (self.ocr_paused or self.ocr_running)

    @property
    def ocr_paused(self) -> bool:
        return pause_queue and bool(OCR_QUEUE) and OCR_QUEUE[0] == self

    @property
    def ocr_running(self) -> bool:
        return not pause_queue and bool(OCR_QUEUE) and OCR_QUEUE[0] == self

    @property
    def ocr_done(self) -> bool:
        done, total = self.ocr_progress
        return done >= total

    @property
    def read_date(self) -> tuple[datetime, timedelta, str] | None:
        if (data := self.get_mark("read")):
            date = datetime.fromisoformat(data).astimezone()
            delta = datetime.now(UTC) - date.astimezone(UTC)
            sec = delta.total_seconds()

            if sec < 60:
                return (date, delta, f"{int(max(1, sec))}s")
            if sec < 60 * 60:
                return (date, delta, f"{int(sec / 60)}m")
            if sec < 60 * 60 * 24:
                return (date, delta, f"{int(sec / 60 / 60)}h")
            if sec < 60 * 60 * 24 * 31:
                return (date, delta, f"{int(sec / 60 / 60 / 24)}d")
            if sec < 60 * 60 * 24 * 31 * 365:
                return (date, delta, f"{int(sec / 60 / 60 / 24 / 31)}mo")
            return (date, delta, f"{int(sec / 60 / 60 / 24 / 31 / 365)}y")
        return None

    def url(self, **params: Any) -> URL:
        # use .absolute() or first \ gets mangled on windows sometimes somehow
        base = "/" + str(self.absolute().as_posix())
        return URL(base).include_query_params(**params)

    def previous_chapter(self, sort: str = "") -> Self | None:
        chapters, own_idx = self._sibling_chapters(sort)
        return chapters[own_idx - 1] if own_idx > 0 else None

    def next_chapters(self, sort: str = "") -> list[Self]:
        chapters, own_idx = self._sibling_chapters(sort)
        return chapters[own_idx + 1:]

    def next_chapter(self, sort: str = "") -> Self | None:
        return next(iter(self.next_chapters(sort)), None)

    def has_mark(self, name: str) -> bool:
        return self._mark(name).exists()

    def get_mark(self, name: str) -> str | None:
        try:
            return self._mark(name).read_text(encoding="utf-8").rstrip()
        except FileNotFoundError:
            return None

    def set_mark(self, name: str, data: str | None) -> None:
        if data is None:
            self._mark(name).unlink(missing_ok=True)
            with suppress(OSError):
                self._mark(name).parent.rmdir()
        else:
            self._mark(name).parent.mkdir(parents=True, exist_ok=True)
            self._mark(name).write_text(data, encoding="utf-8")

    def mark_read(self) -> None:
        self.set_mark("read", datetime.now(UTC).astimezone().isoformat())

    def mark_unread(self) -> None:
        self.set_mark("read", None)

    def ocr_boxes(self, image: Path) -> Iterable[OCRBox]:
        if self.ocr_json_file.exists():
            data = json.load(self.ocr_json_file.open(encoding="utf-8"))
        elif self.ocr_wip_dir.exists():
            data = {"pages": [
                json.load(f.open(encoding="utf-8")) | {"img_path": f.name}
                for f in natsorted(
                    self.ocr_wip_dir.glob("*.json"), key=lambda f: f.name,
                )
            ]}
        else:
            return

        if not (page := next((
            p for p in data["pages"] if Path(p["img_path"]).stem == image.stem
        ), None)):
            return

        page_w, page_h = page["img_width"], page["img_height"]

        for block in page["blocks"]:
            lines: list[str] = block["lines"]
            coords: list[list[list[float]]] = block["lines_coords"]
            if block["vertical"]:
                lines.reverse()
                coords.reverse()

            boxes = []
            prev_start = prev_end = Point(math.inf, math.inf)

            # Split OCR-detected boxes that are probably multiple
            # multiple actual boxes in the image based on text spacing
            for line, coord in zip(lines, coords, strict=True):
                start, _top_right, end, _bot_left = starmap(Point, coord)

                if block["vertical"]:
                    new_box = (
                        abs(start.y - prev_start.y) > page_h / 100 or
                        abs(start.x - prev_end.x) > page_w / 20
                    )
                else:
                    new_box = (
                        abs(start.x - prev_start.x) > page_w / 100 or
                        abs(start.y - prev_end.y) > page_h / 100
                    )

                if new_box:
                    boxes.append(OCRBox(
                        x=start.x,
                        y=start.y,
                        image_w=page["img_width"],
                        image_h=page["img_height"],
                        vertical=block["vertical"],
                        font_size=block["font_size"],
                    ))

                box = boxes[-1]
                box.lines.append("".join(anki.html_mark_known(line)))
                box.x = min(box.x, start.x)
                box.y = min(box.y, start.y)
                box.w = max(box.w, end.x - box.x)
                box.h = max(box.h, end.y - box.y)

                prev_start, prev_end = start, end

            for box in boxes:  # Convert to 0-1 percentages
                box.x /= page_w
                box.y /= page_h
                box.w /= page_w
                box.h /= page_h
                if box.vertical:
                    box.lines.reverse()
                yield box

    @classmethod
    def _sort(cls, sort: str, p: Iterable[Self]) -> list[Self]:
        def ns(key: Callable[[Self], Any], invert: bool = False) -> list[Self]:
            return natsorted(p, key=lambda e: (key(e), e.name), reverse=invert)

        if sort == "m":
            return ns(lambda e: e.stat().st_mtime)
        if sort == "p":
            return ns(lambda e: len(e.images))
        if sort == "d":
            return ns(lambda e: e.difficulty.score or math.inf)
        if sort == "a":
            return ns(lambda e: e.difficulty.anki_learned_percent, invert=True)
        if sort == "r":
            no_date = datetime.fromtimestamp(0, tz=UTC)
            return ns(lambda e: (e.read_date or [no_date])[0], invert=True)
        return ns(lambda _: -1)

    def _sibling_chapters(self, sort: str = "") -> tuple[list[Self], int]:
        if self.unextracted.parent is self:  # drive root
            return ([], -1)
        chapters = self._sort(sort, (
            p for p in self.unextracted.parent.iterdir()
            if (p.is_dir() and p.name != "_ocr") or is_supported_archive(p)
        ))
        return (chapters, chapters.index(self.unextracted.parent / self.name))

    def _mark(self, name: str) -> Path:
        if is_supported_archive(self):
            return DATA_DIR / "Marks" / self.stem / f"{name}.txt"
        return DATA_DIR / "Marks" / self.name / f"{name}.txt"


def _run_mokuro(run: Callable[..., None], chapter: Path | str) -> None:
    from mokuro.volume import Volume

    def patched(self: Volume):
        assert self.path_in.is_dir()
        # Make it non-recursive (we handle that) and handle more image formats
        img_paths = natsorted(
            p.relative_to(self.path_in) for p in self.path_in.glob("*")
            if p.is_file() and is_web_image(p)
        )
        return {p.with_suffix(""): p for p in img_paths}

    Volume.get_img_paths = patched

    with wakepy.keep.running():
        # Will continue any uncompleted work or exit early if already processed
        run(str(chapter), disable_confirmation=True, legacy_html=False)


def queue_loop(stop: Event) -> None:
    from mokuro.run import run  # slow
    current: tuple[MPath, multiprocessing.Process] | None = None

    while not stop.is_set():
        time.sleep(0.5)

        if pause_queue:
            if current:
                current[1].terminate()  # can resume later with mokuro's cache
                current = None
            continue

        if current and current[1].exitcode is not None:
            if current[0].ocr_json_file.exists():  # cache no longer needed
                with suppress(OSError):
                    shutil.rmtree(wip := current[0].ocr_wip_dir)
                    wip.parent.rmdir()

            OCR_QUEUE.popleft()
            current = None

        if current and (not OCR_QUEUE or current[0] != OCR_QUEUE[0]):
            current[1].terminate()
            current = None

        if not current and OCR_QUEUE:
            chapter = OCR_QUEUE[0].unextracted
            proc = multiprocessing.Process(
                target=_run_mokuro, args=[run, chapter], daemon=True,
            )
            proc.start()
            current = (chapter, proc)


def trim_archive_cache(stop: Event) -> None:
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR, ignore_errors=True)

    while not stop.is_set():
        long_ago = datetime.now() - timedelta(hours=2)
        for folder, date in LAST_ARCHIVE_ACCESSES.copy().items():
            if date < long_ago:
                shutil.rmtree(folder, ignore_errors=True)
                del LAST_ARCHIVE_ACCESSES[folder]
        time.sleep(1)


@router.get("/pause")
async def toggle_pause_queue() -> Response:
    global pause_queue  # noqa: PLW0603
    pause_queue = not pause_queue
    return RedirectResponse(url="/jobs", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/start/{chapter:path}")
async def start_ocr(
    chapter: Path | str,
    keep_going: bool = False,
    recursive: bool = False,
    prioritize: bool = False,
    sort: str = "",
    referer: str = "/",
) -> Response:
    job = MPath(chapter).unextracted
    jobs = [job, *job.next_chapters(sort)] if keep_going else [job]
    jobs = flatten(f.glob("**/") if recursive else [f] for f in jobs)
    jobs = [f for f in jobs if not f.ocr_json_file.exists()]
    (OCR_QUEUE.extendleft if prioritize else OCR_QUEUE.extend)(jobs)
    return RedirectResponse(url=referer, status_code=status.HTTP_303_SEE_OTHER)


@router.get("/cancel/{chapter:path}")
async def cancel_ocr(chapter: Path | str, recursive: bool = False) -> Response:
    job = MPath(chapter).unextracted
    OCR_QUEUE.remove(job)

    if recursive:
        queue = OCR_QUEUE.copy()
        OCR_QUEUE.clear()
        OCR_QUEUE.extend(j for j in queue if job not in j.parents)

    return RedirectResponse(url="/jobs", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/clear")
async def cancel_all_ocr() -> Response:
    OCR_QUEUE.clear()
    return RedirectResponse(url="/jobs", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/move/end/{chapter:path}")
async def move_ocr_job_position_end(chapter: Path | str) -> Response:
    return await move_ocr_job_position(chapter, len(OCR_QUEUE))


@router.get("/move/{to}/{chapter:path}")
async def move_ocr_job_position(chapter: Path | str, to: int) -> Response:
    job = MPath(chapter).unextracted
    del OCR_QUEUE[OCR_QUEUE.index(job)]
    OCR_QUEUE.insert(to, job)
    return RedirectResponse(url="/jobs", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/shift/{by}/{chapter:path}")
async def shift_ocr_job_position(chapter: Path | str, by: int) -> Response:
    job = MPath(chapter).unextracted
    del OCR_QUEUE[index := OCR_QUEUE.index(job)]
    OCR_QUEUE.insert(index + by, job)
    return RedirectResponse(url="/jobs", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/edit")
async def manual_edit_ocr_queue(content: str) -> Response:
    OCR_QUEUE.clear()
    jobs = (MPath(x).unextracted for x in content.splitlines())
    OCR_QUEUE.extend(j for j in jobs if j.exists())
    return RedirectResponse(url="/jobs", status_code=status.HTTP_303_SEE_OTHER)
