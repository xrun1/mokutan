from __future__ import annotations

import json
import math
import multiprocessing
import os
import shutil
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Self

from fastapi.responses import RedirectResponse
from natsort import natsorted

from .utils import TEMP, is_supported_archive, is_web_image, log

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from threading import Event

from fastapi import APIRouter, Response, status

from .utils import Point, flatten, get_sorted_dir

OCR_QUEUE: deque[OCRJob] = deque()
pause_queue: bool = False
router = APIRouter(prefix="/ocr")


@dataclass(slots=True)
class OCRBox:
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0
    vertical: bool = False
    font_size: int = 0
    lines: list[str] = field(default_factory=list)


class OCRJob(Path):
    @property
    def wip_folder(self) -> Self:
        return self.origin / "_ocr" / self.name

    @property
    def final_file(self) -> Self:
        return self.origin / (self.name + ".mokuro")

    @property
    def origin(self) -> Self:
        if TEMP not in self.parents:
            return self.parent

        path = str(self.parent).removeprefix(str(TEMP) + os.sep)
        if os.name == "nt":
            path = path[0] + ":" + path[1:]

        print(path)
        return type(self)(path) if Path(path).exists() else self.parent

    @property
    def previous_jobs(self) -> Sequence[Self]:
        if self.origin is self:  # drive root
            return []
        folders = [
            p for p in get_sorted_dir(self.origin)
            if p.is_dir() or is_supported_archive(p)
        ]
        return folders[:folders.index(self.origin / self.name)]

    @property
    def next_jobs(self) -> list[Self]:
        if self.origin is self:  # drive root
            return []
        folders = [
            p for p in get_sorted_dir(self.origin)
            if p.is_dir() or is_supported_archive(p)
        ]
        return folders[folders.index(self.origin / self.name) + 1:]

    @property
    def previous_job(self) -> Self | None:
        try:
            return self.previous_jobs[-1]
        except IndexError:
            return None

    @property
    def next_job(self) -> Self | None:
        try:
            return self.next_jobs[0]
        except IndexError:
            return None

    @property
    def queue_position(self) -> int | None:
        try:
            return OCR_QUEUE.index(self)
        except ValueError:
            return None

    @property
    def progress(self) -> tuple[int, int]:
        total = len(self.images)
        if self.final_file.exists():
            return (total, total)
        if not self.wip_folder.exists():
            return (0, total)
        return (len(list(self.wip_folder.glob("*.json"))), total)

    @property
    def progress_text(self) -> str:
        return "/".join(map(str, self.progress))

    @property
    def paused(self) -> bool:
        return pause_queue and bool(OCR_QUEUE) and OCR_QUEUE[0] == self

    @property
    def running(self) -> bool:
        return not pause_queue and bool(OCR_QUEUE) and OCR_QUEUE[0] == self

    @property
    def complete(self) -> bool:
        done, total = self.progress
        return done >= total

    @property
    def images(self) -> list[Path]:
        return [p for p in get_sorted_dir(self) if is_web_image(p)]

    @property
    def non_images(self) -> list[tuple[Path, Path | None]]:
        def thumb(p: Path) -> Path | None:
            try:
                return self._thumbnail(p)
            except OSError:
                log.exception("Error trying to thumbnail %s", p)
                return None

        return [
            (p, thumb(p)) for p in get_sorted_dir(self) if not (
                is_web_image(p) or p.suffix == ".mokuro" or p.name == "_ocr"
            )
        ]

    def boxes(self, image: Path) -> Iterable[OCRBox]:
        if self.final_file.exists():
            data = json.load(self.final_file.open(encoding="utf-8"))
        elif self.wip_folder.exists():
            data = {"pages": [
                json.load(f.open(encoding="utf-8")) | {"img_path": f.name}
                for f in
                natsorted(self.wip_folder.glob("*.json"), key=lambda f: f.name)
            ]}
        else:
            return

        if not (page := next((
            p for p in data["pages"] if Path(p["img_path"]).stem == image.stem
        ), None)):
            return

        page_w, page_h = page["img_width"], page["img_height"]

        for block in page["blocks"]:
            # left, top, right, bottom = block["box"]
            lines, coords = block["lines"], block["lines_coords"]

            boxes = []
            prev_start = prev_end = Point(math.inf, math.inf)

            # Split OCR-detected boxes that are probably multiple
            # multiple actual boxes in the image based on text spacing
            for line, coord in zip(lines, coords, strict=True):
                start, _top_right, end, _bot_left = starmap(Point, coord)

                if block["vertical"]:
                    new_box = (
                        abs(start.y - prev_start.y) > page_h / 100 or
                        abs(end.x - prev_start.x) > page_w / 50
                    )
                else:
                    new_box = (
                        abs(prev_start.x - start.x) > page_w / 100 or
                        abs(prev_end.y - start.y) > page_h / 100
                    )

                if new_box:
                    boxes.append(OCRBox(
                        x=start.x,
                        y=start.y,
                        vertical=block["vertical"],
                        font_size=block["font_size"] / 14,
                    ))

                box = boxes[-1]
                box.lines.append(line)
                box.x = min(box.x, start.x)
                box.y = min(box.y, start.y)

                if box.vertical:
                    box.w += end.x - start.x
                    box.h = max(box.h, end.y - start.y)
                else:
                    box.w = max(box.w, end.x - start.x)
                    box.h += end.y - start.y

                prev_start, prev_end = start, end

            for box in boxes:  # Convert to 0-1 percentages
                box.x /= page_w
                box.y /= page_h
                box.w /= page_w
                box.h /= page_h
                yield box

    def _thumbnail(self, path: Path, recurse: int = 2) -> Path | None:
        if is_web_image(path):
            return path

        if path.is_dir():
            items = path.iterdir() if recurse else path.glob("*/")
            dirs_tried = 0

            for child in natsorted(items, key=lambda c: (c.is_dir(), c.name)):
                if (thumb := self._thumbnail(child, max(0, recurse - 1))):
                    return thumb
                if child.is_dir():
                    dirs_tried += 1
                if dirs_tried >= 3:
                    break

        return None


def queue_loop(stop: Event) -> None:
    from mokuro.run import run  # slow
    current: tuple[OCRJob, multiprocessing.Process] | None = None

    while not stop.is_set():
        time.sleep(0.5)

        if pause_queue:
            if current:
                current[1].terminate()  # can resume later with mokuro's cache
                current = None
            continue

        if current and current[1].exitcode is not None:
            if current[0].final_file.exists():  # cache no longer needed
                with suppress(OSError):
                    shutil.rmtree(wip := current[0].wip_folder)
                    wip.origin.rmdir()

            OCR_QUEUE.popleft()
            current = None

        if current and (not OCR_QUEUE or current[0] != OCR_QUEUE[0]):
            current[1].terminate()
            current = None

        if not current and OCR_QUEUE:
            job = OCR_QUEUE[0]
            proc = multiprocessing.Process(
                target=run,  # maintains a cache, exits early if already OCR'ed
                args=[str(job)],
                kwargs={"disable_confirmation": True, "legacy_html": False},
                daemon=True,
            )
            proc.start()
            current = (job, proc)


@router.get("/pause")
async def toggle_pause_queue() -> Response:
    global pause_queue  # noqa: PLW0603
    pause_queue = not pause_queue
    return RedirectResponse(url="/jobs", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/start/{folder:path}")
async def start_ocr(
    folder: str,
    keep_going: bool = False,
    recursive: bool = False,
    prioritize: bool = False,
    referer: str = "/",
) -> Response:
    job = OCRJob(folder)
    jobs = job.next_jobs if keep_going else [job]
    jobs = flatten(f.glob("**/") if recursive else [f] for f in jobs)
    (OCR_QUEUE.extendleft if prioritize else OCR_QUEUE.extend)(jobs)
    return RedirectResponse(url=referer, status_code=status.HTTP_303_SEE_OTHER)


@router.get("/cancel/{folder:path}")
async def cancel_ocr(folder: str, recursive: bool = False) -> Response:
    job = OCRJob(folder)
    OCR_QUEUE.remove(job)

    if recursive:
        queue = OCR_QUEUE.copy()
        OCR_QUEUE.clear()
        OCR_QUEUE.extend(j for j in queue if OCRJob(folder) not in j.parents)

    return RedirectResponse(url="/jobs", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/clear")
async def cancel_all_ocr() -> Response:
    OCR_QUEUE.clear()
    return RedirectResponse(url="/jobs", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/move/end/{folder:path}")
async def move_ocr_job_position_end(folder: str) -> Response:
    return await move_ocr_job_position(folder, len(OCR_QUEUE))


@router.get("/move/{to}/{folder:path}")
async def move_ocr_job_position(folder: str, to: int) -> Response:
    job = OCRJob(folder)
    del OCR_QUEUE[OCR_QUEUE.index(job)]
    OCR_QUEUE.insert(to, job)
    return RedirectResponse(url="/jobs", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/shift/{by}/{folder:path}")
async def shift_ocr_job_position(folder: str, by: int) -> Response:
    job = OCRJob(folder)
    del OCR_QUEUE[index := OCR_QUEUE.index(job)]
    OCR_QUEUE.insert(index + by, job)
    return RedirectResponse(url="/jobs", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/edit")
async def manual_edit_ocr_queue(content: str) -> Response:
    OCR_QUEUE.clear()
    jobs = (OCRJob(x) for x in content.splitlines())
    OCR_QUEUE.extend(j for j in jobs if j.exists())
    return RedirectResponse(url="/jobs", status_code=status.HTTP_303_SEE_OTHER)
