from __future__ import annotations

import json
import logging as log
import math
import mimetypes
import multiprocessing
import os
import shutil
from contextlib import suppress
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import jinja2
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.datastructures import URL
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from natsort import natsorted

from . import DISPLAY_NAME, NAME

if TYPE_CHECKING:
    from collections.abc import Iterable

log.basicConfig(level=log.INFO)
log.getLogger("httpx").setLevel(log.WARNING)

DEFAULT_PATH = str(Path.home())
LOADER = jinja2.PackageLoader(NAME, "templates")
ENV = jinja2.Environment(loader=LOADER, autoescape=jinja2.select_autoescape())
TEMPLATES = Jinja2Templates(env=ENV)
app = FastAPI(default_response_class=HTMLResponse, debug=True)

OCRING: set[Path] = set()


def mount(name: str) -> None:
    app.mount(f"/{name}", StaticFiles(packages=[(NAME, name)]), name=name)


list(map(mount, ("scripts", "style")))
if os.getenv("UVICORN_RELOAD"):
    # Fix browser reusing cached files at reload despite disk modifications
    StaticFiles.is_not_modified = lambda *_, **_kws: False  # type: ignore


@lru_cache
def get_sorted_dir(folder: Path) -> list[Path]:
    return natsorted(folder.iterdir(), key=lambda p: p.name)


def get_wip_ocr_folder(folder: Path) -> Path:
    return folder.parent / "_ocr" / folder.name


def get_ocr_file(folder: Path) -> Path:
    return folder.parent / (folder.name + ".mokuro")


class Point(NamedTuple):
    x: float
    y: float


@dataclass(slots=True)
class OCRBox:
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0
    vertical: bool = False
    font_size: int = 0
    lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Page:
    request: Request
    folder: Path
    ocr: bool = False

    @property
    def title(self) -> str:
        return f"{self.folder.parent.name}: {self.folder.name}"

    @property
    def response(self) -> Response:
        passthrough = {type, getattr}

        return TEMPLATES.TemplateResponse("index.html.jinja", {
            a: getattr(self, a) for a in dir(self)
            if not a.startswith("_") and a != "response"
        } | {
            x.__name__: x for x in passthrough
        } | {
            "DISPLAY_NAME": DISPLAY_NAME,
            "no_emoji": "&#xFE0E;",
        })

    @property
    def ocr_running(self) -> bool:
        return self.folder in OCRING

    @property
    def images(self) -> Iterable[Path]:
        for entry in get_sorted_dir(self.folder):
            if (mimetypes.guess_type(entry)[0] or "").startswith("image/"):
                yield entry

    @property
    def previous_folder(self) -> Path | None:
        return self.adjacent_folder(-1)

    @property
    def next_folder(self) -> Path | None:
        return self.adjacent_folder(+1)

    def adjacent_folder(self, step: int) -> Path | None:
        if self.folder.parent is self.folder:  # drive root
            return None

        folders = [p for p in get_sorted_dir(self.folder.parent) if p.is_dir()]
        index = folders.index(self.folder) + step

        if index < 0 or index >= len(folders):
            return None

        return folders[index]

    def ocr_boxes(self, image: Path) -> Iterable[OCRBox]:
        if (final := get_ocr_file(self.folder)).exists():
            data = json.load(final.open(encoding="utf-8"))
        elif (wip := get_wip_ocr_folder(self.folder)).exists():
            data = {"pages": [
                json.load(f.open(encoding="utf-8")) | {"img_path": f.name}
                for f in natsorted(wip.glob("*.json"), key=lambda f: f.name)
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

    @staticmethod
    def local_url(url: Path | str | None) -> str:
        if not url:
            return "/"
        return str(URL(str(url)).replace(scheme="", netloc=""))


def do_ocr(folder: Path) -> None:
    from mokuro.run import run  # slow
    proc = multiprocessing.Process(  # HACK: prevent hanging on SIGINT
        target=run,  # maintains its own cache, exits early if already OCR'ed
        args=[str(folder)],
        kwargs={"disable_confirmation": True, "legacy_html": False,},
        daemon=True,
    )
    proc.start()
    proc.join()
    OCRING.remove(folder)

    # If no final file, process was probably interrupted so keep the cache
    if get_ocr_file(folder).exists():
        shutil.rmtree(wip := get_wip_ocr_folder(folder), ignore_errors=True)
        with suppress(OSError):  # not empty
            wip.parent.rmdir()


@app.get("/{rest:path}")
async def browse(
    request: Request,
    tasks: BackgroundTasks,
    rest: str = DEFAULT_PATH,
    ocr: bool = False,
) -> Response:
    path = Path(rest)

    if path.is_file():
        return FileResponse(path)

    if path.is_dir():
        if ocr and path not in OCRING and not get_ocr_file(path).exists():
            tasks.add_task(do_ocr, path)
            OCRING.add(path)

        return Page(request, path, ocr).response

    return Response(status_code=404)
