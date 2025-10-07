from __future__ import annotations

import asyncio
import os
from abc import ABC
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, ClassVar

import jinja2
from fastapi import FastAPI, Request, status
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from natsort import natsorted

from . import DISPLAY_NAME, NAME, ocr
from .utils import (
    catch_log_exceptions,
    get_auto_extracted_path,
    is_web_image,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

LOADER = jinja2.PackageLoader(NAME, "templates")
ENV = jinja2.Environment(loader=LOADER, autoescape=jinja2.select_autoescape())
TEMPLATES = Jinja2Templates(env=ENV)
EXIT = Event()

if os.getenv("UVICORN_RELOAD"):
    # Fix browser reusing cached files at reload despite disk modifications
    StaticFiles.is_not_modified = lambda *_, **_kws: False


@dataclass(slots=True)
class Page(ABC):
    template: ClassVar[str]
    request: Request

    @property
    def response(self) -> Response:
        passthrough = {type, getattr}

        return TEMPLATES.TemplateResponse(self.template, {
            a: getattr(self, a) for a in dir(self)
            if not a.startswith("_") and a != "response"
        } | {
            x.__name__: x for x in passthrough
        } | {
            "DISPLAY_NAME": DISPLAY_NAME,
            "no_emoji": "&#xFE0E;",
        })

    @staticmethod
    def to_url(path: Path) -> str:
        # use .absolute() or first \ gets mangled on windows sometimes somehow
        return "/" + str(path.absolute().as_posix())


@dataclass(slots=True)
class Browse(Page):
    template: ClassVar[str] = "index.html.jinja"
    folder: ocr.OCRJob

    @property
    def ocr(self) -> ocr.OCRJob:
        return ocr.OCRJob(self.folder)


@dataclass(slots=True)
class WindowsDrives(Page):
    template: ClassVar[str] = "drives.html.jinja"

    @property
    def drives(self) -> list[Path]:
        return list(map(Path, os.listdrives()))


@dataclass(slots=True)
class Jobs(Page):
    template: ClassVar[str] = "jobs.html.jinja"

    @property
    def queue(self) -> Sequence[ocr.OCRJob]:
        return ocr.OCR_QUEUE

    @property
    def paused(self) -> bool:
        return ocr.pause_queue


@asynccontextmanager
async def life(_app: FastAPI):
    wrapped = catch_log_exceptions(ocr.queue_loop)
    task = asyncio.create_task(asyncio.to_thread(wrapped, EXIT))
    yield
    EXIT.set()
    await task


def mount(name: str) -> None:
    app.mount(f"/{name}", StaticFiles(packages=[(NAME, name)]), name=name)


app = FastAPI(default_response_class=HTMLResponse, lifespan=life, debug=True)
app.include_router(ocr.router)

list(map(mount, ["style"]))


@app.get("/jobs")
async def jobs(request: Request) -> Response:
    return Jobs(request).response


@app.get("/thumbnail/{path:path}")
async def thumbnail(path: Path, recurse: int = 2) -> Response:
    if is_web_image(path):
        return RedirectResponse(Page.to_url(path), status.HTTP_303_SEE_OTHER)

    path = await get_auto_extracted_path(path) or path

    if path.is_dir():
        items = path.iterdir() if recurse else path.glob("*/")
        dirs_tried = 0

        for child in natsorted(items, key=lambda c: (c.is_dir(), c.name)):
            thumb = await thumbnail(child, max(0, recurse - 1))
            if thumb.status_code != 404:
                return thumb
            if child.is_dir():
                dirs_tried += 1
            if dirs_tried >= 3:
                break

    return Response(status_code=404)


@app.get("/{folder:path}")
async def browse(request: Request, folder: str = "/") -> Response:
    if os.name == "nt" and folder in {"/", r"\\", ""}:
        return WindowsDrives(request).response

    path = Path(folder or "/")
    path = (await get_auto_extracted_path(path)) or path

    if path.is_file():
        return FileResponse(path)
    if path.is_dir():
        return Browse(request, ocr.OCRJob(path)).response

    return Response(status_code=404)
