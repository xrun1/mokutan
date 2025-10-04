from __future__ import annotations

import asyncio
import os
from abc import ABC
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import ClassVar

import jinja2
from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import DISPLAY_NAME, NAME, ocr
from .utils import catch_log_exceptions

DEFAULT_PATH = str(Path.home())
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


@dataclass(slots=True)
class Browse(Page):
    template: ClassVar[str] = "index.html.jinja"
    folder: Path

    @property
    def ocr(self) -> ocr.OCRJob:
        return ocr.OCRJob(self.folder)


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


@app.get("/{folder:path}")
async def browse(request: Request, folder: str = DEFAULT_PATH) -> Response:
    path = Path(folder)
    if path.is_file():
        return FileResponse(path)
    if path.is_dir():
        return Browse(request, path).response
    return Response(status_code=404)
