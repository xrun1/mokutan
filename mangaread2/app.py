from __future__ import annotations

import asyncio
import os
from abc import ABC
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003
from threading import Event
from typing import TYPE_CHECKING, ClassVar
from urllib.parse import parse_qs
from zipfile import ZipFile

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

from mangaread2 import difficulty, io

from . import DISPLAY_NAME, NAME
from .io import EXTRACT_DIR, MPath
from .utils import (
    ReferrerRedirect,
    catch_log_exceptions,
    ellide,
    is_supported_archive,
    is_web_image,
)

if TYPE_CHECKING:
    from fastapi.datastructures import URL

LOADER = jinja2.PackageLoader(NAME, "templates")
ENV = jinja2.Environment(loader=LOADER, autoescape=jinja2.select_autoescape())
TEMPLATES = Jinja2Templates(env=ENV)
EXIT = Event()

NO_THUMB_ARCHIVES = set()

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
            "os_sep": os.sep,
            "anki": difficulty.ANKI,
            "ellide": ellide,
            "queue": io.OCR_QUEUE,
            "queue_paused": io.pause_queue,
        })

    @property
    def force_refresh_url(self) -> URL:
        return self.request.url.include_query_params(
            r=int((parse_qs(self.request.url.query).get("r") or ["0"])[0]) + 1,
        )


@dataclass(slots=True)
class Browse(Page):
    template: ClassVar[str] = "index.html.jinja"
    path: MPath
    sort: str = ""


@dataclass(slots=True)
class WindowsDrives(Page):
    template: ClassVar[str] = "drives.html.jinja"

    @property
    def drives(self) -> list[MPath]:
        return list(map(MPath, os.listdrives()))


@asynccontextmanager
async def life(_app: FastAPI):
    difficulty.Difficulty.load_cache()
    difficulty.load_dict_data()
    await difficulty.ANKI.safe_load()
    tasks = [
        asyncio.create_task(difficulty.ANKI.keep_updated()),
        asyncio.create_task(difficulty.Difficulty.keep_saving_cache()),
    ]
    tasks += [
        asyncio.create_task(asyncio.to_thread(catch_log_exceptions(f), EXIT))
        for f in (io.queue_loop, io.trim_archive_cache)
    ]
    yield
    EXIT.set()
    [t.cancel() for t in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)


def mount(name: str) -> None:
    app.mount(f"/{name}", StaticFiles(packages=[(NAME, name)]), name=name)


app = FastAPI(default_response_class=HTMLResponse, lifespan=life, debug=True)
app.include_router(io.router)
app.include_router(difficulty.anki_router)

list(map(mount, ["style"]))


@app.get("/thumbnail/{path:path}")
async def thumbnail(path: Path | str, recurse: int = 2) -> Response:
    path = MPath(path)

    if path.is_nt_drive:
        return Response(status_code=404)

    if is_web_image(path):
        return RedirectResponse(path.url(), status.HTTP_303_SEE_OTHER)

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
    elif is_supported_archive(path) and path not in NO_THUMB_ARCHIVES:
        if (base := EXTRACT_DIR / "Thumbnails" / path.name).exists():
            return await thumbnail(base, 99)

        with ZipFile(path) as arc:
            images = natsorted((
                f for f in arc.filelist
                if not f.is_dir() and is_web_image(f.orig_filename)
            ), key=lambda f: f.orig_filename)

            if images:
                base.mkdir(parents=True, exist_ok=True)
                arc.extract(images[0], base)
                return await thumbnail(base, 99)

            NO_THUMB_ARCHIVES.add(path)

    return Response(status_code=404)


@app.get("/mark/read/{path:path}")
async def mark_read(request: Request, path: Path) -> Response:
    MPath(path).mark_read()
    return ReferrerRedirect(request)


@app.get("/mark/unread/{path:path}")
async def mark_unread(request: Request, path: Path) -> Response:
    MPath(path).mark_unread()
    return ReferrerRedirect(request)


@app.get("/{path:path}")
async def browse(
    request: Request, path: Path | str = "/", sort: str = "",
) -> Response:
    if os.name == "nt" and str(path) in {"/", r"\\", ""}:
        return WindowsDrives(request).response

    path = await MPath(path or "/").extracted

    if path.is_file():
        return FileResponse(path)
    if path.is_dir():
        return Browse(request, MPath(path), sort).response

    return Response(status_code=404)
