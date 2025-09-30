from __future__ import annotations

import logging as log
import mimetypes
import multiprocessing
import os
import shutil
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

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
def get_sorted_dir(path: Path) -> list[Path]:
    return natsorted(path.iterdir(), key=lambda p: p.name)


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

    shutil.rmtree(folder.parent / "_ocr" / folder.name, ignore_errors=True)
    with suppress(OSError):  # not empty
        (folder.parent / "_ocr").rmdir()

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
        ocr_json = path.parent / f"{path.name}.mokuro"
        if ocr and path not in OCRING and not ocr_json.exists():
            tasks.add_task(do_ocr, path)
            OCRING.add(path)

        return Page(request, path, ocr).response

    return Response(status_code=404)
