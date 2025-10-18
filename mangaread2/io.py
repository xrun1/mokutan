from __future__ import annotations

import asyncio
import html
import json
import math
import multiprocessing
import os
import shutil
import time
from collections import Counter, defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib import resources
from io import BytesIO
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4
from zipfile import ZipFile

from fastapi.datastructures import URL
import fugashi
import httpx
import unidic
import wakepy
from fastapi.responses import RedirectResponse
from natsort import natsorted

from . import misc
from .utils import (
    TEMP,
    is_supported_archive,
    is_web_image,
    log,
    script_difficulty,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from threading import Event

from fastapi import APIRouter, Response, status

from .utils import Point, flatten, get_sorted_dir

router = APIRouter(prefix="/ocr")

OCR_QUEUE: deque[MPath] = deque()
IGNORED_ARCHIVES: set[Path] = set()
LAST_ARCHIVE_ACCESSES: dict[Path, datetime] = {}
LOCKS: defaultdict[Path, asyncio.Lock] = defaultdict(asyncio.Lock)
pause_queue: bool = False

jp_parser: fugashi.Tagger | None = None  # dict may not be downloaded yet
jp_freqs: dict[str | tuple[str, str], int] = {}

NON_CORE_POS1_DIFFICULTY_FACTORS = {
    "助詞": 0,  # Particles
    "補助記号": 0,  # Punctuation, brackets...
    "記号": 0,  # Symbols
    "空白": 0,  # Whitespace
    "未知語": 0.1,  # Unknown/error
    "感動詞": 0.4,  # Interjections (あっ、えっ...)
    "接頭辞": 0.8,  # Prefixes
    "接尾辞": 0.8,  # Suffixes
    "助動詞": 0.8,  # Helper verbs
}

http = httpx.AsyncClient(follow_redirects=True)
anki_filters = [("Japanese::02 - Kaishi 1.5k", "Kaishi 1.5k", "Word")]
anki_intervals: dict[str, timedelta] = {}
ANKI_MATURE_THRESHOLD = timedelta(days=21)


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

        if (base := TEMP / fix_end(archive)).exists():
            LAST_ARCHIVE_ACCESSES[archive] = datetime.now()
            return type(self)(TEMP) / fix_end(self)

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
        return type(self)(TEMP) / fix_end(self)

    @property
    def unextracted(self) -> Self:
        if TEMP not in self.parents:
            return self

        path = str(self).removeprefix(str(TEMP) + os.sep)
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
    def difficulty(self) -> tuple[int, float, int, int, float, float] | None:
        if not (jp_parser and jp_freqs and self.ocr_json_file.exists()):
            return None

        ocr_data = json.load(self.ocr_json_file.open(encoding="utf-8"))
        text = "\n\n".join(
            "\n".join(box["lines"])
            for page in ocr_data["pages"]
            for box in page["blocks"]
        )
        if not text.strip():
            return None
        terms = jp_parser(text)
        unique_vocab = {t.feature.orthBase: t for t in terms}
        counts = Counter(t.feature.orthBase for t in terms)
        intervals: list[timedelta] = []
        anki_bonus = 0

        def get_score(t: fugashi.fugashi.UnidicNode) -> float:
            base = 50_000 + (
                jp_freqs.get((t.feature.lemma, t.feature.orthBase)) or
                jp_freqs.get(t.feature.orthBase) or
                jp_freqs.get(t.feature.lemma) or
                1000
            )

            if (iv := anki_intervals.get(t.feature.orthBase)):
                nonlocal anki_bonus
                intervals.append(iv)
                new_base = 10_000
                new_base -= new_base * (iv / ANKI_MATURE_THRESHOLD)
                new_base = max(new_base, 0)
                anki_bonus += base - new_base
                base = new_base

            return (
                base
                * NON_CORE_POS1_DIFFICULTY_FACTORS.get(t.feature.pos1, 1)
                * script_difficulty(t.surface)
                / max(1, counts[t.feature.orthBase])
            )

        score = sum(get_score(t) for t in unique_vocab.values())
        avg_terms_per_page = len(terms) / len(ocr_data["pages"])

        def adjust(score: float) -> float:
            return score / len(unique_vocab) * avg_terms_per_page / 20_000

        return (
            len(unique_vocab),
            adjust(score),
            len(intervals),
            len([iv for iv in intervals if iv >= ANKI_MATURE_THRESHOLD]),
            adjust(anki_bonus),
            avg_terms_per_page,
        )

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
                box.lines.append("".join(mark_anki_known_terms(line)))
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
        if sort == "m":
            return sorted(p, key=lambda e: e.stat().st_mtime)
        if sort == "d":
            return sorted(p, key=lambda e: (e.difficulty or (0, math.inf))[1])
        if sort == "p":
            return sorted(p, key=lambda e: len(e.images))
        return natsorted(p, key=lambda e: e.name)

    def _sibling_chapters(self, sort: str = "") -> tuple[list[Self], int]:
        if self.unextracted.parent is self:  # drive root
            return ([], -1)
        chapters = self._sort(sort, (
            p for p in self.unextracted.parent.iterdir()
            if p.is_dir() or is_supported_archive(p)
        ))
        return (chapters, chapters.index(self.unextracted.parent / self.name))


def _run_mokuro(run: Callable[..., None], chapter: Path | str) -> None:
    with wakepy.keep.running():
        # Will continue any uncompleted work or exit early if already processed
        run(str(chapter), disable_confirmation=True, legacy_html=False)


async def load_anki_data() -> None:
    async def do(action: str, **params: Any) -> Any:
        api = "http://localhost:8765"
        body = {"action": action, "version": 6, "params": params}
        got = (await http.post(api, json=body)).json()
        if got["error"]:
            raise RuntimeError(f"Anki: {action}: {params}: {got['error']}")
        return got["result"]

    assert jp_parser
    anki_intervals.clear()

    for deck, note_type, card_field in anki_filters:
        query = f"deck:{json.dumps(deck)} note:{json.dumps(note_type)}"
        ids: list[int] = await do("findCards", query=query)
        info: list[dict[str, Any]] = await do("cardsInfo", cards=ids)

        for card in info:
            iv = card["interval"]

            for part in jp_parser(card["fields"][card_field]["value"]):
                k = part.feature.orthBase
                v = timedelta(seconds=iv) if iv < 0 else timedelta(days=iv)

                if k not in anki_intervals or anki_intervals[k] < v:
                    anki_intervals[k] = v


def mark_anki_known_terms(text: str) -> Iterable[str]:
    assert jp_parser
    for part in jp_parser(text):
        iv = anki_intervals.get(part.feature.orthBase)
        clean = html.escape(part.surface)
        pos1 = part.feature.pos1

        if iv is None or pos1 in NON_CORE_POS1_DIFFICULTY_FACTORS:
            yield clean
        elif not iv:
            yield f"<span class=anki-new>{clean}</span>"
        elif iv < ANKI_MATURE_THRESHOLD:
            yield f"<span class=anki-young>{clean}</span>"
        else:
            yield f"<span class=anki-mature>{clean}</span>"


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
    if TEMP.exists():
        shutil.rmtree(TEMP, ignore_errors=True)

    while not stop.is_set():
        long_ago = datetime.now() - timedelta(hours=2)
        for folder, date in LAST_ARCHIVE_ACCESSES.copy().items():
            if date < long_ago:
                shutil.rmtree(folder, ignore_errors=True)
                del LAST_ARCHIVE_ACCESSES[folder]
        time.sleep(1)


def load_dict_data() -> None:
    if not Path(unidic.DICDIR).exists():
        print(unidic.DICDIR)
        from unidic.download import download_version
        download_version()

    global jp_parser  # noqa: PLW0603
    jp_parser = fugashi.Tagger("-Owakati")

    file = "JPDB_v2.2_Frequency_Kana_2024-10-13.zip"
    with (
        ZipFile(BytesIO(resources.read_binary(misc, file))) as zf,
        zf.open("term_meta_bank_1.json") as bank,
    ):
        raw = json.loads(bank.read().decode("utf-8"))

    out = {}
    for term, _, info in raw:
        if (freq := info.get("value")):
            k, v = (term, freq)
        elif info["frequency"]["displayValue"].endswith("㋕"):
            k, v = ((term, info["reading"]), info["frequency"]["value"])
        else:
            k, v = (term, info["frequency"]["value"])

        if v < out.get(k, math.inf):
            out[k] = v

    jp_freqs.update(out)


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
