from __future__ import annotations

import asyncio
import gzip as gz
import html
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from functools import lru_cache
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self
from zipfile import ZipFile

import fugashi
import httpx
import unidic
from cihai.core import Cihai
from fastapi import APIRouter, Request, Response, status
from fastapi.datastructures import URL

from mokutan.utils import DATA_DIR

from . import misc
from .utils import CACHE_DIR, ReferrerRedirect, flatten, log

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from fugashi.fugashi import Node as FugashiNode


jp_parser: fugashi.Tagger | None = None  # dict may not be downloaded yet
kanji_parser: Cihai | None = None  # same for DB
jp_freqs: dict[str | tuple[str, str], int] = {}

AVG_KANJI_STROKES = 11
NON_CORE_POS1_DIFFICULTY_FACTORS = {
    "補助記号": 0,  # Punctuation, brackets...
    "記号": 0,  # Symbols
    "空白": 0,  # Whitespace
    "助詞": 0.1,  # Particles
    "感動詞": 0.4,  # Interjections (あっ、えっ...)
    "接頭辞": 0.8,  # Prefixes
    "接尾辞": 0.8,  # Suffixes
    "助動詞": 0.8,  # Helper verbs
    "未知語": 2.0,  # Unknown/error
}

sentence_bound = \
    re.compile(r"(?:\n\n|[‥…。！？.!?])+")  # noqa: RUF001

http = httpx.AsyncClient(follow_redirects=True)

anki_router = APIRouter(prefix="/anki")


class AnkiError(RuntimeError): ...
class AnkiPermissionError(AnkiError): ...  # noqa: E302


@dataclass(slots=True)
class Anki:
    DEFAULT_API: ClassVar[URL] = URL("http://localhost:8765")
    MATURE_THRESHOLD: ClassVar[timedelta] = timedelta(days=21)
    SAVE_FILE: ClassVar[Path] = DATA_DIR / "Anki.json"

    api: URL = field(default_factory=lambda: Anki.DEFAULT_API)
    key: str = ""
    loaded: bool = False
    decks: list[str] = field(default_factory=list)
    note_types: list[str] = field(default_factory=list)
    note_fields: set[str] = field(default_factory=set)
    filters: list[tuple[str, str, str]] = field(default_factory=list)
    intervals: dict[str, timedelta] = field(default_factory=dict)

    @property
    def learned(self) -> set[str]:
        return {term for term, i in self.intervals.items() if i}

    @property
    def young(self) -> set[str]:
        return {
            term for term, i in self.intervals.items()
            if i and i < self.MATURE_THRESHOLD
        }

    @property
    def mature(self) -> set[str]:
        return {
            term for term, i in self.intervals.items()
            if i >= self.MATURE_THRESHOLD
        }

    async def do(self, action: str, **params: Any) -> Any:
        body = {"action": action, "version": 6, "params": params}
        if self.key:
            body["key"] = self.key
        resp = await http.post(str(self.api), json=body)
        resp.raise_for_status()
        data = resp.json()
        if data["error"]:
            raise AnkiError(f"{action}: {params}: {data['error']}")
        return data["result"]

    async def load(self) -> Self:
        result = await self.do("requestPermission")

        if result.get("permission") != "granted":
            raise AnkiPermissionError("Permission denied from Anki")

        if result.get("requireApiKey") and not self.key:
            raise AnkiPermissionError("This Anki requires an API key")

        assert jp_parser

        self.decks = await self.do("deckNames")
        self.note_types = await self.do("modelNames")

        note_fields = set()
        for fields in (await asyncio.gather(*[
            self.do("modelFieldNames", modelName=n) for n in self.note_types
        ])):
            note_fields.update(fields)
        self.note_fields = note_fields

        intervals = {}
        for deck, note_type, card_field in self.filters:
            query = f"deck:{json.dumps(deck)} note:{json.dumps(note_type)}"
            ids: list[int] = await self.do("findCards", query=query)
            info: list[dict[str, Any]] = await self.do("cardsInfo", cards=ids)

            for card in info:
                iv = card["interval"]

                for part in jp_parser(card["fields"][card_field]["value"]):
                    if not part.feature.orthBase:
                        continue

                    k = str(part.feature.orthBase)
                    v = timedelta(seconds=iv) if iv < 0 else timedelta(days=iv)

                    if k not in intervals or intervals[k] < v:
                        intervals[k] = v

        learned_before = self.learned
        self.intervals = intervals

        if learned_before != self.learned:
            ivs = {t: i.total_seconds() for t, i in intervals.items()}.items()
            for *_, diff in Difficulty.cache.values():
                diff_ivs = {
                    term: iv for term, iv in diff.anki_intervals.items()
                    if iv or term in intervals
                }.items()
                if not diff_ivs <= ivs:  # <= here checks if A is a subset of B
                    diff.anki_changed = True

        self.loaded = True
        return self

    async def safe_load(self) -> Self:
        try:
            await self.load()
        except httpx.ConnectError:
            log.info("AnkiConnect API at %r not reachable", self.api)
        except (httpx.HTTPError, AnkiError) as e:
            log.warning("AnkiConnect API: %r", e)
        return self

    async def keep_updated(self, interval: float = 600) -> None:
        while True:
            await asyncio.sleep(interval)
            await self.safe_load()

    async def add_filter(
        self, field: str, note_type: str = "*", deck: str = "*",
    ) -> Self:
        self.filters.append((deck, note_type, field))
        if self.loaded:
            return await self.load()
        return self

    async def delete_filter(self, index: int) -> Self:
        del self.filters[index]
        if self.loaded:
            return await self.load()
        return self

    def html_mark_known(self, text: str) -> Iterable[str]:
        assert jp_parser
        for part in jp_parser(text):
            iv = self.intervals.get(part.feature.orthBase)
            clean = html.escape(part.surface)
            pos1 = part.feature.pos1

            if iv is None or pos1 in NON_CORE_POS1_DIFFICULTY_FACTORS:
                yield clean
            elif not iv:
                yield f"<span class=anki-new>{clean}</span>"
            elif iv < self.MATURE_THRESHOLD:
                yield f"<span class=anki-young>{clean}</span>"
            else:
                yield f"<span class=anki-mature>{clean}</span>"

    def save(self) -> Self:
        self.SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.SAVE_FILE.write_text(json.dumps({
            "api": str(self.api),
            "key": self.key,
            "filters": self.filters,
        }, ensure_ascii=False, indent=4), encoding="utf-8")
        return self

    @classmethod
    def restore_saved(cls) -> Self:
        try:
            return cls(**json.loads(cls.SAVE_FILE.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return cls()
        except Exception:  # noqa: BLE001
            log.exception("Failed reading %s", cls.SAVE_FILE)

        return cls()


ANKI = Anki.restore_saved()


@dataclass(slots=True)
class CachedFugashiNode:  # fugashi's Node class can't be dumped to json
    surface: str
    orthBase: str  # noqa: N815
    lemma: str
    pos1: str

    @property
    def feature(self) -> Self:
        return self

    @classmethod
    def from_node(cls, node: fugashi.Node | CachedFugashiNode) -> Self:
        feat = node.feature
        return cls(node.surface, feat.orthBase, feat.lemma, feat.pos1)


@dataclass(slots=True)
class Difficulty:
    cache: ClassVar[dict[Path, tuple[float, int, Self]]] = {}
    cache_changed: ClassVar[bool] = False
    cache_path: ClassVar[Path] = CACHE_DIR / "Difficulty.json.gz"
    cache_version: ClassVar[int] = 5

    pages: list[list[list[CachedFugashiNode]]] = field(default_factory=list)
    page_scores: list[float] = field(default_factory=list)
    unique_terms: int = 0
    anki_learned: int = 0
    anki_mature: int = 0
    anki_score_decrease: float = 0
    anki_intervals: dict[str, float] = field(default_factory=dict)
    anki_changed: bool = False

    def __bool__(self) -> bool:
        return bool(self.score)

    @property
    def score(self) -> float:
        return sum(self.page_scores) / (len(self.page_scores) or 1)

    @property
    def raw_score(self) -> float:
        return self.score + self.anki_score_decrease

    @property
    def sentences_per_page(self) -> float:
        return sum(len(p) for p in self.pages) / (len(self.pages) or 1)

    @property
    def terms_per_sentence(self) -> float:
        sentences = flatten(self.pages)
        return sum(len(s) for s in sentences) / (len(sentences) or 1)

    @property
    def anki_learned_percent(self) -> float:
        return self.anki_learned / (self.unique_terms or 1) * 100

    @property
    def anki_mature_percent(self) -> float:
        return self.anki_mature / (self.unique_terms or 1) * 100

    @property
    def anki_score_decrease_percent(self) -> float:
        return self.anki_score_decrease / (self.raw_score or 1) * 100

    @classmethod
    def calculate(cls, ocr_json: Path) -> Self:
        if not (jp_parser and jp_freqs and ocr_json.exists()):
            return cls()

        fs_stats = ocr_json.stat()
        cached: Self | None = None

        if ocr_json in cls.cache:
            mtime, size, cached = cls.cache[ocr_json]
            if not cached.anki_changed:
                if mtime == fs_stats.st_mtime and size == fs_stats.st_size:
                    return cached

        pages: list[list[list[CachedFugashiNode]]] = []

        if cached:
            log.info("Updating cached difficulty for %s", ocr_json)
            pages = cached.pages
        else:
            log.info("Calculating difficulty for %s", ocr_json)

            for page in json.load(ocr_json.open(encoding="utf-8"))["pages"]:
                pages.append([])
                for box in page["blocks"]:
                    for snt in sentence_bound.split("".join(box["lines"])):
                        pages[-1].append(list(map(
                            CachedFugashiNode.from_node, jp_parser(snt),
                        )))

        dedup_counts = Counter(
            term.feature.orthBase
            for page in pages
            for sentence in page
            for term in sentence
            if term.feature.orthBase
        )
        intervals: dict[str, timedelta] = {}
        anki_not_found: set[str] = set()
        anki_bonuses: list[float] = []

        def get_score(
            page: list[list[CachedFugashiNode]],
            sentence: list[CachedFugashiNode],
            term: CachedFugashiNode,
        ) -> float:
            feat = term.feature

            def get(consider_rarity: bool = True) -> float:
                minimum = 20_000
                base = minimum + int(consider_rarity) * min(minimum, (
                    jp_freqs.get((feat.lemma, feat.orthBase)) or
                    jp_freqs.get(feat.orthBase) or
                    jp_freqs.get(feat.lemma) or
                    math.inf
                ))
                count = max(1, dedup_counts[feat.orthBase])
                return (
                    max(base / 10, base - base / 10 * (count - 1) ** 1.5)
                    * NON_CORE_POS1_DIFFICULTY_FACTORS.get(feat.pos1, 1)
                    * script_difficulty(term.surface)
                    * len(sentence) ** 1.15
                    * len(page) ** 1.075
                ) / 250_000

            score = get()

            if (iv := ANKI.intervals.get(feat.orthBase)):
                intervals[feat.orthBase] = iv
                old_score = score
                score = get(consider_rarity=False)
                score -= score * 0.9 * min(1, iv / ANKI.MATURE_THRESHOLD)
                anki_bonuses.append(old_score - score)
            else:
                anki_not_found.add(feat.orthBase)

            return score

        def page_score(page: list[list[CachedFugashiNode]]) -> float:
            return sum(
                get_score(page, sentence, term)
                for sentence in page
                for term in sentence
            )

        diff = cls(
            pages=pages,
            page_scores=[page_score(p) for p in pages],
            unique_terms=len(dedup_counts),
            anki_learned=len(intervals),
            anki_mature=len([
                iv for iv in intervals.values() if iv >= ANKI.MATURE_THRESHOLD
            ]),
            anki_score_decrease=sum(anki_bonuses) / (len(pages) or 1),
            anki_intervals={
                term: iv.total_seconds() for term, iv in intervals.items()
            } | dict.fromkeys(anki_not_found, 0),
        )
        cls.cache[ocr_json] = (fs_stats.st_mtime, fs_stats.st_size, diff)
        cls.cache_changed = True
        return diff

    @classmethod
    def from_cached(cls, **kwargs: Any) -> Self:
        pages = [
            [CachedFugashiNode(**node_kws) for node_kws in sentence]
            for page in kwargs.pop("pages")
            for sentence in page
        ]
        return cls(pages=pages, **kwargs)

    @classmethod
    def load_cache(cls) -> bool:
        if not cls.cache_path.exists():
            return False

        try:
            data = json.loads(gz.decompress(cls.cache_path.read_bytes()))
            if data.pop("__version__") != cls.cache_version:
                log.warning("Difficulty cache version mismatch, ignoring it")
                return False

            cls.cache |= {
                Path(path): (mtime, size, cls.from_cached(**diff))
                for path, (mtime, size, diff) in data.items()
            }
        except Exception:  # noqa: BLE001
            log.exception("Failed reading %s", cls.cache_path)
            return False

        return True

    @classmethod
    def save_cache(cls) -> bool:
        if not cls.cache_changed:
            return False

        try:
            cls.cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps({"__version__": cls.cache_version} | {
                str(path): [mtime, size, asdict(diff)]
                for path, (mtime, size, diff) in cls.cache.items()
            }, ensure_ascii=False).encode()
            cls.cache_path.write_bytes(gz.compress(data, 3))
            cls.cache_changed = False
        except Exception:  # noqa: BLE001
            log.exception("Failed saving %s", cls.cache_path)
            return False
        return True

    @classmethod
    async def keep_saving_cache(cls, interval: float = 20) -> None:
        while True:
            await asyncio.sleep(interval)
            cls.save_cache()


def load_dict_data() -> None:
    global kanji_parser  # noqa: PLW0603
    kanji_parser = Cihai()
    if not kanji_parser.unihan.is_bootstrapped:
        kanji_parser.unihan.bootstrap()

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


def is_hiragana(char: str) -> bool:
    return 0x3040 <= ord(char) <= 0x309F


def is_katakana(char: str) -> bool:
    return 0x30A0 <= ord(char) <= 0x30FF


def is_kanji(char: str) -> bool:
    return (
        (0x3400 <= (code := ord(char)) <= 0x4DBF) or
        (0x4E00 <= code <= 0x9FFF) or
        (0xF900 <= code <= 0xFAFF)
    )


@lru_cache(32768)
def kanji_stroke_count(char: str) -> int | None:
    assert kanji_parser
    if not (k := kanji_parser.unihan.lookup_char(char).first()):
        return None
    return int(k.kTotalStrokes)  # pyright:ignore[reportAttributeAccessIssue]


def script_difficulty(word: str) -> float:
    has_hira = has_kata = has_kanji = False
    score = 0

    for char in word:
        if is_hiragana(char):
            has_hira = True
            score += 0.3
        elif is_katakana(char):
            has_kata = True
            score += 0.4
        elif is_kanji(char):
            has_kanji = True
            strokes = kanji_stroke_count(char) or AVG_KANJI_STROKES
            score += max(0.5, 1.5 * (strokes / AVG_KANJI_STROKES))

    factor = len(word) ** 1.2

    if has_kanji and not (has_hira or has_kata):
        factor *= 1.3
    elif has_hira and not (has_kanji or has_kata):
        factor *= 0.5
    elif has_kata and not (has_kanji or has_hira):  # probably loanword or SFX
        factor *= 0.2

    return score / (len(word) or 1) * factor


@anki_router.get("/load")
async def anki_load(request: Request, api: str, key: str = "") -> Response:
    try:
        ANKI.api = URL(api)
        ANKI.key = key
        await ANKI.load()
    except AnkiPermissionError as e:
        return Response(str(e), status.HTTP_403_FORBIDDEN)
    ANKI.save()
    return ReferrerRedirect(request)


@anki_router.get("/filter/add")
async def anki_add_filter(
    request: Request, field: str, note_type: str = "*", deck: str = "*",
) -> Response:
    (await ANKI.add_filter(field, note_type, deck)).save()
    return ReferrerRedirect(request)


@anki_router.get("/filter/del")
async def anki_delete_filter(request: Request, index: int) -> Response:
    (await ANKI.delete_filter(index)).save()
    return ReferrerRedirect(request)
