from __future__ import annotations

import asyncio
import html
import json
import math
from collections import Counter
from dataclasses import dataclass, field
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
from fastapi import APIRouter, Response, status
from fastapi.datastructures import URL
from fastapi.responses import RedirectResponse

from mangaread2.utils import DATA_DIR

from . import misc

if TYPE_CHECKING:
    from collections.abc import Iterable

jp_parser: fugashi.Tagger | None = None  # dict may not be downloaded yet
kanji_parser: Cihai | None = None  # same for DB
jp_freqs: dict[str | tuple[str, str], int] = {}

AVG_KANJI_STROKES = 11
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
        self.loaded = False
        result = await self.do("requestPermission")

        if result.get("permission") != "granted":
            raise AnkiPermissionError("Permission denied from Anki")

        if result.get("requireApiKey") and not self.key:
            raise AnkiPermissionError("This Anki requires an API key")

        assert jp_parser
        learned_before = {term for term, i in self.intervals.items() if i}
        self.intervals.clear()
        self.note_fields.clear()

        self.decks = await self.do("deckNames")
        self.note_types = await self.do("modelNames")
        for fields in (await asyncio.gather(*[
            self.do("modelFieldNames", modelName=n) for n in self.note_types
        ])):
            self.note_fields.update(fields)

        for deck, note_type, card_field in self.filters:
            query = f"deck:{json.dumps(deck)} note:{json.dumps(note_type)}"
            ids: list[int] = await self.do("findCards", query=query)
            info: list[dict[str, Any]] = await self.do("cardsInfo", cards=ids)

            for card in info:
                iv = card["interval"]

                for part in jp_parser(card["fields"][card_field]["value"]):
                    k = part.feature.orthBase
                    v = timedelta(seconds=iv) if iv < 0 else timedelta(days=iv)

                    if k not in self.intervals or self.intervals[k] < v:
                        self.intervals[k] = v

        if learned_before != {term for term, i in self.intervals.items() if i}:
            Difficulty.cache.clear()

        self.loaded = True
        self.SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.SAVE_FILE.write_text(json.dumps({
            "api": str(self.api),
            "key": self.key,
            "filters": self.filters,
        }, ensure_ascii=False, indent=4), encoding="utf-8")
        return self

    async def add_filter(
        self, field: str, note_type: str = "*", deck: str = "*",
    ) -> Self:
        self.filters.append((deck, note_type, field))
        return await self.load()

    async def delete_filter(self, index: int) -> Self:
        del self.filters[index]
        return await self.load()

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

    @classmethod
    def restore_saved(cls) -> Self:
        if cls.SAVE_FILE.exists():
            return cls(**json.loads(cls.SAVE_FILE.read_text(encoding="utf-8")))
        return cls()


anki = Anki.restore_saved()


@dataclass(slots=True)
class Difficulty:
    cache: ClassVar[dict[Path, tuple[float, int, Self]]] = {}

    score: float = 0
    unique_terms: int = 0
    terms_per_page: float = 0
    anki_learned: int = 0
    anki_mature: int = 0
    anki_score_decrease: float = 0

    def __bool__(self) -> bool:
        return bool(self.score)

    @property
    def raw_score(self) -> float:
        return self.score + self.anki_score_decrease

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

        if ocr_json in cls.cache:
            mtime, size, diff = cls.cache[ocr_json]
            if mtime == fs_stats.st_mtime and size == fs_stats.st_size:
                return diff

        ocr_data = json.load(ocr_json.open(encoding="utf-8"))
        text = "\n\n".join(
            "\n".join(box["lines"])
            for page in ocr_data["pages"]
            for box in page["blocks"]
        )
        if not text.strip():
            cls.cache[ocr_json] = (fs_stats.st_mtime, fs_stats.st_size, cls())
            return cls.cache[ocr_json][2]

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

            if (iv := anki.intervals.get(t.feature.orthBase)):
                nonlocal anki_bonus
                intervals.append(iv)
                new_base = 10_000
                new_base -= new_base * (iv / anki.MATURE_THRESHOLD)
                new_base = max(new_base, 0)
                anki_bonus += base - new_base
                base = new_base

            return (
                base
                * NON_CORE_POS1_DIFFICULTY_FACTORS.get(t.feature.pos1, 1)
                * script_difficulty(t.surface)
                / max(1, counts[t.feature.orthBase])
            )

        avg_terms_per_page = len(terms) / len(ocr_data["pages"])

        def adjust(score_like: float) -> float:
            return score_like / len(unique_vocab) * avg_terms_per_page / 20_000

        diff = cls(
            score=adjust(sum(get_score(t) for t in unique_vocab.values())),
            unique_terms=len(unique_vocab),
            terms_per_page=avg_terms_per_page,
            anki_learned=len(intervals),
            anki_mature=len([
                iv for iv in intervals if iv >= anki.MATURE_THRESHOLD
            ]),
            anki_score_decrease=adjust(anki_bonus),
        )
        cls.cache[ocr_json] = (fs_stats.st_mtime, fs_stats.st_size, diff)
        return diff


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
    def score_char(char: str) -> float:
        if is_hiragana(char):
            return 0.3
        if is_katakana(char):
            return 0.4
        if is_kanji(char):
            strokes = kanji_stroke_count(char) or AVG_KANJI_STROKES
            return max(0.5, 1.5 * (strokes / AVG_KANJI_STROKES))
        return 0

    scores = [score_char(char) for char in word]
    factor = len(word) ** 1.2

    if set(scores) == {1}:  # all kanji
        factor *= 1.3
    elif set(scores) == {0.3}:  # all hiragana
        factor *= 0.5
    elif set(scores) == {0.4}:  # all katakana, probably loanword or SFX
        factor *= 0.1

    return sum(scores) / len(word) * factor


@anki_router.get("/load")
async def anki_load(api: str, key: str = "", referer: str = "/") -> Response:
    global anki  # noqa: PLW0603
    try:
        anki = await Anki(URL(api), key).load()
    except AnkiPermissionError as e:
        return Response(str(e), status.HTTP_403_FORBIDDEN)
    return RedirectResponse(referer, status.HTTP_303_SEE_OTHER)


@anki_router.get("/filter/add")
async def anki_add_filter(
    field: str, note_type: str = "*", deck: str = "*", referer: str = "/",
) -> Response:
    await anki.add_filter(field, note_type, deck)
    return RedirectResponse(referer, status.HTTP_303_SEE_OTHER)


@anki_router.get("/filter/del")
async def anki_delete_filter(index: int, referer: str) -> Response:
    await anki.delete_filter(index)
    return RedirectResponse(referer, status.HTTP_303_SEE_OTHER)
