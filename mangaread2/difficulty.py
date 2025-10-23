from __future__ import annotations

import asyncio
import html
import json
import math
from collections import Counter
from dataclasses import dataclass
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
anki_connected: URL | None = None
anki_api_key: str = ""
anki_decks: list[str] = []
anki_note_types: list[str] = []
anki_note_fields: set[str] = set()
anki_filters: list[tuple[str, str, str]] = []
anki_intervals: dict[str, timedelta] = {}

ANKI_DEFAULT_API = "http://localhost:8765"
ANKI_MATURE_THRESHOLD = timedelta(days=21)

anki_router = APIRouter(prefix="/anki")


class AnkiError(RuntimeError):
    ...


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

        avg_terms_per_page = len(terms) / len(ocr_data["pages"])

        def adjust(score_like: float) -> float:
            return score_like / len(unique_vocab) * avg_terms_per_page / 20_000

        diff = cls(
            score=adjust(sum(get_score(t) for t in unique_vocab.values())),
            unique_terms=len(unique_vocab),
            terms_per_page=avg_terms_per_page,
            anki_learned=len(intervals),
            anki_mature=len([
                iv for iv in intervals if iv >= ANKI_MATURE_THRESHOLD
            ]),
            anki_score_decrease=adjust(anki_bonus),
        )
        cls.cache[ocr_json] = (fs_stats.st_mtime, fs_stats.st_size, diff)
        return diff


def load_dict_data() -> None:
    global kanji_parser
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
async def anki_load(
    api: str = ANKI_DEFAULT_API, key: str = "", referer: str = "/",
) -> Response:
    async def anki(action: str, **params: Any) -> Any:
        body = {"action": action, "version": 6, "params": params}
        if key:
            body["key"] = key
        resp = await http.post(api, json=body)
        resp.raise_for_status()
        data = resp.json()
        if data["error"]:
            raise AnkiError(f"{action}: {params}: {data['error']}")
        return data["result"]

    result = await anki("requestPermission")

    if result.get("permission") != "granted":
        return Response("Permission denied from Anki", 403)

    if result.get("requireApiKey") and not key:
        return Response("This Anki requires an API key", 403)

    assert jp_parser
    learned_before = {term for term, iv in anki_intervals.items() if iv}
    anki_intervals.clear()
    anki_decks.clear()
    anki_note_types.clear()
    anki_note_fields.clear()

    anki_decks.extend(await anki("deckNames"))
    anki_note_types.extend(await anki("modelNames"))
    for fields in (await asyncio.gather(*[
        anki("modelFieldNames", modelName=note) for note in anki_note_types
    ])):
        anki_note_fields.update(fields)

    for deck, note_type, card_field in anki_filters:
        query = f"deck:{json.dumps(deck)} note:{json.dumps(note_type)}"
        ids: list[int] = await anki("findCards", query=query)
        info: list[dict[str, Any]] = await anki("cardsInfo", cards=ids)

        for card in info:
            iv = card["interval"]

            for part in jp_parser(card["fields"][card_field]["value"]):
                k = part.feature.orthBase
                v = timedelta(seconds=iv) if iv < 0 else timedelta(days=iv)

                if k not in anki_intervals or anki_intervals[k] < v:
                    anki_intervals[k] = v

    if learned_before != {term for term, iv in anki_intervals.items() if iv}:
        Difficulty.cache.clear()

    global anki_connected, anki_api_key  # noqa: PLW0603
    anki_connected = URL(api)
    anki_api_key = key
    return RedirectResponse(referer, status.HTTP_303_SEE_OTHER)


@anki_router.get("/filter/add")
async def anki_add_filter(
    deck: str, note_type: str, field: str, referer: str = "/",
) -> Response:
    assert anki_connected
    anki_filters.append((deck, note_type, field))
    return await anki_load(str(anki_connected), anki_api_key, referer)


@anki_router.get("/filter/del")
async def anki_delete_filter(index: int, referer: str = "/") -> Response:
    assert anki_connected
    del anki_filters[index]
    return await anki_load(str(anki_connected), anki_api_key, referer)
