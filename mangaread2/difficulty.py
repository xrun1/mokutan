from __future__ import annotations

import html
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from zipfile import ZipFile

import fugashi
import httpx
import unidic

from mangaread2 import misc

from .utils import (
    script_difficulty,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

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
class Difficulty:
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
    def calculate(cls, ocr_json_file: Path) -> Self:
        if not (jp_parser and jp_freqs and ocr_json_file.exists()):
            return cls()

        ocr_data = json.load(ocr_json_file.open(encoding="utf-8"))
        text = "\n\n".join(
            "\n".join(box["lines"])
            for page in ocr_data["pages"]
            for box in page["blocks"]
        )
        if not text.strip():
            return cls()
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

        return cls(
            score=adjust(sum(get_score(t) for t in unique_vocab.values())),
            unique_terms=len(unique_vocab),
            terms_per_page=avg_terms_per_page,
            anki_learned=len(intervals),
            anki_mature=len([
                iv for iv in intervals if iv >= ANKI_MATURE_THRESHOLD
            ]),
            anki_score_decrease=adjust(anki_bonus),
        )


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
