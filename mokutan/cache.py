from __future__ import annotations

import asyncio
import gzip as gz
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from .utils import log

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class Cache[K, V](ABC):
    path: ClassVar[Path]
    version: ClassVar[int]

    data: dict[K, V] = field(default_factory=dict)
    changed: bool = False

    @abstractmethod
    def load_data(self, data: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def dump_data(self) -> dict[str, Any]:
        ...

    def load(self) -> bool:
        if not self.path.exists():
            return False

        try:
            data = json.loads(gz.decompress(self.path.read_bytes()))
            if data.pop("__version__") != self.version:
                log.warning("Difficulty cache version mismatch, ignoring it")
                return False
            self.load_data(data)
        except Exception:  # noqa: BLE001
            log.exception("Failed reading %s", self.path)
            return False

        return True

    def save(self) -> bool:
        if not self.changed:
            return False

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            dump = self.dump_data()
            dump["__version__"] = self.version
            data = json.dumps(dump, ensure_ascii=False).encode()
            self.path.write_bytes(gz.compress(data, 3))
            self.changed = False
        except Exception:  # noqa: BLE001
            log.exception("Failed saving %s", self.path)
            return False
        return True

    async def keep_saving(self, interval: float = 20) -> None:
        while True:
            await asyncio.sleep(interval)
            self.save()
