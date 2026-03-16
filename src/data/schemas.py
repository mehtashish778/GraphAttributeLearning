from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class SampleRecord:
    image_id: int
    object_id: int
    object_name: str
    image_url: str
    image_path: str
    attributes_raw: List[str]
    attributes_norm: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
