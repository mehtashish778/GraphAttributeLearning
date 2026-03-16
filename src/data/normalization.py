from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set, Tuple


_NON_ALPHA_NUM = re.compile(r"[^a-z0-9 ]+")
_MULTI_SPACE = re.compile(r"\s+")


def simple_lemma(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def normalize_text(value: str, lowercase: bool = True, lemmatize: bool = True) -> str:
    text = value.strip()
    if lowercase:
        text = text.lower()
    text = _NON_ALPHA_NUM.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    if not text:
        return ""
    if lemmatize:
        text = " ".join(simple_lemma(part) for part in text.split(" "))
    return text


def normalize_label(
    label: str,
    lowercase: bool,
    lemmatize: bool,
    synonym_map: Dict[str, str],
) -> str:
    normalized = normalize_text(label, lowercase=lowercase, lemmatize=lemmatize)
    if not normalized:
        return ""
    if normalized in synonym_map:
        return synonym_map[normalized]
    return normalized


def normalize_labels(
    labels: Sequence[str],
    lowercase: bool,
    lemmatize: bool,
    synonym_map: Dict[str, str],
    keep_unmapped: bool,
    allowed_labels: Set[str],
) -> List[str]:
    out: List[str] = []
    for label in labels:
        normalized = normalize_label(
            label=label,
            lowercase=lowercase,
            lemmatize=lemmatize,
            synonym_map=synonym_map,
        )
        if not normalized:
            continue
        if normalized in allowed_labels or keep_unmapped:
            out.append(normalized)
    return sorted(set(out))


def filter_by_min_support(
    normalized_label_lists: Sequence[Sequence[str]],
    min_support: int,
) -> Tuple[List[List[str]], Dict[str, int]]:
    counter: Counter[str] = Counter()
    for labels in normalized_label_lists:
        counter.update(set(labels))

    keep_labels = {name for name, count in counter.items() if count >= min_support}
    filtered: List[List[str]] = []
    for labels in normalized_label_lists:
        filtered.append(sorted([label for label in set(labels) if label in keep_labels]))

    filtered_counter: Counter[str] = Counter()
    for labels in filtered:
        filtered_counter.update(set(labels))
    return filtered, dict(sorted(filtered_counter.items()))


def build_label_vocab(freq_map: Dict[str, int]) -> Dict[str, int]:
    labels = sorted(freq_map.keys())
    return {label: idx for idx, label in enumerate(labels)}
