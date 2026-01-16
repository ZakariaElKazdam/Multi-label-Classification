#from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd


@dataclass
class ProblemExample:
    """Container for a single problem example from a JSON file."""

    uid: str
    text: str
    tags: List[str]


DEFAULT_FOCUS_TAGS: Sequence[str] = (
    "math",
    "graphs",
    "strings",
    "number theory",
    "trees",
    "geometry",
    "games",
    "probabilities",
)


def load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_dataset(directory: str) -> Iterable[ProblemExample]:
    """Yield all examples from ``sample_*.json`` files in a directory."""
    for name in sorted(os.listdir(directory)):
        if not name.startswith("sample_") or not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        record = load_json_file(path)

        # Use only the problem description
        text = record.get("prob_desc_description") or ""

        tags = record.get("tags") or []
        # Normalise to plain list of strings
        tags = [str(t) for t in tags]

        uid = record.get("src_uid") or record.get("code_uid") or name

        yield ProblemExample(uid=uid, text=text, tags=tags)


def load_dataset_as_dataframe(directory: str) -> pd.DataFrame:
    """Load the dataset into a DataFrame with ``uid``, ``text`` and ``tags``."""
    rows = []
    for ex in iter_dataset(directory):
        rows.append({"uid": ex.uid, "text": ex.text, "tags": ex.tags})
    return pd.DataFrame(rows)


def filter_to_focus_tags(df: pd.DataFrame, focus_tags: Sequence[str] | None = None) -> pd.DataFrame:
    """Keep only the requested tags in the ``tags`` column."""
    if focus_tags is None:
        focus_tags = DEFAULT_FOCUS_TAGS
    focus_set = set(focus_tags)

    def _filter(tags: List[str]) -> List[str]:
        return [t for t in tags if t in focus_set]

    out = df.copy()
    out["tags"] = out["tags"].apply(_filter)
    return out
