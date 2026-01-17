import re
from typing import List, Sequence

import pandas as pd


DEFAULT_FOCUS_TAGS: Sequence[str] = (
    "math",
    "graphs",
    "strings",
    "number theory",
    "trees",
    "geometry",
    "games",
    "probabilities",
)  # the eight tags we'll focus on


def preprocess_description(text: str) -> str:
    """
    Preprocess problem description text.
    
    - Removes LaTeX delimiters ($$$) to avoid polluting TF-IDF vocabulary
    - Normalizes whitespace (multiple spaces -> single space)
    - Strips leading/trailing whitespace
    
    Args:
        text: Raw text from problem description
    
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    # Remove $$$ delimiters (replace with space to avoid concatenating words)
    text = text.replace("$$$", " ")
    
    # Normalize whitespace: replace multiple spaces/tabs/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text


def preprocess_code(code: str) -> str:
    """
    Preprocess source code text.
    
    - Normalizes whitespace (multiple spaces -> single space)
    - Strips leading/trailing whitespace
    
    Args:
        code: Raw source code
    
    Returns:
        Preprocessed code
    """
    if not code:
        return ""
    
    # Normalize whitespace: replace multiple spaces/tabs/newlines with single space
    code = re.sub(r'\s+', ' ', code)
    
    # Remove leading and trailing whitespace
    code = code.strip()
    
    return code


def filter_to_focus_tags(df: pd.DataFrame, focus_tags: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Keep only the requested tags in the ``tags`` column.
    
    Args:
        df: DataFrame with 'tags' column
        focus_tags: List of tags to keep. If None, uses DEFAULT_FOCUS_TAGS
    
    Returns:
        DataFrame with filtered tags
    """
    if focus_tags is None:
        focus_tags = DEFAULT_FOCUS_TAGS
    focus_set = set(focus_tags)

    out = df.copy()
    out["tags"] = out["tags"].apply(lambda x: [t for t in x if t in focus_set])
    return out


def remove_empty_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that have no tags (empty tag list).
    
    Args:
        df: DataFrame with 'tags' column
    
    Returns:
        DataFrame with rows without tags removed
    """
    return df[df["tags"].map(len) > 0].copy()