import re
from typing import Sequence

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
)  # the eight tags we'll focus on according to the pdf


def preprocess_description(text: str) -> str:
    """
    Preprocess problem description text.
    
    - Removes Latex delimiters ($$$) 
    - Normalizes whitespace (multiple spaces -> single space)
    - Strips leading/trailing whitespace
    
    Args:
        text: the text from problem description
    
    Returns:
        Clean preprocessed text
    """
    if not text:
        return ""
    
    text = text.replace("$$$", " ")
    
    # replace multiple spaces/tabs/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    
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