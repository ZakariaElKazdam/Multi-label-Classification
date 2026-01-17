import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd


@dataclass
class ProblemExample:
    """Container for a single problem example from a JSON file"""

    problem_id: str
    text_description: str
    text_code: str
    tags: List[str]


def load_json_file(path: str) -> dict:
    """load the json file using its path"""  
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_dataset(
    directory: str,
    features: Optional[List[str]] = None,
) -> Iterable[ProblemExample]:
    """
    Yield all examples from ``sample_*.json`` files in a directory.
    
    Args:
        directory: Directory containing sample_*.json files
        features: List of features to load. Options: ["description"], ["code"], or ["description", "code"]
                  Default: ["description"]
    
    Yields:
        ProblemExample with text_description and text_code (empty string if feature not requested)
    """
    if features is None:
        features = ["description"]
    
    # Validate features
    valid_features = {"description", "code"}
    for feat in features:
        if feat not in valid_features:
            raise ValueError(f"Invalid feature: {feat}. Must be one of {valid_features}")
    
    use_description = "description" in features
    use_code = "code" in features
    
    for name in sorted(os.listdir(directory)): # alphabetical sort  
        if not name.startswith("sample_") or not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        record = load_json_file(path)

        # Load description if requested (raw, no preprocessing)
        text_description = ""
        if use_description:
            text_description = record.get("prob_desc_description") or ""
        
        # Load code if requested (raw, no preprocessing)
        text_code = ""
        if use_code:
            text_code = record.get("source_code") or ""

        tags = record.get("tags") or []
        tags = [str(t) for t in tags]

        yield ProblemExample(
            problem_id=name.split(".")[0],
            text_description=text_description,
            text_code=text_code,
            tags=tags
        )


def load_dataset_as_dataframe(
    directory: str,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load the dataset into a DataFrame with separate columns for description and code.
    
    Args:
        directory: Directory containing sample_*.json files
        features: List of features to load. Options: ["description"], ["code"], or ["description", "code"]
                  Default: ["description"]
    
    Returns:
        DataFrame with columns: problem_id, text_description, text_code, tags
        (text_description or text_code will be empty string if feature not requested)
    """
    dataframe = pd.DataFrame(columns=["problem_id", "text_description", "text_code", "tags"])
    for ex in iter_dataset(directory, features=features):
        dataframe.loc[len(dataframe)] = [
            ex.problem_id,
            ex.text_description,
            ex.text_code,
            ex.tags
        ]
        
    return dataframe
