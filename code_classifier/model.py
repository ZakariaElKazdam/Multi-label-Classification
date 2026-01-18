#from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import hstack
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer




@dataclass
class TrainedModel:
    vectorizers: dict  # TF-IDF vectorizers for each feature
    clf: OneVsRestClassifier  # classifier
    features: List[str]  # list of features used
    mlb: MultiLabelBinarizer  # convertit les tags en format binaire 
    classes_: List[str]  # liste des tags possibles

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "vectorizers": self.vectorizers,
                "clf": self.clf,
                "features": self.features,
                "mlb": self.mlb,
                "classes": self.classes_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TrainedModel":

        obj = joblib.load(path)
        return cls(
            vectorizers=obj["vectorizers"],
            clf=obj["clf"],
            features=obj.get("features", ["description"]), # the default features used to train the model
            mlb=obj["mlb"],
            classes_=list(obj["classes"]),
        )

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> List[List[str]]:
        """Predict labels for a batch of examples.
        
        Args:
            X: DataFrame with columns text_description and/or text_code
            threshold: Probability threshold for predictions
        """
        X_list = []
        if "description" in self.features:
            X_desc = self.vectorizers["description"].transform(X["text_description"].tolist())
            X_list.append(X_desc)
        if "code" in self.features:
            X_code = self.vectorizers["code"].transform(X["text_code"].tolist())
            X_list.append(X_code)
        X_combined = hstack(X_list) if len(X_list) > 1 else X_list[0]
        
        probas = self.clf.predict_proba(X_combined)
        labels_bin = (probas >= threshold).astype(int)
        return self.mlb.inverse_transform(labels_bin)


def build_pipeline(features: List[str] = None):
    """
    Create TF-IDF vectorizers and classifier.
    
    Args:
        features: List of features to use. Options: ["description"], ["code"], or ["description", "code"]
                  Default: ["description"]
    
    Returns:
        Tuple of (vectorizers_dict, classifier)
    """
    
    base_clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=250,
        random_state=42,
    )
    clf = OneVsRestClassifier(base_clf)
    
    # TF-IDF parameters
    tfidf_params = {
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.9,
        "sublinear_tf": True,
    }
    
    # Create vectorizers for each feature
    vectorizers = {}
    if "description" in features:
        vectorizers["description"] = TfidfVectorizer(**tfidf_params)
    if "code" in features:
        vectorizers["code"] = TfidfVectorizer(**tfidf_params)
    
    return vectorizers, clf


def train_model(
    df: pd.DataFrame,
    features: List[str],
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[TrainedModel, dict, List[str]]:
    """
    Train the model with train/val/test split and return validation metrics and test set IDs.
    
    Args:
        df: DataFrame with columns: text_description, text_code, tags, problem_id
        features: List of features to use (e.g., ["description"], ["code"], or ["description", "code"])
        train_size: Fraction of data for training (default: 0.7)
        val_size: Fraction of data for validation (default: 0.15)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (model, validation_report, test_ids)
        test_ids contains the problem IDs of the test set (preserved for later evaluation)
    """
    # Validate split sizes
    test_size = 1.0 - train_size - val_size
    if test_size <= 0:
        raise ValueError(f"train_size + val_size must be < 1.0, got {train_size} + {val_size} = {train_size + val_size}")

    # Extract tags and IDs
    tag_lists = df["tags"].tolist()

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tag_lists)
    
    # Create indices array to track splits
    indices = np.arange(len(df))
    
    # First split: train vs (val+test)
    temp_size = val_size + test_size
    df_train, df_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        df,
        Y,
        indices,
        test_size=temp_size,
        random_state=random_state,
        stratify=(Y.sum(axis=1) > 0).astype(int),
    )
    
    # Second split: val vs test
    val_ratio = val_size / temp_size
    df_val, df_test, y_val, y_test, idx_val, idx_test = train_test_split(
        df_temp,
        y_temp,
        idx_temp,
        test_size=1.0 - val_ratio,
        random_state=random_state,
        stratify=(y_temp.sum(axis=1) > 0).astype(int),
    )
    
    # Extract test IDs using indices
    ids_test = []

    ids_list = df["problem_id"].tolist()
    for idx in idx_test:
        ids_test.append(ids_list[idx])

    # building the pipeline
    vectorizers, clf = build_pipeline(features=features)
    
    # Vectorize each feature separately
    X_train_list = []
    X_val_list = []
    
    if "description" in features:
        X_train_desc = vectorizers["description"].fit_transform(df_train["text_description"].tolist())
        X_val_desc = vectorizers["description"].transform(df_val["text_description"].tolist())
        X_train_list.append(X_train_desc)
        X_val_list.append(X_val_desc)
    
    if "code" in features:
        X_train_code = vectorizers["code"].fit_transform(df_train["text_code"].tolist())
        X_val_code = vectorizers["code"].transform(df_val["text_code"].tolist())
        X_train_list.append(X_train_code)
        X_val_list.append(X_val_code)
    
    # Concatenate features
    X_train = hstack(X_train_list) # works also if there is only one feature
    X_val = hstack(X_val_list)
    
    # Train classifier
    clf.fit(X_train, y_train)
    
    # Evaluate only on validation set (test set is preserved for later)
    y_val_pred = clf.predict(X_val)
    val_report = classification_report(
        y_val,
        y_val_pred,
        target_names=mlb.classes_,
        output_dict=True,
        zero_division=0,
    )

    model = TrainedModel(
        vectorizers=vectorizers,
        clf=clf,
        features=features,
        mlb=mlb,
        classes_=list(mlb.classes_)
    )
    return model, val_report, ids_test


def evaluate_model(
    model: TrainedModel,
    texts: Sequence[str],
    ground_truth: Sequence[Sequence[str]],
) -> dict:
    """Evaluate an existing model on a new dataset."""
    Y_true = model.mlb.transform(ground_truth)
    Y_pred = model.pipeline.predict(list(texts))
    report = classification_report(
        Y_true,
        Y_pred,
        target_names=model.classes_,
        output_dict=True,
        zero_division=0,
    )
    return report
