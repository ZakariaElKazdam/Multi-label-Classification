#from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, jaccard_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import hstack
from sklearn.preprocessing import MultiLabelBinarizer

from gensim.models import Word2Vec as W2V


@dataclass
class TrainedModel:
    vectorizers: dict  # Embedding vectorizers for each feature
    clf: OneVsRestClassifier  # classifier
    features: List[str]  # list of features used
    embedding_type: dict  # type of embedding per feature: {"description": "tfidf", "code": "codebert"}
    mlb: MultiLabelBinarizer  # convertit les tags en format binaire 
    classes_: List[str]  # liste des tags 

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "vectorizers": self.vectorizers,
                "clf": self.clf,
                "features": self.features,
                "embedding_type": self.embedding_type,
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
            features=obj.get("features", ["description"]),
            embedding_type=obj.get("embedding_type", {'description': 'tfidf', 'code': 'tfidf'}),
            mlb=obj["mlb"],
            classes_=obj["classes"],
        )

    def embed_feature(self, feature: str, texts: List[str]) -> np.ndarray:
        """Embed a list of texts for a feature using the appropriate vectorizer."""
        embedding_type = self.embedding_type.get(feature, "tfidf")
        if embedding_type == "codebert":
            return get_codebert_embeddings(texts)
        else:
            # vectorizer (TF-IDF or Word2Vec) depending on user's previous choice : flexibility hehe :)
            return self.vectorizers[feature].transform(texts)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> List[List[str]]:
        """Predict labels for a batch of examples.
        
        Args:
            X: DataFrame with columns text_description and/or text_code
            threshold: Probability threshold for predictions
        """
        # Vectorize each feature
        X_list = []
        if "description" in self.features:
            X_desc = self.embed_feature("description", X["text_description"].tolist())
            X_list.append(X_desc)
        
        if "code" in self.features:
            X_code = self.embed_feature("code", X["text_code"].tolist())
            X_list.append(X_code)

        # Combine embeddings: TF-IDF gives sparse matrices, Word2Vec/CodeBERT give dense
        if len(X_list) == 1:
            X_combined = X_list[0]
        else:
            # Check types to use the most efficient concatenation
            all_sparse = all(hasattr(x, "toarray") for x in X_list)
            all_dense = all(not hasattr(x, "toarray") for x in X_list)
            
            if all_sparse:
                X_combined = hstack(X_list)  # hstack is a method of scipy.sparse to concatenate sparse matrices
            elif all_dense:
                X_combined = np.hstack(X_list)  # Direct concatenation, no conversion needed
            else:
                # Mixed: convert sparse to dense then concatenate
                X_combined = np.hstack([x.toarray() if hasattr(x, "toarray") else x for x in X_list])
        
        probas = self.clf.predict_proba(X_combined)
        labels_bin = (probas >= threshold).astype(int)
        return self.mlb.inverse_transform(labels_bin)


class Word2VecVectorizer:
    """Simple Word2Vec vectorizer - trains on data."""
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2, epochs: int = 10):
        try:
            self.Word2Vec = W2V
        except ImportError:
            raise ImportError("gensim package is required here, try installing it with: pip install gensim")
            # normally this won't happen because it's in the requirements.txt file ;)
        
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Train Word2Vec on texts and transform them."""
        # Tokenize texts (simple split, can be improved !!!!!)
        sentences = [text.split() for text in texts]
        # sentences is a list of lists of words, documentary of w2v name the "sentences" parameter
        
        # Train Word2Vec model sur sentences
        self.model = self.Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1,
            sg=1,  # skip-gram
            epochs=self.epochs,
        )
        
        return self.transform(texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to Word2Vec vectors (average of word vectors)."""
        if self.model is None:
            raise ValueError("Word2Vec model must be fitted first")
        
        vectors = []
        for text in texts:
            words = text.split()
            if words:
                word_vecs = []
                for word in words:
                    if word in self.model.wv: 
                        #check if the word exist in the vocabulary already, we skip it if it doesn't 
                        word_vecs.append(self.model.wv[word])
                if word_vecs:
                    vectors.append(np.mean(word_vecs, axis=0))
                else:
                    vectors.append(np.zeros(self.vector_size))
            else:
                vectors.append(np.zeros(self.vector_size))
        return np.array(vectors)

def get_codebert_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Get CodeBERT embeddings for a list of texts. Returns numpy array."""
    from transformers import AutoTokenizer, AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    model.eval()  # inference mode, modele deja préentrenné
    
    embeddings = []
    import torch
    with torch.no_grad():  # on desactive le calcul des gradients (pas besoin lors de l'inference)
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512, # tronque si text est plus long, perte d'info ? idk, have to search more
                padding=True
            )
            output = model(**encoded) # debaquetage du dictionnaire encoded en parametres du modele
            batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy() # on prend le token [CLS]
            embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings) # on concatene les embeddings par lignes


def build_components(features: List[str] = None, embedding_desc: str = "tfidf", embedding_code: str = "tfidf", word2vec_params: dict = None):
    """
    Create embedding vectorizers and classifier components.
    Each feature can have its own embedding type.
    Returns vectorizers dict, classifier, and embedding_types dict.
    """
    if word2vec_params is None:
        word2vec_params = {"vector_size": 100, "min_count": 2, "epochs": 10}
    
    # Logistic regression with class balancing for imbalanced data
    base_clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=250,
        random_state=42,
    )
    clf = OneVsRestClassifier(base_clf)
    
    vectorizers = {}
    embedding_types = {}
    
    # TF-IDF parameters
    tfidf_params = {
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.9,
        "sublinear_tf": True,
    }
    
    # Handle description
    if "description" in features:
        embedding_types["description"] = embedding_desc
        if embedding_desc == "tfidf":
            vectorizers["description"] = TfidfVectorizer(**tfidf_params)
        elif embedding_desc == "word2vec":
            vectorizers["description"] = Word2VecVectorizer(**word2vec_params)
        elif embedding_desc == "codebert":
            # Codebert, we'll use the function directly
            vectorizers["description"] = None
        else:
            raise ValueError(f"Unknown embedding_desc: {embedding_desc}. Supported: 'tfidf', 'word2vec', 'codebert'")
    
    # Handle code
    if "code" in features:
        embedding_types["code"] = embedding_code
        if embedding_code == "tfidf":
            vectorizers["code"] = TfidfVectorizer(**tfidf_params)
        elif embedding_code == "word2vec":
            vectorizers["code"] = Word2VecVectorizer(**word2vec_params)
        elif embedding_code == "codebert":
            vectorizers["code"] = None
        else:
            raise ValueError(f"Unknown embedding_code: {embedding_code}. Supported: 'tfidf', 'word2vec', 'codebert'")
    
    return vectorizers, clf, embedding_types


def train_model(
    df: pd.DataFrame,
    features: List[str],
    embedding_desc: str = "tfidf",
    embedding_code: str = "tfidf",
    word2vec_params: dict = None,
    split_params: dict = None,
) -> Tuple[TrainedModel, dict, List[str]]:
    """
    Train the model with train/val/test split.
    Returns validation metrics and test set IDs.
    """
    if split_params is None:
        split_params = {"train_size": 0.7, "val_size": 0.15, "random_state": 42}
    
    train_size = split_params["train_size"]
    val_size = split_params["val_size"]
    random_state = split_params["random_state"]
    
    # Check that split sizes make sense
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

    # vectorizers and classifier
    vectorizers, clf, embedding_types = build_components(
        features=features,
        embedding_desc=embedding_desc,
        embedding_code=embedding_code,
        word2vec_params=word2vec_params,
    )
    
    # Vectorize each feature separately
    X_train_list = []
    X_val_list = []
    
    if "description" in features:
        if embedding_types["description"] == "codebert":
            # Use CodeBERT function directly
            X_train_desc = get_codebert_embeddings(df_train["text_description"].tolist())
            X_val_desc = get_codebert_embeddings(df_val["text_description"].tolist())
        else:
            # Use vectorizer (TF-IDF or Word2Vec)
            X_train_desc = vectorizers["description"].fit_transform(df_train["text_description"].tolist())
            X_val_desc = vectorizers["description"].transform(df_val["text_description"].tolist())
        X_train_list.append(X_train_desc)
        X_val_list.append(X_val_desc)
    
    if "code" in features:
        if embedding_types["code"] == "codebert":
            X_train_code = get_codebert_embeddings(df_train["text_code"].tolist())
            X_val_code = get_codebert_embeddings(df_val["text_code"].tolist())
        else:
            X_train_code = vectorizers["code"].fit_transform(df_train["text_code"].tolist())
            X_val_code = vectorizers["code"].transform(df_val["text_code"].tolist())
        X_train_list.append(X_train_code)
        X_val_list.append(X_val_code)
    
    # Concatenate features: TF-IDF gives sparse matrices, Word2Vec/CodeBERT give dense
    if len(X_train_list) == 1:
        X_train = X_train_list[0]
        X_val = X_val_list[0]
    else:
        # Check types to use the most efficient concatenation
        all_sparse = all(hasattr(x, "toarray") for x in X_train_list)
        all_dense = all(not hasattr(x, "toarray") for x in X_train_list)
        
        if all_sparse:
            X_train = hstack(X_train_list)
            X_val = hstack(X_val_list)
        elif all_dense:
            X_train = np.hstack(X_train_list)
            X_val = np.hstack(X_val_list)
        else:
            # Mixed: convert sparse to dense then concatenate
            X_train = np.hstack([x.toarray() if hasattr(x, "toarray") else x for x in X_train_list])
            X_val = np.hstack([x.toarray() if hasattr(x, "toarray") else x for x in X_val_list])
    
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
    
    # Add multi-label specific metrics
    # Jaccard similarity: average over samples (each problem gets a score)
    jaccard_avg = jaccard_score(y_val, y_val_pred, average='samples', zero_division=0)
    # Hamming loss: fraction of labels that are incorrectly predicted
    hamming = hamming_loss(y_val, y_val_pred)
    
    # Add these to the report
    val_report['jaccard_similarity'] = jaccard_avg
    val_report['hamming_loss'] = hamming

    model = TrainedModel(
        vectorizers=vectorizers,
        clf=clf,
        features=features,
        embedding_type=embedding_types,
        mlb=mlb,
        classes_=list(mlb.classes_)
    )
    return model, val_report, ids_test


