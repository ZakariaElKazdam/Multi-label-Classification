import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from .data import load_dataset_as_dataframe, load_json_file
from .preprocessing import (
    DEFAULT_FOCUS_TAGS,
    filter_to_focus_tags,
    remove_empty_tags,
    preprocess_description,
    preprocess_code,
)

from .model import TrainedModel, train_model


def cmd_train(args: argparse.Namespace) -> None:
    # Parse features from comma-separated string
    if args.features:
        features = [f.strip() for f in args.features.split(",")]
    else:
        features = ["description"]  # default to description only
    
    # Validate features - we only support description and code for now
    valid_features = {"description", "code"}
    for feat in features:
        if feat not in valid_features:
            raise ValueError(f"Unknown feature: {feat}. Available: {valid_features}")
    
    # Warn if using both features - might not always improve performance
    if len(features) > 1:
        print(f"Using features: {features}")
    
    # Load the dataset
    df = load_dataset_as_dataframe(args.data_dir, features=features)
    print(f"Loaded {len(df)} examples")
    
    # Clean the text data
    if "description" in features:
        df["text_description"] = df["text_description"].apply(preprocess_description)
    if "code" in features:
        df["text_code"] = df["text_code"].apply(preprocess_code)
    
    # Optionally filter to only the 8 focus tags
    # This reduces the dataset but makes the problem more focused
    if args.focus_tags_only:
        df = filter_to_focus_tags(df, DEFAULT_FOCUS_TAGS)
        df = remove_empty_tags(df)
        print(f"After filtering to focus tags: {len(df)} examples")

    # Get embedding types for description and code separately
    embedding_desc = getattr(args, "embedding_desc", "tfidf")
    embedding_code = getattr(args, "embedding_code", "tfidf")
    word2vec_params = {
        "vector_size": getattr(args, "word2vec_vector_size", 100),
        "min_count": getattr(args, "word2vec_min_count", 2),
        "epochs": getattr(args, "word2vec_epochs", 10),
    }
    split_params = {
        "train_size": args.train_size,
        "val_size": args.val_size,
        "random_state": args.random_state,
    }
    
    # Warning if CodeBERT is used for description (it's optimized for code)
    if "description" in features and embedding_desc == "codebert":
        print("  WARNING: CodeBERT is optimized for code, not natural language descriptions.")
        print("   Consider using TF-IDF or Word2Vec for description instead.")
    
    # Print what we're using
    if "description" in features:
        print(f"Description embedding: {embedding_desc}")
    if "code" in features:
        print(f"Code embedding: {embedding_code}")
    
    # Train the model with train/val/test split
    # We keep test set separate for final evaluation after comparing models
    model, val_report, test_ids = train_model(
        df=df,
        features=features,
        embedding_desc=embedding_desc,
        embedding_code=embedding_code,
        word2vec_params=word2vec_params,
        split_params=split_params,
    )
    
    # Save the trained model
    model_dir = os.path.dirname(args.model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    model.save(args.model_path)

    # Build the report with all the config and results
    test_size = 1.0 - split_params["train_size"] - split_params["val_size"]
    
    # Embedding parameters - can be different for description and code
    embedding_params = {}
    if "description" in features:
        if embedding_desc == "tfidf":
            embedding_params["description"] = {
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_df": 0.9,
                "sublinear_tf": True,
            }
        elif embedding_desc == "word2vec":
            embedding_params["description"] = word2vec_params.copy()
            embedding_params["description"]["window"] = 5
        else:  # codebert
            embedding_params["description"] = {
                "model": "microsoft/codebert-base",
                "max_length": 512,
            }
    
    if "code" in features:
        if embedding_code == "tfidf":
            embedding_params["code"] = {
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_df": 0.9,
                "sublinear_tf": True,
            }
        elif embedding_code == "word2vec":
            embedding_params["code"] = word2vec_params.copy()
            embedding_params["code"]["window"] = 5
        else:  # codebert
            embedding_params["code"] = {
                "model": "microsoft/codebert-base",
                "max_length": 512,
            }
    
    report_data = {
        "features": features,
        "embedding_types": model.embedding_type,
        "embedding_params": embedding_params,
        "classifier": "LogisticRegression",
        "classifier_params": {
            "solver": "liblinear",  # works well for this dataset size
            "class_weight": "balanced",  # helps with imbalanced classes
            "max_iter": 250,
            "strategy": "OneVsRest",
        },
        "train_size": split_params["train_size"],
        "val_size": split_params["val_size"],
        "test_size": test_size,
        "random_state": split_params["random_state"],
        "focus_tags_only": args.focus_tags_only,
        "validation_metrics": val_report,
        "test_set_ids": test_ids,  # save test IDs to avoid using them during development
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save report in reports/ folder
    os.makedirs("reports", exist_ok=True)
    
    # Generate model name from features and embedding types
    features_str = "_".join(sorted(features))  # sorted for consistency
    embedding_str = "_".join([f"{k}_{v}" for k, v in sorted(model.embedding_type.items())])
    model_name = f"{embedding_str}_{features_str}_{report_data["classifier"]}"
    
    # Format: month_day_hour_minute
    timestamp_str = datetime.now().strftime("%m_%d_%H_%M")
    report_filename = f"reports/report_{model_name}_{timestamp_str}.json"
    
    with open(report_filename, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"Model saved to: {args.model_path}")
    print(f"Report saved to: {report_filename}")
    print(f"Test set contains {len(test_ids)} examples")
    print("\n=== Validation Metrics ===")
    print(json.dumps(val_report, indent=2))




def cmd_predict(args: argparse.Namespace) -> None:
    # Load the trained model
    model = TrainedModel.load(args.model_path)
    
    # Check that user provided either single file or directory, not both
    if args.json_file and args.data_dir:
        raise ValueError("Please specify either --json-file OR --data-dir, not both")
    if not args.json_file and not args.data_dir:
        raise ValueError("Need to specify either --json-file or --data-dir")
    
    if args.json_file:
        # Single file prediction
        record = load_json_file(args.json_file)
        desc = record.get("prob_desc_description") or ""
        code = record.get("source_code") or ""
        
        # Apply same preprocessing as during training
        desc_processed = preprocess_description(desc)
        code_processed = preprocess_code(code)
        
        # Build DataFrame with the features the model expects
        data = {}
        if "description" in model.features:
            data["text_description"] = [desc_processed]
        if "code" in model.features:
            data["text_code"] = [code_processed]
        X = pd.DataFrame(data)
        
        tags = model.predict(X, threshold=args.threshold)[0]
        
        out = { "predicted_tags": tags }
        print(json.dumps(out, indent=2))
    
    else:
        # Batch prediction on a directory
        df = load_dataset_as_dataframe(args.data_dir, features=model.features)
        
        # Preprocess the data
        if "description" in model.features:
            df["text_description"] = df["text_description"].apply(preprocess_description)
        if "code" in model.features:
            df["text_code"] = df["text_code"].apply(preprocess_code)
        
        # Get predictions for all examples
        all_tags = model.predict(df, threshold=args.threshold)
        
        # Build output with problem IDs
        predictions = []
        for i, tags in enumerate(all_tags):
            predictions.append({
                "problem_id": df.iloc[i]["problem_id"],
                "predicted_tags": tags,
            })
        
        out = {
            "predictions": predictions,
        }
        print(json.dumps(out, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codeforces problem tag classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    p_train = subparsers.add_parser("train", help="Train a new model")
    p_train.add_argument("--data-dir", required=True, help="Directory with json files")
    p_train.add_argument("--model-path", required=True, help="Path to save the trained model")
    p_train.add_argument(
        "--focus-tags-only",
        action="store_true",
        help="Filter to only the 8 focus tags (reduces dataset size)",
    )
    p_train.add_argument("--random-state", type=int, default=42, help="Random seed")
    p_train.add_argument("--train-size", type=float, default=0.7, help="Training set fraction")
    p_train.add_argument("--val-size", type=float, default=0.15, help="Validation set fraction")
    p_train.add_argument(
        "--features",
        type=str,
        default="description",
        help="Features to use: 'description', 'code', or 'description,code'",
    )
    p_train.add_argument(
        "--embedding-desc",
        type=str,
        default="tfidf",
        choices=["tfidf", "word2vec", "codebert"],
        help="Embedding method for description (default: tfidf). Warning: CodeBERT is optimized for code, not descriptions.",
    )
    p_train.add_argument(
        "--embedding-code",
        type=str,
        default="tfidf",
        choices=["tfidf", "word2vec", "codebert"],
        help="Embedding method for code (default: tfidf). CodeBERT gives best results for code.",
    )
    p_train.add_argument("--word2vec-vector-size", type=int, default=100, help="Word2Vec dimension")
    p_train.add_argument("--word2vec-min-count", type=int, default=2, help="Word2Vec min word count")
    p_train.add_argument("--word2vec-epochs", type=int, default=10, help="Word2Vec training epochs")
    p_train.set_defaults(func=cmd_train)

    # Predict command
    p_pred = subparsers.add_parser("predict", help="Predict tags")
    p_pred.add_argument("--model-path", required=True, help="Path to trained model")
    p_pred.add_argument("--json-file", help="Single file to predict")
    p_pred.add_argument("--data-dir", help="Directory with files to predict")
    p_pred.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
