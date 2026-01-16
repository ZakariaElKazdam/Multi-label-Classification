#from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from .data import DEFAULT_FOCUS_TAGS, load_dataset_as_dataframe, filter_to_focus_tags, load_json_file
from .model import TrainedModel, evaluate_model, train_model


def cmd_train(args: argparse.Namespace) -> None:
    df = load_dataset_as_dataframe(args.data_dir)
    if args.focus_tags_only:
        df = filter_to_focus_tags(df, DEFAULT_FOCUS_TAGS)

    random_state = args.random_state if hasattr(args, "random_state") else 42
    test_size = args.test_size if hasattr(args, "test_size") else 0.2

    model, report = train_model(df["text"].tolist(), df["tags"].tolist(), test_size=test_size, random_state=random_state)
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    model.save(args.model_path)

    print(json.dumps(report, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    df = load_dataset_as_dataframe(args.data_dir)
    if args.focus_tags_only:
        df = filter_to_focus_tags(df, DEFAULT_FOCUS_TAGS)

    model = TrainedModel.load(args.model_path)
    report = evaluate_model(model, df["text"].tolist(), df["tags"].tolist())
    print(json.dumps(report, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    model = TrainedModel.load(args.model_path)
    record = load_json_file(args.json_file)

    # Use only the problem description
    desc = record.get("prob_desc_description") or ""
    text = desc

    tags = model.predict([text], threshold=args.threshold)[0]

    out: Dict[str, Any] = {
        "predicted_tags": tags,
    }
    print(json.dumps(out, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codeforces problem tag classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train a new model")
    p_train.add_argument("--data-dir", required=True, help="Directory with sample_*.json files")
    p_train.add_argument(
        "--model-path",
        required=True,
        help="Path to save the trained model (e.g. models/model.joblib)",
    )
    p_train.add_argument(
        "--focus-tags-only",
        action="store_true",
        help="Keep only the 8 focus tags in targets",
    )
    p_train.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p_train.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (default: 0.2)",
    )
    p_train.set_defaults(func=cmd_train)

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate an existing model")
    p_eval.add_argument("--data-dir", required=True, help="Directory with sample_*.json files")
    p_eval.add_argument(
        "--model-path",
        required=True,
        help="Path to a trained model file",
    )
    p_eval.add_argument(
        "--focus-tags-only",
        action="store_true",
        help="Keep only the 8 focus tags in targets",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    # Predict
    p_pred = subparsers.add_parser("predict", help="Predict tags for a single JSON example")
    p_pred.add_argument("--model-path", required=True, help="Path to a trained model file")
    p_pred.add_argument("--json-file", required=True, help="Path to a sample_*.json file")
    p_pred.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for assigning tags",
    )
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
