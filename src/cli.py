import argparse
import ast
import csv
import json
import logging
import random
import sys
from pathlib import Path

logger = logging.getLogger("cli")
logging.basicConfig(level=logging.INFO)


def train(args: argparse.Namespace):
    from data import load_sentences
    from mlm import MLMModel

    train_sentences = load_sentences(args.trainset)
    train_size = len(train_sentences)
    dev_sentences = load_sentences(args.devset) if args.devset else None

    augment_size = 0
    if args.augment_trainset:
        augment_sentences = (
            load_sentences(args.augment_trainset) if args.augment_trainset else []
        )
        if args.augment_samples is not None:
            random.seed(args.augment_sample_seed)
            augment_sentences = random.sample(augment_sentences, args.augment_samples)
        augment_size = len(augment_sentences)
        train_sentences.extend(augment_sentences)

    logger.info(
        f"Training model on {train_size + augment_size} samples ({train_size} train + {augment_size} augmented)"
    )
    model = MLMModel(args.base_model)
    model.train(
        train_sentences,
        dev_sentences,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )
    logger.info(f"Saving model to {args.save_to}")
    model.save(args.save_to)


def predict(args: argparse.Namespace):
    from data import load_sentences
    from mlm import MLMModel

    test_sentences = load_sentences(args.testset)
    model = MLMModel(args.model)
    model.predict_to_csv(test_sentences, args.outfile)


def evaluate(args: argparse.Namespace):
    from data import load_sentences
    from evaluation import evaluate_num_statements, evaluate_statement_spans

    test_sentences = load_sentences(args.testset)
    reader = csv.DictReader(args.predictions)
    predictions = [
        (
            int(row["num_statements"]),
            (
                ast.literal_eval(row["statement_spans"])
                if row["statement_spans"]
                else None
            ),
        )
        for row in reader
    ]
    assert len(predictions) == len(
        test_sentences
    ), f"Expected {len(test_sentences)} predictions, got {len(predictions)}"

    metrics_task1 = evaluate_num_statements(
        [num for num, _ in predictions], test_sentences
    )
    metrics_task2 = evaluate_statement_spans(
        [spans for _, spans in predictions], test_sentences
    )
    print(json.dumps({"task1": metrics_task1, "task2": metrics_task2}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--base-model", required=True)
    train_parser.add_argument("--trainset", type=Path, required=True)
    train_parser.add_argument("--devset", type=Path)
    train_parser.add_argument("--augment-trainset", type=Path)
    train_parser.add_argument("--augment-samples", type=int)
    train_parser.add_argument("--augment-sample-seed", type=int, default=42)
    train_parser.add_argument("--num-epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--save-to", type=Path, required=True)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--testset", type=Path, required=True)
    predict_parser.add_argument(
        "--outfile", type=argparse.FileType("w", encoding="utf-8"), default=sys.stdout
    )

    predict_parser = subparsers.add_parser("evaluate")
    predict_parser.add_argument(
        "--predictions", type=argparse.FileType(encoding="utf-8"), default=sys.stdin
    )
    predict_parser.add_argument("--testset", type=Path, required=True)

    args = parser.parse_args()
    logger.info(" ".join(sys.argv))

    if args.command == "train":
        return train(args)
    if args.command == "predict":
        return predict(args)
    if args.command == "evaluate":
        return evaluate(args)


if __name__ == "__main__":
    main()
