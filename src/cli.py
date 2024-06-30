import argparse
import logging
import random
import sys
from pathlib import Path

logger = logging.getLogger("cli")


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
    model.save(args.save_to)


def predict(args: argparse.Namespace):
    from data import load_sentences
    from mlm import MLMModel

    test_sentences = load_sentences(args.testset)
    model = MLMModel(args.model)
    model.predict_to_csv(test_sentences, args.outfile)


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

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.command == "train":
        return train(args)
    if args.command == "predict":
        return predict(args)


if __name__ == "__main__":
    main()
