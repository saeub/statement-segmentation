from math import sqrt
from statistics import harmonic_mean, mean
from typing import Iterable

from data import Sentence


def evaluate_num_statements(
    predictions: Iterable[int], sentences: Iterable[Sentence]
) -> dict[str, float]:
    true_num = [len(sentence.statement_spans) for sentence in sentences]
    pred_num = predictions

    metrics = {}
    metrics["accuracy"] = mean([pred == true for pred, true in zip(pred_num, true_num)])
    metrics["mae"] = mean([abs(pred - true) for pred, true in zip(pred_num, true_num)])
    metrics["rmse"] = sqrt(
        mean([(pred - true) ** 2 for pred, true in zip(pred_num, true_num)])
    )

    return metrics


def evaluate_statement_spans(
    predictions: Iterable[list[list[int]]], sentences: Iterable[Sentence]
) -> dict[str, float]:
    true_spans = [sentence.statement_spans for sentence in sentences]
    pred_spans = [
        pred if pred else [[i for i in range(len(sentence.tokens))]]
        for pred, sentence in zip(predictions, sentences)
    ]

    tp = fp = fn = 0
    for true, pred in zip(true_spans, pred_spans):
        for pred_span in pred:
            if pred_span in true:
                tp += 1
            else:
                fp += 1
        for true_span in true:
            if true_span not in pred:
                fn += 1

    metrics = {}
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["precision"] = tp / (tp + fp)
    metrics["recall"] = tp / (tp + fn)
    metrics["f1"] = harmonic_mean([metrics["precision"], metrics["recall"]])

    tp = fp = fn = 0
    for true, pred in zip(true_spans, pred_spans):
        if len(true) > 1:
            for pred_span in pred:
                if pred_span in true:
                    tp += 1
                else:
                    fp += 1
            for true_span in true:
                if true_span not in pred:
                    fn += 1

    metrics["tp_nosingle"] = tp
    metrics["fp_nosingle"] = fp
    metrics["fn_nosingle"] = fn
    metrics["precision_nosingle"] = tp / (tp + fp)
    metrics["recall_nosingle"] = tp / (tp + fn)
    metrics["f1_nosingle"] = harmonic_mean(
        [metrics["precision_nosingle"], metrics["recall_nosingle"]]
    )

    return metrics
