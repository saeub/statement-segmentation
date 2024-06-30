import csv
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Generator, Iterable
from math import sqrt
from statistics import harmonic_mean, mean
from typing import TextIO

from data import Sentence

Error = namedtuple("Error", ["sentence", "true", "pred"])


class Task1Model(ABC):
    @abstractmethod
    def predict_num_statements(self, sentences: Iterable[Sentence]) -> Iterable[int]:
        pass

    def evaluate_num_statements(
        self, sentences: Iterable[Sentence]
    ) -> dict[str, float]:
        true_num = [len(sentence.statement_spans) for sentence in sentences]
        pred_num = list(self.predict_num_statements(sentences))

        metrics = {}
        metrics["accuracy"] = mean(
            [pred == true for pred, true in zip(pred_num, true_num)]
        )
        metrics["mae"] = mean(
            [abs(pred - true) for pred, true in zip(pred_num, true_num)]
        )
        metrics["rmse"] = sqrt(
            mean([(pred - true) ** 2 for pred, true in zip(pred_num, true_num)])
        )

        return metrics

    def predict_to_csv(self, sentences: Iterable[Sentence], filename: str) -> None:
        pred_num_statements = list(self.predict_num_statements(sentences))
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["sent-id", "num_statements"])
            for sentence, pred in zip(sentences, pred_num_statements):
                writer.writerow(
                    [
                        sentence.id,
                        pred,
                    ]
                )

    def errors(self, sentences: Iterable[Sentence]) -> Generator[Error, None, None]:
        true_num = (len(sentence.statement_spans) for sentence in sentences)
        pred_num = self.predict_num_statements(sentences)

        for sentence, true, pred in zip(sentences, true_num, pred_num):
            if true != pred:
                yield Error(sentence=sentence, true=true, pred=pred)


class Task2Model(Task1Model):
    @abstractmethod
    def predict_statement_spans(
        self, sentence: Iterable[Sentence]
    ) -> Iterable[list[list[int]]]:
        pass

    def evaluate_statement_spans(
        self, sentences: Iterable[Sentence]
    ) -> dict[str, float]:
        true_spans = [sentence.statement_spans for sentence in sentences]
        pred_spans = list(self.predict_statement_spans(sentences))

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

    def predict_num_statements(
        self, sentences: Iterable[Sentence]
    ) -> Generator[int, None, None]:
        for spans in self.predict_statement_spans(sentences):
            yield len(spans)

    def predict_to_csv(self, sentences: Iterable[Sentence], file: TextIO) -> None:
        pred_statement_spans = list(self.predict_statement_spans(sentences))
        writer = csv.writer(file)
        writer.writerow(["sent-id", "num_statements", "statement_spans"])
        for sentence, pred in zip(sentences, pred_statement_spans):
            writer.writerow(
                [
                    sentence.id,
                    len(pred),
                    pred if len(pred) > 1 else None,
                ]
            )
