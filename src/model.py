from abc import ABC, abstractmethod
from math import sqrt
from statistics import mean
from typing import Sequence

from data import Sentence


class Task1Model(ABC):
    @abstractmethod
    def predict_num_statements(self, sentences: Sequence[Sentence]) -> Sequence[int]:
        pass

    def evaluate_num_statements(
        self, sentences: Sequence[Sentence]
    ) -> dict[str, float]:
        true_num_statements = [len(sentence.statement_spans) for sentence in sentences]
        pred_num_statements = self.predict_num_statements(sentences)

        metrics = {}
        metrics["accuracy"] = mean(
            [
                pred == true
                for pred, true in zip(pred_num_statements, true_num_statements)
            ]
        )
        metrics["mae"] = mean(
            [
                abs(pred - true)
                for pred, true in zip(pred_num_statements, true_num_statements)
            ]
        )
        metrics["rmse"] = sqrt(
            mean(
                [
                    (pred - true) ** 2
                    for pred, true in zip(pred_num_statements, true_num_statements)
                ]
            )
        )

        return metrics


class Task2Model(ABC):
    @abstractmethod
    def predict_statement_spans(
        self, sentence: Sequence[Sentence]
    ) -> Sequence[list[int]]:
        pass

    def evaluate_statement_spans(
        self, sentences: Sequence[Sentence]
    ) -> dict[str, float]:
        raise NotImplementedError()
