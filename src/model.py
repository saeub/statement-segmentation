from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Generator, Iterable
from math import sqrt
from statistics import mean

from data import Sentence


Error = namedtuple("Error", ["sentence", "true", "pred"])


class Task1Model(ABC):
    @abstractmethod
    def predict_num_statements(self, sentences: Iterable[Sentence]) -> Iterable[int]:
        pass

    def evaluate_num_statements(
        self, sentences: Iterable[Sentence]
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

    def errors(self, sentences: Iterable[Sentence]) -> Generator[Error, None, None]:
        true_num_statements = (len(sentence.statement_spans) for sentence in sentences)
        pred_num_statements = self.predict_num_statements(sentences)

        for sentence, true, pred in zip(
            sentences, true_num_statements, pred_num_statements
        ):
            if true != pred:
                yield Error(sentence=sentence, true=true, pred=pred)


class Task2Model(ABC):
    @abstractmethod
    def predict_statement_spans(
        self, sentence: Iterable[Sentence]
    ) -> Iterable[list[int]]:
        pass

    def evaluate_statement_spans(
        self, sentences: Iterable[Sentence]
    ) -> dict[str, float]:
        raise NotImplementedError()
