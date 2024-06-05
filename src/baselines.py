# %%
from collections import defaultdict
from collections.abc import Collection, Sequence
import random

from data import Sentence
from model import Task1Model


class RandomBaseline(Task1Model):
    def __init__(self):
        self.count_distribution = defaultdict(int)

    def train(
        self,
        train_sentences: Collection[Sentence],
    ):
        for sentence in train_sentences:
            self.count_distribution[len(sentence.statement_spans)] += 1

    def predict_num_statements(self, sentences: Sequence[Sentence]) -> list[int]:
        return [
            random.choices(
                list(self.count_distribution.keys()),
                list(self.count_distribution.values()),
            )[0]
            for _ in sentences
        ]


class MajorityBaseline(Task1Model):
    def __init__(self):
        self.majority_count = None

    def train(
        self,
        train_sentences: Collection[Sentence],
    ):
        num_statements = [len(sentence.statement_spans) for sentence in train_sentences]
        self.majority_count = max(set(num_statements), key=num_statements.count)

    def predict_num_statements(self, sentences: Sequence[Sentence]) -> list[int]:
        return [self.majority_count] * len(sentences)


class VerbBaseline(Task1Model):
    def predict_num_statements(self, sentences: Sequence[Sentence]) -> list[int]:
        pred = []
        for sentence in sentences:
            pos = [token.pos_ for tokens in sentence.spacy_tokens for token in tokens]
            pred.append(pos.count("VERB"))
        return pred

# %%
from data import load_sentences

train_sentences = load_sentences("train")
test_sentences = load_sentences("test")

# %%
baselines = []

random_baseline = RandomBaseline()
random_baseline.train(train_sentences)
baselines.append(random_baseline)

verb_baseline = VerbBaseline()
baselines.append(verb_baseline)

majority_baseline = MajorityBaseline()
majority_baseline.train(train_sentences)
baselines.append(majority_baseline)

# %%
for baseline in baselines:
    metrics = baseline.evaluate_num_statements(test_sentences)
    print(f"{baseline.__class__.__name__}: {metrics}")
