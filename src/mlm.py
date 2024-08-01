# %%
import logging
import random

from collections.abc import Collection, Generator, Iterable

from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import Sentence
from model import Task2Model


class TokenTaggedDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.data = []

        # TODO: Refactor preprocessing into separate function

        for sentence in sentences:
            # Get character-level locations of statement tags
            tag_locations = set()
            for span in sentence.statement_spans:
                # Get the span's highest token in the dependency tree
                spacy_tokens = [
                    (i, token) for i in span for token in sentence.spacy_tokens[i]
                ]
                tag_index, _ = min(
                    spacy_tokens, key=lambda token: len(list(token[1].ancestors))
                )
                # Convert token-level index to character-level location
                tag_location = sum(
                    len(token) for token in sentence.clean_tokens[:tag_index]
                )
                tag_location += sum(
                    token != "" for token in sentence.clean_tokens[:tag_index]
                )  # whitespace
                tag_locations.add(tag_location)
                # TODO: handle overlapping spans with same root

            # Tokenize text
            text = sentence.clean_text
            encoded = tokenizer(text, return_offsets_mapping=True, truncation=True)

            # Create binary labels aligned with tokens
            labels = [0] * len(encoded["input_ids"])
            for i, (start, end) in enumerate(encoded["offset_mapping"]):
                if not tag_locations:
                    break
                if start == 0 and end == 0:
                    continue
                next_tag_location = min(tag_locations)
                if start >= next_tag_location:
                    labels[i] = 1
                    tag_locations.remove(next_tag_location)
            encoded["labels"] = labels
            del encoded["offset_mapping"]

            self.data.append(encoded)
        random.seed(42)
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MLMModel(Task2Model):
    def __init__(self, model_name: str, tokenizer_name: str | None = None):
        self.model_name = model_name
        set_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(
        self,
        train_sentences: Collection[Sentence],
        dev_sentences: Collection[Sentence] | None,
        *,
        num_epochs: int,
        batch_size: int,
    ):
        self.logger.info("Preprocessing data...")
        train_dataset = TokenTaggedDataset(train_sentences, self.tokenizer)
        dev_dataset = TokenTaggedDataset(dev_sentences, self.tokenizer) if dev_sentences else None

        set_seed(42)
        # TODO: Support GPU
        training_args = TrainingArguments(
            output_dir=f"model-output/{self.model_name}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            logging_steps=50,
            eval_strategy="steps" if dev_dataset else "no",
            eval_steps=50,
            save_strategy="epoch",
        )
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
        )

        self.logger.info("Training...")
        trainer.train()

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def predict_num_statements(
        self, sentences: Iterable[Sentence]
    ) -> Generator[int, None, None]:
        # Using the unprocessed span tags (i.e., not ignoring punctuation etc.)
        # appears to work better for predicting the number of statements
        for sentence in sentences:
            encoded = self.tokenizer(
                sentence.clean_text, truncation=True, return_tensors="pt"
            )
            outputs = self.model(**encoded)
            yield max(1, outputs.logits.argmax(-1).sum().item())

    def predict_statement_spans(
        self, sentences: Iterable[Sentence]
    ) -> Generator[list[list[int]]]:
        for sentence in sentences:
            encoded = self.tokenizer(
                sentence.clean_text,
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            offset_mapping = encoded.pop("offset_mapping")[0]

            outputs = self.model(**encoded)
            labels = outputs.logits.argmax(-1)[0]

            if sum(labels) == 0:
                yield [[i for i in range(len(sentence.tokens))]]
                continue

            if sum(labels) == 1:
                yield [[i for i in range(len(sentence.tokens))]]
                continue

            # TODO: Refactor postprocessing into separate function

            # Get character-level locations of statement tags
            tag_locations = set()
            for label, (start, end) in zip(labels, offset_mapping):
                if start == 0 and end == 0:
                    continue
                if label == 1:
                    tag_locations.add(start)

            # Map character-level locations to token-level indices
            tag_indices = []
            location = 0
            for i, token in enumerate(sentence.clean_tokens):
                if not tag_locations:
                    break
                next_tag_location = min(tag_locations)
                if location >= next_tag_location:
                    tag_indices.append(i)
                    tag_locations.remove(next_tag_location)
                location += len(token) + (token != "")  # token + whitespace

            spans = [[i] for i in tag_indices]

            def ignore(token):
                return (
                    token.dep_ == "punct" or token.dep_ == "cc" or token.pos_ == "DET"
                )

            # Find root token of each span
            tag_roots = []
            for span in spans:
                spacy_tokens = [
                    token
                    for i in span
                    for token in sentence.spacy_tokens[i]
                    if not ignore(token)
                ]
                if not spacy_tokens:
                    # Probably punctuation or determiner was tagged, ignore this span
                    continue
                root = min(spacy_tokens, key=lambda token: len(list(token.ancestors)))
                tag_roots.append(root)

            def get_descendants(token, stop=()):
                # Stop to avoid overlapping spans
                if token in stop or ignore(token):
                    return []
                descendants = [token]
                for child in token.children:
                    descendants.extend(get_descendants(child, stop=stop))
                return descendants

            # Handle deepest spans first
            tag_roots.sort(key=lambda token: len(list(token.ancestors)), reverse=True)

            # Expand spans to include all descendants in dependency tree
            spacy_token_to_index = {
                token: i
                for i, tokens in enumerate(sentence.spacy_tokens)
                for token in tokens
            }
            expanded_spans = []
            used_descendants = set()
            for root in tag_roots:
                descendants = get_descendants(root, stop=used_descendants)
                used_descendants.update(descendants)
                # Convert tokens to indices, deduplicate, and sort
                span = sorted({spacy_token_to_index[token] for token in descendants})
                expanded_spans.append(span)

            # There should be no overlap
            for i, span in enumerate(expanded_spans):
                for j, other_span in enumerate(expanded_spans):
                    if i != j:
                        assert not set(span) & set(other_span)

            yield expanded_spans
