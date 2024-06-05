# %%
import logging

logging.basicConfig(level=logging.INFO)
from collections.abc import Collection, Sequence

from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from data import Sentence, load_sentences
from model import Task1Model, Task2Model


class TokenTaggedDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.data = []

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MLMModel(Task1Model, Task2Model):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(
        self,
        train_sentences: Collection[Sentence],
        eval_sentences: Collection[Sentence],
    ):
        self.logger.info("Preprocessing data...")
        train_dataset = TokenTaggedDataset(train_sentences, self.tokenizer)
        eval_dataset = TokenTaggedDataset(eval_sentences, self.tokenizer)

        training_args = TrainingArguments(
            output_dir=f"./{self.model_name}-output",
            num_train_epochs=1,
            per_device_train_batch_size=16,
            logging_dir="./mlm-logs",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="epoch",
        )
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        self.logger.info("Training...")
        trainer.train()

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def predict_num_statements(self, sentences: Sequence[Sentence]) -> Sequence[int]:
        pred = []
        for sentence in sentences:
            encoded = self.tokenizer(
                sentence.clean_text, truncation=True, return_tensors="pt"
            )
            outputs = self.model(**encoded)
            pred.append(outputs.logits.argmax(-1).sum().item())
        return pred

    def predict_statement_spans(
        self, sentences: Sequence[Sentence]
    ) -> Sequence[list[int]]:
        raise NotImplementedError()


# %%
train_sentences = load_sentences("train")
test_sentences = load_sentences("test")

# %%
model = MLMModel("google-bert/bert-base-german-cased")
model.train(train_sentences, test_sentences)
model.save("mlm-2epoch-batch16")

# %%
model = MLMModel("./mlm-2epoch-batch16")

# %%
model.evaluate_num_statements(test_sentences)
