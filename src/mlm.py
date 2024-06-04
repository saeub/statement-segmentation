# %%
from torch.utils.data import Dataset


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


# %%
from transformers import AutoModelForTokenClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-german-cased")
model = AutoModelForTokenClassification.from_pretrained(
    "google-bert/bert-base-german-cased"
)

# %%
from data import load_data

train_data = load_data("train")
train_dataset = TokenTaggedDataset(train_data, tokenizer)
test_data = load_data("test")
test_dataset = TokenTaggedDataset(test_data, tokenizer)

# %%
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

model = AutoModelForTokenClassification.from_pretrained(
    "google-bert/bert-base-german-cased"
)

training_args = TrainingArguments(
    output_dir="./mlm-output",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_dir="./mlm-logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_dataset,
    data_collator=DataCollatorForTokenClassification(tokenizer),
)

trainer.train()

# %%
correct = total = 0
for sentence in test_sentences:
    encoded = tokenizer(sentence.clean_text, truncation=True, return_tensors="pt")
    outputs = model(**encoded)
    pred = outputs.logits.argmax(-1).sum()
    print(sentence.clean_text)
    print(outputs.logits.argmax(-1))
    print(pred.item(), len(sentence.statement_spans), sentence.clean_text)
    correct += int(pred.item() == len(sentence.statement_spans))
    total += 1

correct / total
