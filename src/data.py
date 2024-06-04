# %%
import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import spacy
import spacy.tokens

nlp = spacy.load("de_dep_news_trf")


# %%
@dataclass
class Sentence:
    id: int
    topic: str
    text: str
    tokens: list[str]
    clean_tokens: list[str]
    statement_spans: list[list[int]]

    @classmethod
    def from_row(cls, row: pd.Series) -> "Sentence":
        tokens = []
        clean_tokens = []
        for i, token in enumerate(row["phrase_tokenized"].split(" ")):
            match = re.match(r"^(\d+):=([^ ]*)$", token)
            assert match is not None, token
            assert int(match.group(1)) == i
            token = match.group(2)
            tokens.append(token)

            clean_token = token.replace("\\newline", " ")
            clean_token = re.sub(r"\s+", " ", clean_token)
            clean_tokens.append(clean_token.strip())

        spans = row["statement_spans"]
        num_statements = row["num_statements"]
        if pd.notna(spans) and num_statements > 1:
            spans = ast.literal_eval(spans)
        elif num_statements > 0:
            spans = [[i for i in range(len(tokens))]]
        else:
            spans = []

        return cls(
            id=row["sent-id"],
            topic=row["topic"],
            text=row["phrase"],
            tokens=tokens,
            clean_tokens=clean_tokens,
            statement_spans=spans,
        )

    @property
    def clean_text(self) -> str:
        """Whitespace-cleaned text."""
        text = ""
        for i, token in enumerate(self.clean_tokens):
            if token == "":
                continue
            text += token
            if i < len(self.clean_tokens) - 1:
                text += " "
        return text

    @property
    def statements(self) -> list[str]:
        """Statement spans as (uncleaned) strings."""
        statements = []
        for span in self.statement_spans:
            tokens = [self.tokens[i] for i in span]
            statements.append(" ".join(tokens))
        return statements

    @property
    def spacy_tokens(self) -> list[list[spacy.tokens.Token]]:
        """spaCy tokens parsed based on cleaned text but aligned with original (uncleaned) tokens."""
        if hasattr(self, "_spacy_tokens"):
            return self._spacy_tokens

        doc = nlp(self.clean_text)

        # Map spacy tokens to self tokens
        spacy_tokens = []
        self_token_index = 0
        partial_token = ""
        for spacy_token in doc:
            # Skip whitespace self tokens
            self_token = self.clean_tokens[self_token_index]
            while self_token == "":
                spacy_tokens.append([])
                self_token_index += 1
                self_token = self.clean_tokens[self_token_index]

            # Read spacy tokens until the text matches the current self token
            if partial_token == "":
                spacy_tokens.append([])
            else:
                partial_token += doc[spacy_token.i - 1].whitespace_
            partial_token += spacy_token.text
            spacy_tokens[-1].append(spacy_token)
            if len(partial_token) >= len(self_token):
                assert partial_token == self_token, (partial_token, self_token)
                self_token_index += 1
                partial_token = ""

        self._spacy_tokens = spacy_tokens
        return self._spacy_tokens


# %%
DATA_PATH = Path(__name__).absolute().parent.parent / "data"


def load_data(name: str) -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH / f"{name}.csv")
    data["sentence"] = data.apply(Sentence.from_row, axis=1)
    return data
