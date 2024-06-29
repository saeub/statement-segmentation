# %%
import json
import random

import requests

from data import load_sentences


def generate(
    model: str, prompt: str, num_predict: int | None = None, stop: str | None = None
) -> list[str]:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "options": {"num_predict": num_predict, "stop": [stop] if stop else None, "temperature": 0},
        },
    )
    if response.status_code != 200:
        raise Exception(response.text)
    dicts = [json.loads(line) for line in response.iter_lines()]
    return [d["response"] for d in dicts]


# %%
train_sentences = load_sentences("train")
test_sentences = load_sentences("test")
train_sentences_by_num_statements = {
    num_statements: [
        sentence
        for sentence in train_sentences
        if len(sentence.statement_spans) == num_statements
    ]
    for num_statements in range(6)
}

num_examples_by_num_statements = {
    1: 10,
    2: 5,
    3: 2,
    4: 1,
}
fewshot_examples = []
for num_statements, num_examples in num_examples_by_num_statements.items():
    fewshot_examples.extend(
        random.sample(train_sentences_by_num_statements[num_statements], num_examples)
    )
random.shuffle(fewshot_examples)


# %%
def build_prompt(test_sentence):
    prompt = "Wie viele Aussagen gibt es in folgenden Sätzen?\n\n"
    for sentence in fewshot_examples:
        num_statements = len(sentence.statement_spans)
        prompt += f'"{sentence.clean_text}" -> {num_statements} '
        if num_statements == 1:
            prompt += "Aussage"
        else:
            prompt += "Aussagen "
            if num_statements > 1:
                statements = ", ".join(
                    f'"{statement}"' for statement in sentence.statements
                )
                prompt += f"({statements})"
        prompt += "\n"
    prompt += f'"{test_sentence.clean_text}" -> '
    return prompt


for sentence in test_sentences:
    prompt = build_prompt(sentence)
    pred = generate("llama3:8b-text", prompt, num_predict=30, stop="\n")
    # pred = int(pred[0])
    pred = "".join(pred)
    print(len(sentence.statement_spans), pred, sep="\t")
    # print()
