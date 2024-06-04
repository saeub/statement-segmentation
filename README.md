# Statement Detection for StaGE at KONVENS 2024s

This repository contains the code for the Statement Segmentation in German Easy Language ([StaGE](https://Statement Segmentation in German Easy Language)) shared task at KONVENS 2024 ([KONVENS](https://konvens-2024.univie.ac.at/)).

The two subtasks are:
1. **Determining the number of statements** in a sentence. The number of statements can be 0, 1, or more. Label Format: `2`.
2. **Annotating the statements** in the sentences. The statements can be non-contiguous and may overlap. Label Format: ``[[1, 2], [4, 5, 6]]``.

The concept of "statement" is not motivated semantically, but coming from the grammar of valence.
A verb demands different arguments, its valence determines the number of arguments. All *optional* arguments are considered additional statements.

## Data

The annotation guidelines can be found [here](https://german-easy-to-read.github.io/statements/annotations/). 

### Informal Insights

- Pronoun tokens are not part of the statement (in the annotation).



