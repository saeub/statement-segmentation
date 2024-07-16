# Statement Segmentation in German Easy Language (StaGE) at KONVENS 2024

This repository contains the code for our submission to the [*Statement Segmentation in German Easy Language (StaGE)*](https://german-easy-to-read.github.io/statements/) shared task at [KONVENS 2024](https://konvens-2024.univie.ac.at/).

The two subtasks are:
1. **Determining the number of statements** in a sentence. The number of statements can be 0, 1, or more. Label Format: `2`.
2. If there is more than one statement in the sentence, **annotating the statement spans**. The statements can be non-contiguous and may overlap. Label Format: `[[1, 2], [4, 5, 6]]`.

The concept of *statement* is defined within the framework of valency grammar: A verb demands different arguments, its valency determines the number of arguments. All *optional* arguments are considered additional statements.

## Instructions

- Install dependencies: `pip install -r requirements.txt`
- Train all models (grid search): `./train_all.sh`  
  Models will be stored in the `models` directory.
- Predict on test data: `python src/cli.py predict --model [model-path] --testset data/test.csv --outfile prediction.csv`
- Evaluate: `python src/cli.py predict --model [model-path] --testset data/test.csv --predictions prediction.csv`

## Results

Full results from our grid search can be found in [`results.tsv`](results.tsv).

## License

The data included in this repository is copied from or based on the [shared task repository](https://github.com/german-easy-to-read/statements/tree/master/data) and published under [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

Our code is published under the [MIT License](LICENSE).
