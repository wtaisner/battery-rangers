# bmd-battery-rangers

![Battery rangers](static/battery_rangers.jpeg)

## Setup

### 1. Python environment
Using conda or mamba, run `mamba env create -f environment.yaml` to create an environment. Use `mamba activate battery` to activate it in the terminal.

### 2. [pre-commit](https://pre-commit.com)
Install pre-commit hooks: `pre-commit install`. From now on, it will run check automatically on `git commit`.

## Directory structure
```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── environment.yaml   <- The requirements file for reproducing the analysis environment.
│
├── pre-commit.yaml   <- Pre-commit hooks to run on commit. See https://pre-commit.com for details.
│
└── src   <- Source code for use in this project.
```

Roughly following [cookiecuter](https://cookiecutter-data-science.drivendata.org/) template.
