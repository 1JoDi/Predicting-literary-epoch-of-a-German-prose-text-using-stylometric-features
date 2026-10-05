# Stylometric Epoch Classification of German Prose

Final project for the course *Advanced Python for NLP* (HHU Düsseldorf, 2026).

**Research question:** How well can a classifier predict the literary epoch
of a German prose text (Goethezeit, Realismus, Moderne) from four
stylometric features: median sentence length, standardised type-token
ratio, function word ratio and character density?

The pipeline downloads German narrative prose from
[Project Gutenberg](https://www.gutenberg.org), labels every text with an
epoch based on its author's death year, extracts the four features with
[spaCy](https://spacy.io) and compares four classifiers (majority baseline,
a NumPy nearest-centroid classifier, logistic regression, random forest).
Training and test data are split by author, so the classifiers cannot
recognise an author instead of an epoch.

The written report is in [`report/report.pdf`](report/report.pdf).

---

## Repository structure

```
.
├── README.md               this file
├── main.py                 entry point: runs the whole pipeline
├── config.yaml             all parameters (paths, epochs, sample size, ...)
├── requirements.txt        Python packages
├── src/                    one module per pipeline step
│   ├── config.py           reads config.yaml
│   ├── catalog.py          downloads + filters the Gutenberg catalog
│   ├── epochs.py           author death year -> epoch
│   ├── download.py         downloads the texts, builds the corpus
│   ├── cleaning.py         removes boilerplate, extracts text samples
│   ├── features.py         the four stylometric features
│   ├── dataset.py          runs spaCy, writes the feature table
│   ├── describe.py         descriptive statistics per epoch
│   ├── centroid.py         nearest-centroid classifier (NumPy)
│   ├── classify.py         training and evaluation
│   └── plots.py            figures
├── tests/                  unit tests (pytest)
├── data/                   created by the pipeline (not in git)
│   ├── raw/                downloaded catalog and texts
│   └── processed/          candidates, corpus metadata, feature table
├── results/                created by the pipeline (not in git)
└── report/                 the report (PDF)
```

## Pipeline

```
 catalog ──► download ──► features ──► analyse
    │            │            │            │
    ▼            ▼            ▼            ▼
candidates   corpus_      features.csv   results/
   .csv      metadata.csv
```

| Step | Module(s) | Input | Output |
|------|-----------|-------|--------|
| `catalog` | `catalog.py`, `epochs.py` | Gutenberg catalog (downloaded) | `data/processed/candidates.csv` |
| `download` | `download.py`, `cleaning.py` | candidates | `data/raw/texts/*.txt`, `data/processed/corpus_metadata.csv` |
| `features` | `dataset.py`, `features.py`, `cleaning.py` | corpus + raw texts | `data/processed/features.csv` |
| `analyse` | `describe.py`, `classify.py`, `centroid.py`, `plots.py` | feature table | everything in `results/` |

Every step writes its output to disk and the next step reads it, so the
steps can be run (and re-run) separately. Downloads and features that
already exist are not computed again.

## Installation

The project was developed and tested with **Python 3.14.0**. In the project folder:

**Windows (PowerShell or cmd)**

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download de_core_news_lg
```

**Linux / macOS**

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download de_core_news_lg
```

The spaCy model is about 550 MB. For a quick test, the small model
`de_core_news_sm` also works: download it and change `spacy_model` in
`config.yaml`.

## Usage

Run the complete pipeline:

```
python main.py
```

Run a single step (`catalog`, `download`, `features` or `analyse`):

```
python main.py --step analyse
```

Use a different configuration file:

```
python main.py --config my_config.yaml
```

Run the unit tests:

```
python -m pytest
```

**Runtime.** The repository contains no data: the first run of
`python main.py` downloads the catalog and all texts and computes every
result from scratch. This takes about 45–60 minutes on a laptop. The
`download` step waits 2 seconds between requests to be polite to the
Gutenberg server (about 20–30 minutes for a few hundred texts), the
`features` step processes 10,000 words per text (about 15–30 minutes),
and `analyse` takes under a minute.

Every step saves its output in `data/` or `results/`, so later runs are
fast: already downloaded texts and already computed features are reused,
and `python main.py --step analyse` alone re-creates all results in
under a minute once `data/processed/features.csv` exists.

**Starting over.** To rebuild everything from scratch, delete the folders
`data/` and `results/`. If you change the sampling or feature settings,
delete `data/processed/features.csv`, otherwise already processed texts
are skipped.

**Reproducibility.** All random steps (corpus selection, data split,
models) use the fixed seed from `config.yaml`, so the same catalog
always produces the same corpus and the same results. Project Gutenberg
keeps adding books to its catalog, however, and a different catalog
leads to a different random selection of texts. The results in the
report were produced with the catalog downloaded on 30 September 2026;
a run with a newer catalog will give a different corpus and slightly
different numbers.

## Expected output

The `analyse` step prints the corpus distribution, a
Kruskal-Wallis test per feature and the test-set scores of all models,
and writes the following files to `results/`:

| File | Content |
|------|---------|
| `epoch_distribution.csv/.png` | texts and authors per epoch |
| `feature_summary.csv`, `feature_boxplots.png` | mean / sd / median of every feature per epoch |
| `kruskal_tests.csv` | does each feature differ between epochs? |
| `cv_results.csv`, `cv_comparison.png` | cross-validation on the training set, author-disjoint vs. random folds |
| `test_results.csv` | accuracy and macro-F1 of every model on the held-out authors |
| `classification_reports.txt` | precision / recall / F1 per epoch and model |
| `confusion_matrix.png` | confusion matrix of the best model |
| `test_predictions.csv`, `misclassified.csv` | predictions per text (for the qualitative analysis) |
| `logreg_coefficients.csv`, `permutation_importance.csv`, `feature_ablation.csv`, `feature_importance.png` | which features matter |
| `metrics.json` | summary of the most important numbers |
| `best_model.pkl` | the trained best model (pickle) |

## Results

Corpus: 268 texts by 192 authors (Goethezeit 55, Realismus 93,
Moderne 120). Test set: 54 texts by 38 authors that do not appear in
the training data.

| Model | Accuracy | Macro-F1 |
|-------|---------:|---------:|
| Majority baseline | 0.444 | 0.205 |
| Nearest centroid (NumPy) | 0.444 | **0.438** |
| Logistic regression | 0.426 | 0.375 |
| Random forest | 0.463 | 0.420 |

* The classifiers clearly beat the baseline on macro-F1, but not on
  accuracy: the epoch leaves a measurable but weak trace in the four
  features.
* Median sentence length (17 → 14 → 12 words from Goethezeit to Moderne)
  and the function word ratio carry most of the signal; the type-token
  ratio adds little and the character density nothing.
* Many errors come from the death-year labels, from authors with an
  untypical style and from non-narrative texts that passed the genre
  filter. See the report for the full analysis.

## Configuration

All settings are in `config.yaml` and documented there, e.g.

* `epochs.bins` – the epoch boundaries (by author death year),
* `corpus.exclude_keywords` – the keywords used to remove drama, poetry
  and non-fiction,
* `corpus.max_texts_per_author` / `max_texts_per_epoch` – corpus balance,
* `sampling.sample_words` – size of the analysed passage per text,
* `features.spacy_model` – the spaCy model.

## Data and licences

The texts come from Project Gutenberg and are in the public domain in the
USA. They are downloaded by the pipeline and not redistributed in this
repository, and neither are any derived files: everything in `data/`
and `results/` is created by running the code. The Gutenberg catalog is
used under the
[Project Gutenberg terms](https://www.gutenberg.org/policy/terms_of_use.html).

## Author

Joel Dick (single-author project), B.A. Computerlinguistik,
Heinrich-Heine-Universität Düsseldorf. Final project for the course
*Advanced Python for NLP* (Rainer Osswald, Yulia Zinova), 2026.
