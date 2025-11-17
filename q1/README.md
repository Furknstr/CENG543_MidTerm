# Question 1 – Comparative Analysis of Recurrent Architectures and Embedding Paradigms

Objective: To investigate how the inductive biases of recurrent architectures and the
representational capacity of distinct embedding schemes jointly influence sequence modeling performance.

This notebook compares the impact of different embedding paradigms (**Static-GloVe** vs. **Contextual-DistilBERT**) 
on the performance of two recurrent architectures (**BiLSTM** and **BiGRU**) for sentiment classification on the IMDb dataset.

## What you get
- Identical preprocessing and tokenization pipeline
- Same hyperparameters (hidden size, layers, dropout, optimizer, LR, batch size, epochs)
- Clear **t-SNE and PCA** graphs.

## Quickstart

## QuickNote1
This project is worked on Google Colab (https://colab.research.google.com/). We recommend you to use Google Colab aswell. I've left environment.yml and requirements.txt in case you want to run it on your computer aswell.

## QuickNote2 
Also you have to set your HF_TOKEN for the getting the datasets from the HuggingFace. See how to get the HF_TOKEN from the HuggingFace.

```bash
# (not Recommended) Create a fresh virtual environment
python -m venv .venv && . .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Launch Jupyter and open the notebook
jupyter notebook q1.ipynb

```

## On Conda
```bash
# (not Recommended) Create a fresh virtual environment on Conda
conda env create -f environment.yml
conda activate midterm-q1-env


jupyter notebook q1.ipynb

```

> **Note**: The notebook will download the IMDb dataset via `datasets` on first run.

## Files
- `q1.ipynb` — the main end-to-end notebook
- `requirements.txt` — `pip` dependencies
- `environment.yml` — `conda` environment file
- `README.md` — this file

## Reproducible Training Regime
- RNN hidden size: 128, **2 layers**, **bidirectional=True**
- Dropout: 0.3
- Optimizer: Adam (`lr=1e-3`), weight decay 0
- Batch size: 64 (train), 128 (val/test)
- Epochs: 3 (It is 3 because it's training time was time consuming.)
- Metrics: accuracy, F1 (macro)

## Pipelines
## Pipeline 1: Static (GloVe)
Tokenizer: Regex (TOKEN_RE)
Embedding: glove-wiki-gigaword-300 (300d)
Vocab Size: 30,000
Sequence Length: 300

## Pipeline 2: Contextual
Tokenizer: AutoTokenizer (distilbert-base-uncased)
Embedding: AutoModel (distilbert-base-uncased) (768d, frozen)
Sequence Length: 256

## Results Interpretation
The notebook prints a **comparison table** showing each model's test accuracy/F1
and total training time and convergence time. It also plots **t-SNE** and **PCA** graphs.