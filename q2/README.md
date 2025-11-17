# Question 2 – Token Classification (NER) with a Pre-trained Transformer
Objective:  To explore the functional role of attention in aligning and weighting linguistic
contexts across source and target sequences

This notebook compares three attention mechanisms using a Bi-GRU Encoder and GRU Decoder architecture on the Multi30k (English-German) dataset:
- Bahdanau (Additive) Attention
- Luong (General Multiplicative) Attention
- Scaled-Dot Product Attention

## What you get
- Identical data preprocessing (Multi30k) and word-level tokenization pipeline for all models.
- A consistent base architecture (Bi-GRU Encoder, GRU Decoder).
- Identical hyperparameters (embedding size, hidden size, dropout, optimizer, LR, batch size, epochs) for each attention mechanism.
- Clear Attention Heatmaps showing where the model "focuses."
- An Attention Entropy analysis to quantify the "sharpness" of the attention distribution and correlate it with BLEU scores.

## Quickstart

## QuickNote1
This project is worked on Google Colab (https://colab.research.google.com/). We recommend you to use Google Colab aswell. I've left environment.yml and requirements.txt in case you want to run it on your computer aswell.

## QuickNote2
The notebook uses the datasets library to automatically download the bentrevett/multi30k dataset. This dataset is public, so a HuggingFace token (HF_TOKEN) is not required.

```bash

# (not Recommended) Create a fresh virtual environment
python -m venv .venv && . .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Launch Jupyter and open the notebook
jupyter notebook q2.ipynb

```
## On Conda
```Bash

# (not Recommended) Create a fresh virtual environment on Conda
conda env create -f environment.yml
conda activate midterm-q2-env # (You can change the env name in environment.yml)


jupyter notebook q2.ipynb
```

## Files
- `q2.ipynb` — the main end-to-end notebook
- `requirements.txt` — `pip` dependencies
- `environment.yml` — `conda` environment file
- `README.md` — this file


## Reproducible Training Regime
RNN Architecture: Bi-GRU Encoder (1 layer), GRU Decoder (1 layer)
Embedding Dim: 256
Hidden Size: Encoder (256), Decoder (256)
Dropout: 0.2
Optimizer: Adam (lr=1e-3), Grad Clip: 1.0
Batch Size: 64
Epochs: 8 (Set as target_epoch in the notebook)
Metrics: BLEU, ROUGE-L, Perplexity (and Token-level Cross-Entropy Loss)

## Pipelines
There are 3 pipelines which are:
Pipeline 1: Bahdanau (Additive) Attention
Pipeline 2: Luong (Multiplicative) Attention
Pipeline 3: Scaled-Dot Product Attention

## Results Interpretation
The notebook prints the BLEU and ROUGE-L scores to the console during the training loop for each model (Bahdanau, Luong, Scaled-Dot).
The final section provides two main analysis tools:
Attention Heatmaps
Attention Entropy