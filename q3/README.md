# Question 3 – Transition from Recurrent Encoder–Decoder Models to Transformer Architectures
Objective: To analyze the conceptual and empirical transition from recurrence-based
sequence modeling to self-attention-driven architectures.


Notebook evaluates two architectures on the Multi30k EN→DE dataset under controlled conditions:

- GRU-based Seq2Seq model with additive attention.
- Transformer encoder–decoder with sinusoidal positional encoding.

Both pipelines share identical preprocessing, batching strategy, tokenization setup, and evaluation metrics, enabling direct comparison.

## What you get

- Shared hyperparameters wherever architecturally possible (embedding size, hidden size, dropout, optimizer, LR, batch size, epochs).
- Full training loops with loss tracking and checkpointing.
- BLEU and ROUGE-L evaluation.
- Runtime, GPU memory, and parameter count reporting for performance comparison.
- Final summary table contrasting Seq2Seq vs Transformer.

## Quickstart

## QuickNote1
This project is worked on Google Colab (https://colab.research.google.com/). We recommend you to use Google Colab aswell. I've left environment.yml and requirements.txt in case you want to run it on your computer aswell.

## QuickNote2
Also you have to set your HF_TOKEN for the getting the datasets from the HuggingFace. See how to get the HF_TOKEN from the HuggingFace.


```bash

# (not Recommended) Create a fresh virtual environment
python -m venv .venv && . .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Launch Jupyter and open the notebook
jupyter notebook q3.ipynb

```
## On Conda
```Bash

# (not Recommended) Create a fresh virtual environment on Conda
conda env create -f environment.yml
conda activate midterm-q3-env # (You can change the env name in environment.yml)


jupyter notebook q3.ipynb
```

## Files
- `q3.ipynb` — the main end-to-end notebook
- `requirements.txt` — `pip` dependencies
- `environment.yml` — `conda` environment file
- `README.md` — this file


## Reproducible Training Regime
# Seq2Seq model

- Encoder: Bi-GRU, 1 layer, 512 hidden
- Decoder: GRU, 1 layer, 512 hidden
- Attention: Additive (Bahdanau-style)
- Embedding dim: 300
- Dropout: applied to encoder and decoder outputs
- Optimizer: Adam (lr=1e-3)
- Max grad norm: clipped
- Epochs: 8
- Batch size: 64
- Greedy decode

# Transformer model
- Encoder–decoder stack with multi-head attention
- Sinusoidal positional encodings
- Same embedding dimension
- Same optimizer, LR, batch size, and epoch count
- Dropout inside attention and feed-forward layers
- Greedy decode

## Pipelines
There are 2 pipelines which are:
- Pipeline 1: Seq2Seq + Additive Attention
- Pipeline 2: Transformer (multi-head self-attention)

## Results Interpretation
The notebook prints the BLEU and ROUGE-L scores to the console during the training loop for each model (Transformer and Seq2Seq).
## Evaluation Metrics
- Training loss
- Validation loss
- BLEU
- ROUGE-L
- Training time (seconds)
- Max GPU memory consumption (MB)
- Parameter count