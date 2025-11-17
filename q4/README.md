# Question 4 – Retrieval-Augmented Generation (RAG) and Knowledge-Grounded Text Synthesis

Objective: To design a retrieval-augmented pipeline integrating information retrieval
and neural text generation for factual, context-aware generation.

This notebook implements a full RAG system on the HotpotQA (distractor) dataset by combining multiple retrievers with a unified seq2seq generator. The goal is to analyze how retrieval quality affects factual correctness, hallucination behavior, and overall generation performance.

## What you get

Three retrieval backends:
- BM25 (sparse)
- TF-IDF (sparse)
- Sentence-BERT / all-MiniLM-L6-v2 (dense)

A unified generator:
- FLAN-T5-small (with option to replace by T5-small or GPT-Neo)

Retrieval evaluation using:
- Precision@k, Recall@k
- Generation evaluation using:
- BLEU, ROUGE-L, BERTScore (P/R/F1)

Faithfulness analysis:
- Distinguishes whether the generated answer is supported by retrieved evidence.
- A complete comparison table and a qualitative case study.

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
jupyter notebook q4.ipynb

```
## On Conda
```Bash

# (not Recommended) Create a fresh virtual environment on Conda
conda env create -f environment.yml
conda activate midterm-q4-env # (You can change the env name in environment.yml)


jupyter notebook q4.ipynb
```

## Files
- `q4.ipynb` — the main end-to-end notebook
- `requirements.txt` — `pip` dependencies
- `environment.yml` — `conda` environment file
- `README.md` — this file


## Reproducible Training Regime

- Retriever:
  - BM25
  - TF-IDF
  - Dense retriever (all-MiniLM-L6-v2 Sentence-BERT)

- Generator (Sequence-to-Sequence Model):
  - FLAN-T5-small (default)

- Supports replacing with T5-small or GPT-Neo if desired

- Top-k retrieved contexts: 5

- Max input length: 1024 tokens (question + retrieved context)

- Max output length: 64 tokens

- Evaluation metrics:
  - Precision@k, Recall@k (retrieval-only)
  - BLEU, ROUGE-L, BERTScore (RAG end-to-end)

- Faithfulness Detection:
  - A prediction is marked faithful if the gold answer appears in retrieved evidence.

- Reproducibility:
  - Random seeds for Python, NumPy, and PyTorch are fixed.

## Pipelines
## Pipeline 1 — Sparse RAG

- Retrieve top-k documents with:
- BM25, or TF-IDF
- Concatenate question + context
- Generate answer using FLAN-T5
- Compute BLEU, ROUGE-L, BERTScore
- Mark faithful vs hallucinated

## Pipeline 2 — Dense RAG

- Retrieve with Sentence-BERT all-MiniLM-L6-v2
- Same generation pipeline as above
- Compare against sparse retrievers

## Results Interpretation
The notebook prints:

- The Precision@k and Recall@k
- The BLEU, ROUGE-L, and BERTScore
- Questions, answers, retrieved evidence, model predictions, and faithful vs. hallucinated labels
