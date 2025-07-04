## Retrieval-Augmented Question Answering using TinyLlama and Mistral

This project implements a complete **RAG (Retrieval-Augmented Generation)** system to answer user queries from a document using 
open-source large language models (LLMs): **TinyLlama** and **Mistral**. It also includes an automatic evaluation framework using 
**BLEU**, **ROUGE-L**, and **BERTScore** to compare the quality of generated answers.

## Features
✅ Document ingestion and chunking

✅ Embedding using Sentence Transformers (`all-MiniLM-L6-v2`)

✅ Vector-based retrieval using FAISS

✅ Answer generation using:
  - [TinyLlama-1.1B-Chat-v1.0]
  - [Mistral-7B-Instruct-v0.1]

✅ Hugging Face Token:Some models (e.g., Mistral) hosted on Hugging Face require authentication.
  How to Get Your Token:
  - Go to https://huggingface.co/settings/tokens
  - Click "New Token"
  - Set scope to "read" access

✅ Automatic evaluation using:
  - BLEU (n-gram overlap)
  - ROUGE-L (sequence overlap)
  - BERTScore (semantic similarity)

## How It Works
1. Split the input document (plain_text.txt) into overlapping chunks.
2. Embed each chunk using a Sentence Transformer.
3. Index the embeddings using FAISS for fast retrieval.
4. For each question:
   - Retrieve top-k relevant chunks.
   - Generate an answer using both TinyLlama and Mistral.
   - Evaluate both answers against the reference answer. (CSV file used for evaluation: test_data.csv)

## Evaluation Metrics

1. BLEU:	Measures n-gram overlap with reference
2. ROUGE-L:	Longest common subsequence with reference
3. BERTScore:	Semantic similarity using BERT embeddings

License: Please refer to the licenses of TinyLlama and Mistral models if you plan to use them commercially.
