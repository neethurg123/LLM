# -*- coding: utf-8 -*-
"""rag based_querying


!pip install -r requirements.txt

import torch
torch.cuda.empty_cache()

print(torch.__version__)
print(torch.cuda.is_available())

"""**Upload the document **"""

from google.colab import files
uploaded = files.upload()
document=list(uploaded.keys())[0]

with open(document, 'r', encoding='utf-8') as f:
    document= f.read()

"""**Splitting the Text into chunks using Langchain**"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = splitter.split_text(document)
print(f"Total chunks: {len(chunks)}")

"""**Sentence Transformers for Embedding and Storing**"""

from sentence_transformers import SentenceTransformer
embedder= SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embed_chunks=embedder.encode(chunks)

import faiss
import numpy as np
dimension=embed_chunks.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(np.array(embed_chunks))

print(f"FAISS index built with {index.ntotal} vectors")

"""**Use Hugging Face token for Transformers**"""

from huggingface_hub import login
import getpass

token = getpass.getpass("Enter your Hugging Face token (won’t show): ")
login(token)

"""**Defining the Model (Llama)**"""

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

"""**Defining the Model (Mistral)**"""

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id1 = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer1 = AutoTokenizer.from_pretrained(model_id1, use_auth_token=True)
model1 = AutoModelForCausalLM.from_pretrained(
    model_id1,
    device_map="auto",
    torch_dtype=torch.float16  # or float32 if needed
)

"""**Querying**"""

def retrieve_context(query, k=3):
    query_embed = embedder.encode([query])
    _, top_k = index.search(np.array(query_embed), k)
    return "\n".join([chunks[i] for i in top_k[0]])

"""**Querying the model for Generating Responses**"""

def answer_with_llama(query):
    context = retrieve_context(query)

# Construct prompt as a plain conversation
    prompt1 = f"""### Instruction:
Use only the context below to answer the question. Respond in one sentence. No explanation.

### Context:
{context}

### Question:
{query}

### Answer:"""

    inputs1 = tokenizer(prompt1, return_tensors="pt").to(model.device)
    input_ids1 = inputs1["input_ids"]
    attention_mask1 = inputs1["attention_mask"]

    output_ids1 = model.generate(
        input_ids=input_ids1,
        attention_mask=attention_mask1,
        max_new_tokens=80,
        do_sample=False,
    )

    new_tokens1= output_ids1[0][input_ids1.shape[1]:]
    outputs1 = tokenizer.decode(new_tokens1, skip_special_tokens=True).replace("\n", " ").strip()

    return clean_output(outputs1)

def answer_with_mistral(query):
    context = retrieve_context(query)  # or use separate context retriever

    prompt2 = f"""<s>[INST] Use only the context below to answer the question directly.Do not repeat or rephrase the context. No explanation. Just the final answer.

    Context:{context}

    Question:{query} [/INST]"""

    inputs2 = tokenizer1(prompt2, return_tensors="pt").to(model1.device)
    input_ids2 = inputs2["input_ids"]
    attention_mask2 = inputs2["attention_mask"]

    output_ids2 = model1.generate(
        input_ids=input_ids2,
        attention_mask=attention_mask2,
        max_new_tokens=80,
        do_sample=False
    )
    new_tokens2 = output_ids2[0][input_ids2.shape[1]:]

    outputs2 = tokenizer1.decode(new_tokens2, skip_special_tokens=True)
    return outputs2.strip()

"""**Cleaning Data for LLama**"""

def clean_output(text):
    for tag in ["### Instruction", "### Context", "### Question", "### Answer"]:
        text = text.split(tag)[0]
    return text.strip()

"""**Response Using both**"""

question = "How can individuals reduce their risk of developing Type 2 Diabetes?"
answer1 = answer_with_llama(question)
answer2 = answer_with_mistral(question)

print(f"Answer using LLaMA:\n{answer1}")

print(f"\nAnswer using Mistral:\n{answer2}")

"""**Saving answers to csv file for Evaluation**"""

import pandas as pd
from tqdm import tqdm
df=pd.read_csv('test_data.csv')
df["llama_answer"] = ""
df["bleu_llama"] = 0.0
df["rouge_llama"] = None  # Initialize with None or an appropriate empty value
df["bert_llama"] = 0.0
df["mistral_answer"] = ""
df["bleu_mistral"] = 0.0
df["rouge_mistral"] = None  # Initialize with None or an appropriate empty value
df["bert_mistral"] = 0.0

for i in tqdm(range(len(df))):
  question=df.loc[i,'question']
  llama_ans = answer_with_llama(question)
  mistral_ans = answer_with_mistral(question)

  df.at[i, "llama_answer"] = llama_ans
  df.at[i, "mistral_answer"] = mistral_ans
df.head()

"""**Model Evaluation**"""

import nltk
nltk.download('punkt')
nltk.download('punkt_tab') # Added to download punkt_tab
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import word_tokenize
from bert_score import score as bert_score
import evaluate
rouge_metric = evaluate.load("rouge")


def bleu(pred,ref):
    ref_tokens = word_tokenize(ref.lower())
    pred_tokens = word_tokenize(pred.lower())
    smoothing = SmoothingFunction()
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing.method1)

def rouge(pred, ref):
    result = rouge_metric.compute(predictions=[pred], references=[ref], use_stemmer=True)
    return result["rougeL"]  # Returns the F1 score directly (as float)

def bert(pred,ref):
 P, R, f1 = bert_score([pred], [ref], lang="en", verbose=False)
 return f1.item() # Return the f1 score

for i in tqdm(range(len(df))):
  pred1=df.loc[i,'llama_answer']
  pred2=df.loc[i,'mistral_answer']
  ref=df.loc[i,'Reference']
  df.at[i, "bleu_llama"] = bleu(pred1,ref)
  df.at[i, "rouge_llama"] = rouge(pred1,ref)
  df.at[i, "bert_llama"] = bert(pred1,ref)
  df.at[i, "bleu_mistral"] = bleu(pred2,ref)
  df.at[i, "rouge_mistral"] = rouge(pred2,ref)
  df.at[i, "bert_mistral"]= bert(pred2,ref)

df

"""**Mean Evaluation Metric **"""

metrics = ["bleu", "rouge", "bert"]

for metric in metrics:
    llama_avg = df[f"{metric}_llama"].mean()
    mistral_avg = df[f"{metric}_mistral"].mean()

    print(f"{metric.upper()} Mean Scores:")
    print(f"  LLaMA:   {llama_avg:.4f}")
    print(f"  Mistral: {mistral_avg:.4f}")
    print("-" * 30)
