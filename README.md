# Legal Judgment Summarization with Llama-2 (LoRA Fine-Tuning + RAG)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](#)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange.svg)](#)
[![TRL](https://img.shields.io/badge/TRL-SFTTrainer-purple.svg)](#)

Fine-tune **Llama-2** using **LoRA (PEFT)** for **legal judgment summarization**, run inference on raw judgment text files, and optionally use a simple **RAG pipeline** (TF-IDF + FAISS) to retrieve relevant legal judgments before summarization.

---

## ✨ What’s inside

- **Data preprocessing**
  - Reads raw judgments (`.txt`) and author-wise summaries
  - Produces training-ready JSONL files (e.g., `full_summaries_A1.jsonl`, `full_summaries_A2.jsonl`)
- **LoRA fine-tuning (SFT)**
  - Uses `transformers` + `trl` (`SFTTrainer`) + `peft` (LoRA)
  - Designed for low VRAM using `bitsandbytes`
- **Inference notebook**
  - Loads base Llama-2 + LoRA weights and generates summaries
- **RAG pipeline (optional)**
  - TF-IDF embeddings + FAISS similarity search
  - Retrieves top-k relevant judgments and summarizes them

---

## 📁 Repository structure

```text
legal-llama-summarization/
├─ notebooks/
│  ├─ 01_Data_preprocessing.ipynb
│  ├─ 02_Finetune_LLama.ipynb
│  ├─ 03_Llama_inference.ipynb
│  └─ 04_RAG_Pipeline.ipynb
├─ data/
│  ├─ raw/IN-Ext/judgement/
│  ├─ raw/IN-Ext/summary/full/
│  ├─ raw/IN-Ext/summary/segment-wise/
│  └─ processed/processed-IN-Ext/
├─ models/
│  ├─ fine_tuned_lora_adapter/
│  └─ fine_tuned_lora_model/
├─ results/runs/
├─ requirements.txt
├─ .env.example
├─ .gitignore
└─ README.md
```
## ✅ Requirements
- **Python 3.9+
- **NVIDIA GPU recommended for training (inference may work on CPU, but will be slow)
Main libraries:
- **transformers, datasets, trl, peft, bitsandbytes
- **For RAG: faiss-cpu, scikit-learn
## 🔐 Hugging Face Access (Llama-2)
1. Have access approved on Hugging Face for the Llama-2 model repo
2. Use a Hugging Face token with access
Create a .env file (see .env.example) and set:
```bash
HF_TOKEN=your_huggingface_token_here
```
## 🚀 Setup
```bash
# 1) Clone
git clone <your-repo-url>
cd legal-llama-summarization

# 2) Create environment
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 3) Install deps
pip install -r requirements.txt

# 4) Create env file
cp .env.example .env
# edit .env and set HF_TOKEN
```
## 🧪 Notebooks workflow (recommended order)
### 1. Data preprocessing
Open:
- **notebooks/01_Data_preprocessing.ipynb
What it does:
- **Reads judgments from data/raw/IN-Ext/judgement/
- **Reads summaries from data/raw/IN-Ext/summary/full/ and segment-wise/
- **Writes JSONL to data/processed/processed-IN-Ext/
  Output example:
- **full_summaries_A1.jsonl
- **full_summaries_A2.jsonl
### 2. Fine-tune Llama-2 with LoRA (SFTTrainer)
```text
Open:
- **notebooks/02_Finetune_LLama.ipynb
