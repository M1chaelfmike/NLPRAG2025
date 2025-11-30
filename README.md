# COMP5423 Natural Language Processing - RAG System Project

This project implements a comprehensive **Retrieval-Augmented Generation (RAG)** system for Open-Domain Question Answering, developed for the COMP5423 course at The Hong Kong Polytechnic University.

The system integrates multiple retrieval strategies and advanced generation workflows (Multi-turn & Agentic) to answer complex questions from the [HotpotQA](https://hotpotqa.github.io/) dataset.

## ✨ Key Features

### 1. Retrieval Module (Six Methods)
We implement a diverse set of retrieval algorithms to handle different query types:
- **BM25**: Traditional sparse retrieval based on term frequency.
- **Static Embedding**: Dense retrieval using GloVe embeddings.
- **Dense Retrieval**: Semantic search using `all-mpnet-base-v2`.
- **Instruction Dense**: Advanced dense retrieval using `intfloat/e5-base-v2` with instruction tuning.
- **Hybrid Retrieval**: Combines BM25 and Static/Dense scores for robust performance.
- **Multi-vector Retrieval**: Fine-grained interaction using a simplified ColBERT-style scoring mechanism.

### 2. Generation Module (Three Modes)
- **Basic RAG**: Standard retrieve-then-generate pipeline.
- **Feature A: Multi-turn Conversation**: Handles context-dependent queries with coreference resolution and conversation history management.
- **Feature B: Agentic Workflow**: Implements a ReAct-style agent that performs query rewriting, step-by-step reasoning, and self-verification to answer complex multi-hop questions.

### 3. User Interface
- **Terminal UI**: A rich, interactive command-line interface for testing all features.
- **GUI**: A graphical user interface for easier interaction.

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd NLPRAG2025-master1
   ```

2. **Install Dependencies**
   Ensure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: The system will automatically download necessary models (Qwen2.5, BERT, etc.) to the `hf_cache/` directory upon first run.*

## 🚀 Usage

### 1. Interactive Terminal UI (Recommended)
The best way to explore the system is through the interactive terminal interface.
```bash
python UI/terminal_ui.py
```
Follow the on-screen prompts to:
- Select a retrieval method (e.g., BM25, Dense, Agentic).
- Choose a generation mode.
- Enter your questions.

### 2. Graphical User Interface (GUI)
For a visual experience:
```bash
python UI/gui_ui.py
```

### 3. Evaluation & Scripts
The `scripts/` directory contains utilities for evaluating the system performance.

- **Generate Predictions for Test Set**:
  ```bash
  python scripts/generate_predictions.py
  ```
  This will produce `test_prediction.jsonl`.

- **Evaluate Retrieval Performance**:
  ```bash
  python scripts/eval_retrieval.py
  ```

- **Evaluate RAG Generation**:
  ```bash
  python scripts/eval_rag.py
  ```

## 📂 Project Structure

```text
NLPRAG2025-master1/
├── main.py                     # Core RAG pipeline entry point
├── requirements.txt            # Python dependencies
├── test_prediction.jsonl       # Generated output for test set
├── README.md                   # Project documentation
│
├── retrieval/                  # [Module 1] Retrieval Algorithms
│   ├── bm25.py                 # Sparse retrieval
│   ├── dense.py                # Dense retrieval (MPNet)
│   ├── instruction_dense.py    # Instruction-based dense (E5)
│   ├── hybrid.py               # Hybrid (Sparse + Dense)
│   ├── multivector.py          # Multi-vector (ColBERT-style)
│   └── static_embed.py         # Static embeddings (GloVe)
│
├── generator/                  # [Module 2] Generation Logic
│   ├── rag_llm.py              # Basic RAG generation
│   ├── multi_turn.py           # Feature A: Multi-turn conversation
│   └── agentic.py              # Feature B: Agentic workflow
│
├── retrieval pred outputs/     # evaluations of retrieval module
│
├── UI/                         # [Module 3] User Interfaces
│   ├── terminal_ui.py          # CLI entry point
│   └── gui_ui.py               # GUI entry point
│
├── scripts/                    # Evaluation & Utility Scripts
│   ├── generate_predictions.py # Main prediction script
│   ├── eval_retrieval.py       # Retrieval metrics
│   └── ...
│
└── hf_cache/                   # Model & Data Cache
    ├── datasets/               # Cached datasets
    ├── embeddings/             # FAISS indexes
    └── hub/                    # HuggingFace models
```

## ⚙️ Configuration
- **Model Cache**: All models and datasets are stored in `hf_cache/` to avoid repeated downloads.
- **Environment**: The system automatically configures `HF_HOME` to point to this local cache.

## 👥 Team Members
- **Group 15**
- Zhang Yuanlin (25104351g)
- CHEN Zhuokai (25106193g)
- LIU Huawei (25104825g)
- LUO Senhang (25123978g)

---
*Date: November 2025*
