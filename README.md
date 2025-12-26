# Efficient Query Routing: Attention Analysis & Benchmarking

This repository contains the code and experimental artifacts for the Master’s Thesis:

**Author:** Gaurav Kumar (24M0786)  
**Guide:** Prof. Sunita Sarawagi  
**Institution:** IIT Bombay  

---

## 📌 Abstract

Large Language Models (LLMs) are increasingly used as retrievers and routers in structured systems such as Text-to-SQL and tool 
selection. However, their behavior under long contexts remains poorly understood.  

This project studies **database schema routing**—selecting the correct database schema(s) for a natural language query—through a 
combination of:

- **White-box attention flow analysis**, to understand *how* LLMs retrieve information.
- **End-to-end benchmarking**, to evaluate *which routing strategies work best* in practice.

The work reveals strong architectural biases in LLM attention and demonstrates how different prompting and ranking strategies trade 
off accuracy and latency.

---

## 🧠 Part 1: Attention Flow Analysis

We conduct a fine-grained attention analysis on **Llama-3.1-8B-Instruct** to understand how it retrieves information from long schema 
contexts.

### 🔑 Key Findings

#### 1. Positional Bias (“J-Curve”)
- Total attention exhibits a **strong recency bias**.
- Schemas placed near the **end of the prompt receive disproportionately high attention**, regardless of relevance.
- A positional sweep of the gold schema produces a characteristic **J-shaped attention curve**.

#### 2. Attention Fingerprints
- Fixed sets of irrelevant (“distractor”) schemas induce **deterministic attention patterns**.
- These attention peaks and troughs persist across different queries, indicating architectural bias rather than semantic reasoning.

#### 3. Specialist (Reasoning) Heads
- A small, stable subset of attention heads (≈ Top 20) consistently:
  - Ignore positional bias
  - Perform robust semantic matching
- When analysis is restricted to these heads, the J-curve disappears.
- This shows that semantic understanding exists but is drowned out by non-specialized heads.

---

## 📂 Part 1: Code Structure (Attention Analysis)

Scripts for systematic positional sweeps and attention aggregation:

- **`llama_3.1_8B_+ve_-ve_all_in_one_attention_all_heads_aggregration.py`**  
  Runs a positional sweep for a single query, generating:
  - Layer/head attention heatmaps  
  - The global J-curve of total attention

- **`llama_3.1_8B_attention_aggregration.py`**  
  Runs sweeps across multiple queries to compute an **Average Success Matrix**, identifying consistently useful heads.

- **`imp_head_analysis.py`**  
  Visualization utilities for head-importance heatmaps and generalized performance analysis.

- **`llama_3.1_8B_attention_imp_heads_aggregration.py`**  
  Verification script that repeats positional sweeps using only the Top 20 reasoning heads, demonstrating robustness to positional 
bias.

---

## 🚦 Part 2: Routing Strategy Benchmarking

We benchmark database routing strategies on **BIRD** and **SPIDER** using:
- **Llama-3.1-8B-Instruct**
- **Qwen-2.5-7B-Instruct**

### Routing Strategies

#### 1. One-by-One (Pointwise Classification)
Each schema is evaluated independently:
> *Is schema X relevant? (Yes / No)*

- **Pros:** High accuracy, robust reasoning
- **Cons:** Very high latency

#### 2. All-in-One (Listwise Ranking)
A single-pass ranking:
> *Rank the top-K relevant schemas from this list*

- **Pros:** Low latency
- **Cons:** Suffers from context length limits and distraction effects

---

## 📂 Part 2: Code Structure (Benchmarking)

### All-in-One Strategy
- **`llama_3.1_8B_+ve_-ve_all_in_one_<dataset>.py`**  
  All-in-One Benchmarking for Llama-3.1-8B.

- **`qwen_2.5_7B_+ve_-ve_all_in_one_<dataset>.py`**  
  All-in-One benchmarking for Qwen-2.5-7B.

### One-by-One Strategy
- **`llama_3.1_8B_+ve_-ve_one_by_one_<dataset>.py`**  
  Pointwise classification benchmark for Llama-3.1-8B.
  
- **`qwen_2.5_7B_+ve_-ve_one_by_one_<dataset>.py`**  
  Pointwise classification benchmark for Qwen-2.5-7B.

---

## �� Benchmark Results (Summary)

| Strategy     | Accuracy | Latency   | Observation |
|--------------|----------|-----------|-------------|
| One-by-One   | High     | Very High | Best for precision-critical tasks |
| All-in-One   | Moderate | Low       | Fast but vulnerable to distraction |

**Key Observations**
- Qwen-2.5-7B consistently outperformed Llama-3.1-8B in the One-by-One setting  
  (e.g., ~89% Recall@1 on BIRD vs ~71%).
- Adding **+ve / −ve in-context examples** significantly boosts One-by-One performance.
- In-context examples have mixed or negligible impact in All-in-One routing.

---

## 🏆 RankGPT Experiments (Listwise Reranking)

We also implement a **RankGPT-style listwise reranking** approach, treating schema routing as a sliding-window ranking task.

### Motivation
- Combines the global view of All-in-One ranking with iterative refinement.
- Mitigates the **Lost-in-the-Middle** problem while maintaining ranked outputs.
- Designed to re-rank candidates from a cheaper retriever (e.g., BM25). In our case, we consider the prompt order to be the ground 
ranking to start from.

### Files
- **`rank_gpt_reranker.py`**  
  Core RankGPT implementation using sliding windows and permutation-based ranking prompts.

- **`rankGPT_final.py`**  
  Driver script for running RankGPT on **BIRD**, **SPIDER**, or **ToolE** datasets.

---
