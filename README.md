# 🔍 Explainable Misinformation Detection

> **Module:** Deep Learning and Generative AI  
> **Team:** Tanay Kadam · Priya · Namratha · Lokesh  
> **Institution:** National College of Ireland  

---

## 📌 Project Overview

This project builds an end-to-end pipeline to detect misinformation in news and social media text using deep learning and generative AI. The system not only classifies a statement as **Real** or **Fake** but also generates a **human-readable explanation** for each prediction using GPT-2.

The key challenge addressed is the lack of transparency in automated misinformation detection systems. Users often distrust AI predictions when no explanation is provided. This project combines:

- Classical machine learning (TF-IDF + Logistic Regression) as a baseline
- Transformer-based deep learning (DistilBERT) for improved classification
- LIME for word-level explainability
- GPT-2 for generating natural language supporting explanations

---

## 🗂️ Project Structure

```
Explainable-Misinformation-Detection/
│
├── Explainable-Misinformation-Detection_.ipynb   # Main notebook (final pipeline)
├── README.md                                      # This file
├── requirements.txt                               # All Python dependencies

```

---

## 📊 Datasets Used

Data was collected from two publicly available sources — no API keys required:

| Source | Type | Label | Access |
|--------|------|-------|--------|
| [GDELT Project API](https://www.gdeltproject.org/) | Real-time global news articles | Real (1) | Free, no key needed |
| [Reddit Public JSON](https://www.reddit.com/r/politics.json) | User-generated social media posts | Real (1) | Free, no key needed |
| Synthetic fake phrases | Common misinformation patterns from literature | Fake (0) | Generated in-code |

> **Note:** GDELT and Reddit provide real news content. Fake samples were constructed from well-known misinformation patterns described in academic literature to create a balanced binary classification dataset.

---

## ⚙️ Pipeline

```
Raw Text (GDELT + Reddit)
        ↓
Text Cleaning & Preprocessing
        ↓
Train / Test Split (80/20, stratified)
        ↓
┌─────────────────────┐     ┌──────────────────┐
│  Baseline Model     │     │  DistilBERT       │
│  TF-IDF + LR        │     │  Transformer      │
└─────────────────────┘     └──────────────────┘
        ↓                           ↓
  Metrics & Confusion Matrix   Metrics & Confusion Matrix
        ↓
  LIME Explainability
  (Top contributing words per prediction)
        ↓
  GPT-2 Explanation Generation
  (Human-readable natural language justification)
        ↓
  Final Comparison Table + Dashboard
```

---

## 🧠 Models

### 1. Baseline — TF-IDF + Logistic Regression
- TF-IDF vectorisation with bigrams (`ngram_range=(1,2)`, `max_features=20000`)
- Logistic Regression (`max_iter=1000`, `C=1.0`)
- Fast, interpretable, strong baseline

### 2. DistilBERT
- `distilbert-base-uncased` from HuggingFace Transformers
- 40% smaller and 60% faster than BERT
- Fine-tuned for binary sequence classification
- Trained with AdamW optimiser, linear warmup scheduler
- Single epoch on reduced subset due to CPU constraints

---

## 🔍 Explainability

### LIME (Local Interpretable Model-agnostic Explanations)
- Identifies which words in a statement contributed most to the prediction
- Applied to the baseline model for speed
- Produces a ranked list of words with positive/negative influence

### GPT-2 Natural Language Explanations
- Generates a human-readable sentence explaining each prediction
- Prompt-based: `"The statement ... is FAKE/REAL because"`
- **Important:** GPT-2 explanations are illustrative only and may not always be factually reliable. They are designed to improve user trust and understanding, not to serve as ground-truth justifications.

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline (TF-IDF + LR) | — | — | — | — |
| DistilBERT | — | — | — | — |

> **Note:** Fill in the actual numbers from your notebook output before submission.

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
requests
tensorflow
torch
transformers
lime
```

Install all at once:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests tensorflow torch transformers lime
```

---

## 📁 How to Upload to GitHub

### Step 1 — Create a GitHub account
Go to [https://github.com](https://github.com) and sign up if you don't have an account.

### Step 2 — Create a new repository
1. Click the **+** button (top right) → **New repository**
2. Name it: `Explainable-Misinformation-Detection`
3. Set to **Public**
4. Tick **Add a README file** → **NO** (we already have one)
5. Click **Create repository**

### Step 3 — Install Git on your computer
Download from: [https://git-scm.com/downloads](https://git-scm.com/downloads)

### Step 4 — Open Command Prompt
Press `Windows + R` → type `cmd` → Enter

### Step 5 — Navigate to your project folder
```bash
cd C:\Users\Kanaga Priya\path\to\your\project
```

Find the path by running this in Jupyter:
```python
import os
print(os.getcwd())
```

### Step 6 — Initialise Git and push
```bash
git init
git add .
git commit -m "Final submission: Explainable Misinformation Detection"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Explainable-Misinformation-Detection.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### Step 7 — Verify
Go to `https://github.com/YOUR_USERNAME/Explainable-Misinformation-Detection` — your files should all be visible.

---

## 🔁 Reproducibility Notes

- All results in the project report are taken directly from the notebook outputs
- Random seed `42` is used throughout for reproducibility
- GDELT and Reddit data is fetched live — results may vary slightly depending on when the notebook is run
- DistilBERT was trained on a reduced subset due to CPU constraints
- LIME explanations are deterministic given the same model and input
- GPT-2 explanations use `do_sample=False` for deterministic output

---

## 📚 References

- Wang, W. Y. (2017). "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection. *ACL 2017*
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD 2016*
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. *NeurIPS 2019*
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI GPT-2*
- GDELT Project: [https://www.gdeltproject.org](https://www.gdeltproject.org)
- Reddit API: [https://www.reddit.com/dev/api](https://www.reddit.com/dev/api)
