# 🧠 DermaKG-Bench  
### A Fitzpatrick-Stratified Knowledge Graph Benchmark for Equitable Dermatology Drug Recommendation

---

## 📌 Overview

**DermaKG-Bench** is a fairness-aware biomedical knowledge graph system designed to address **demographic bias in dermatology drug recommendation**.

Traditional knowledge graphs like PrimeKG treat all medical evidence equally — but in reality, **clinical evidence is heavily skewed toward lighter skin types (FST I–III)**. This project introduces:

- ✅ **Fitzpatrick Skin Type (FST) stratification**
- ✅ **Bayesian posterior modeling of evidence**
- ✅ **Equity gap measurement**
- ✅ **Fairness-aware recommendation pipeline**

---

## 🚀 Key Features

### 🧬 Fairness-Aware Knowledge Graph
- Stratifies drug-disease relationships by **FST I–III vs IV–VI**
- Identifies **hidden demographic bias in medical data**

### 📊 Bayesian Evidence Modeling (SCP-KG + LEH)
- Uses **Beta posteriors** for each edge
- Handles **low-data regimes** via empirical Bayes

### ⚖️ Counterfactual Equity Gap (CEG)
- Measures disparity using **KL divergence**
- Captures both:
  - Mean difference (effectiveness)
  - Uncertainty difference (confidence)

### 🕳️ Topological Void Detection (TVD)
- Detects missing knowledge regions in minority subgraphs
- Uses persistence-based graph analysis

### 🔍 Inverse Graph Reasoning (IGR)
- Generates **new candidate drug-disease links**
- Prioritizes **equity improvement + cost efficiency**

### 🛡️ Clinical Safety Layer
- ATC-based filtering to avoid unsafe recommendations
- Independent **contraindication oracle**

### 🎯 Conformal Calibration (SSCC)
- Ensures **fair prediction reliability across subgroups**
- Maintains coverage guarantees for minority populations

---

## 🏗️ System Architecture

### Pipeline Overview

```
Data Sources
   ↓
Knowledge Graph Construction
   ↓
FST Stratification
   ↓
SCP-KG (Bayesian Posterior Modeling)
   ↓
Equity Analysis (CEG + TVD)
   ↓
IGR Candidate Generation
   ↓
Safety Filtering (ATC)
   ↓
LTR Ranking + MMR Diversification
   ↓
Conformal Calibration (SSCC)
   ↓
Final Recommendations
```

---

## 📂 Data Sources

| Dataset | Role |
|--------|------|
| PrimeKG | Core biomedical KG |
| Fitzpatrick17k | Skin-type labeled dermatology images |
| DermaCon-IN | Indian cohort (FST III–VI heavy) |
| DrugCentral | Drug-disease relations |
| OpenTargets | Supplementary biomedical links |

---

## 🧠 Core Contributions

### 1. First FST-Stratified KG Benchmark
- 49 diseases, 166 drugs, 419 indication edges  
- Designed for **small-data fairness evaluation**

### 2. Subgroup-Conditional Posterior KG
- Bayesian modeling of drug effectiveness per demographic group

### 3. Equity Diagnostics Suite
- CEG → disparity measurement  
- TVD → structural bias detection  
- IGR → missing knowledge discovery  

### 4. Fairness Without Accuracy Loss
- Matches baseline performance  
- Significant reduction in fairness gap  

---

## 📁 Project Files

👉 [View Architecture Details](Architecture.md)  
👉 [Open Notebook (Code)](Code.ipynb)

## ▶️ How to Use

- Open the notebook directly on GitHub  
or  
- Download and run using Jupyter / Google Colab  

---

## ⚙️ Requirements

```
pip install pandas numpy igraph sentence-transformers rank-bm25 fuzzywuzzy
```

---

## 💡 What This Project Does

- Builds a biomedical knowledge graph  
- Applies Fitzpatrick skin type (FST) fairness analysis  
- Detects bias in drug recommendations  
- Suggests more equitable treatments  

---

## ⚠️ Note

This project is for research purposes only and not for clinical use.

---


## 💡 Why This Matters

Most AI in healthcare works well for **majority populations**.  

> 🧑🏾‍⚕️ *AI should work equally well for all skin types — not just the majority.*
