# DermaKG-Bench

> **A Fitzpatrick-Stratified Benchmark and Empirical-Bayes Diagnostic Suite for Equitable Dermatology Drug Recommendation**
>
> *Submitted to NeurIPS 2026 — Do not distribute*

***

## Overview

Knowledge-graph (KG) drug recommenders built on PrimeKG, HetioNet, and 1DRKG are **demographically blind**: their edges record drug–disease relationships without any indication of who the underlying evidence was collected from. DermaKG-Bench closes that gap.

This repository contains:

- **DermaKG-Bench** — an FST-stratified benchmark joining PrimeKG indications to Fitzpatrick17k and DermaCon-IN cohort statistics (49 diseases, 166 drugs, 419 indication edges, 278 contraindication edges as independent safety oracle, 5-fold CV harness over 12 methods).
- **Diagnostic Suite** — SCP-KG with Empirical-Bayes LEH prior pooling, Counterfactual Equity Gap (CEG), Topological Void Detection (TVD), and Inverse Graph Reasoning (IGR).
- **Safety & Calibration** — ATC safety filter and Subgroup-Stratified Conformal Calibration (SSCC).

**Key result:** SCP-KG-Posterior statistically ties the strongest retrieval baseline on Hits@10 (0.570 vs 0.560, p = 0.069) while cutting the FST fairness gap by **21×** (0.004 vs 0.086).

***

## Repository Structure

```
dermakg-bench/
│
├── README.md                          ← This file
│
├── Code.ipynb                         ← Main pipeline notebook (data construction,
│                                         SCP-KG fitting, CEG/TVD/IGR, baseline eval)
│
├── pipeline_metrics.json              ← End-to-end pipeline run metrics & timing profile
│
├── data/
│   ├── benchmark/
│   │   ├── skin_stats_v5_5.csv        ← Per-FST disease cohort statistics (Fitzpatrick17k + DermaCon-IN)
│   │   └── structural_voids.csv       ← TVD-detected topological voids (55 voids)
│   │
│   ├── scp_kg/
│   │   ├── scp_all_posteriors.csv     ← All Beta posteriors per (edge, FST subgroup)
│   │   ├── scp_eb_prior.json          ← Fitted empirical-Bayes LEH prior (α₀, β₀)
│   │   └── scp_kg_summary.txt         ← SCP-KG construction summary & statistics
│   │
│   ├── igr/
│   │   ├── igr_all_candidates.csv     ← All 463 IGR candidates (Type A/B/C) pre-safety filter
│   │   ├── igr_disease_gaps.csv       ← Per-disease directional equity scores (Stage 1)
│   │   ├── igr_pareto_frontier.csv    ← 4 Pareto-optimal candidates (Stage 4)
│   │   ├── igr_quick_wins.csv         ← 9 Type-A quick wins (acquisition cost ≤ 5)
│   │   └── safety_rejected.csv        ← 96 candidates rejected by ATC safety filter
│   │
│   └── eval/
│       ├── baseline_comparison.csv    ← Hits@1/5/10, MRR, NDCG@10 across 12 methods (FST-stratified)
│       ├── ceg_top100.csv             ← Top-100 edges by Counterfactual Equity Gap (CEG)
│       └── paper_table_fairness.csv   ← FST fairness gap metrics (paper Table — fairness)
│
├── results/
│   ├── paper_table_main.csv           ← Main results table (paper Table 2 / Table 3)
│   ├── paper_table4_top10.csv         ← Top-10 IGR candidates (paper Table 4)
│   ├── paper_table4.tex               ← LaTeX source for paper Table 4
│   ├── paper_table4_summary.txt       ← Human-readable Table 4 summary
│   ├── paper_table_disagreement.csv   ← Disagreement cases between methods (paper Table)
│   │
│   └── ablation/
│       ├── ablation_igr_correlations.csv          ← IGR ablation: BED-IGR vs. equity_gain/cost correlation
│       ├── ablation_igr_disagreement_cases.csv    ← Ablation cases where IGR & heuristic diverge
│       ├── ablation_igr_per_disease_correlations.csv  ← Per-disease Spearman ρ (Appendix B)
│       └── ablation_igr_top_k_overlap.csv         ← Top-k overlap between IGR variants
│
└── LICENSE
```

***

## File Reference

### Core Files

| File | Description |
|------|-------------|
| `Code.ipynb` | Full pipeline: data ingestion → SCP-KG fitting → CEG/TVD computation → IGR → baseline evaluation → table generation |
| `pipeline_metrics.json` | Runtime profile and fold-level raw numbers (Appendix C equivalent) |

### Data Files

| File | Description |
|------|-------------|
| `skin_stats_v5_5.csv` | Per-FST cohort statistics from Fitzpatrick17k and DermaCon-IN for 49 benchmark diseases |
| `structural_voids.csv` | 55 topological voids detected by TVD; majority-side persistence vs. minority Jaccard-matched persistence |
| `scp_all_posteriors.csv` | Beta posterior parameters (α, β) for every (edge, subgroup) pair in the SCP-KG (518 edges × 2 subgroups) |
| `scp_eb_prior.json` | Fitted LEH empirical-Bayes prior: `{"alpha0": 1.10, "beta0": 0.50}` |
| `scp_kg_summary.txt` | SCP-KG construction log: edge counts, evidence record counts, EB fitting details |
| `igr_all_candidates.csv` | All 463 IGR candidates before ATC safety filter (columns: type, disease, drug, confidence, CEG, cost) |
| `igr_disease_gaps.csv` | Directional equity scores per disease (Stage 1 of IGR) |
| `igr_pareto_frontier.csv` | 4 non-dominated Pareto candidates in (cost, EIG·conf) space |
| `igr_quick_wins.csv` | 9 Type-A quick wins (existing PrimeKG edges, minority weight < 60% of majority, cost ≤ 5) |
| `safety_rejected.csv` | 96 candidates rejected by ATC domain safety filter (top reasons: `atc_not_allowlisted_for_neoplastic_skin`, `atc_blocked_D07_for_neoplastic_skin`) |
| `baseline_comparison.csv` | FST-stratified Hits@1/5/10, MRR, NDCG@10 for all 12 methods across 5 CV folds |
| `ceg_top100.csv` | Top-100 drug–disease edges ranked by CEG (KL divergence between FST subgroup posteriors) |
| `paper_table_fairness.csv` | Fairness gap metrics used in the paper's fairness table |

### Results & Ablation Files

| File | Description |
|------|-------------|
| `paper_table_main.csv` | Main benchmark results table (retrieval, KG-embedding, posterior-based families) |
| `paper_table4_top10.csv` | Top-10 IGR acquisition candidates for Table 4 |
| `paper_table4.tex` | LaTeX source for Table 4 (IGR recommendations) |
| `paper_table4_summary.txt` | Plain-text summary of Table 4 findings |
| `paper_table_disagreement.csv` | Cases where ranking methods disagree (supports disagreement analysis section) |
| `ablation_igr_correlations.csv` | Overall Spearman ρ between BED-IGR and equity_gain/cost heuristic (Appendix B) |
| `ablation_igr_disagreement_cases.csv` | The ~17% of Type-A cases where IGR and heuristic diverge |
| `ablation_igr_per_disease_correlations.csv` | Per-disease Spearman ρ (83% of diseases with |ρ| ≥ 0.95; Appendix B) |
| `ablation_igr_top_k_overlap.csv` | Top-k list overlap between IGR variants (Appendix B) |

***

## Quickstart

```bash
# Clone the repository
git clone https://github.com/evolunix/DermaKG-Bench.git
cd dermakg-bench

# Install dependencies
pip install -r requirements.txt   # (or see notebook cell 1 for inline installs)

# Open the main pipeline notebook
jupyter notebook Code.ipynb
```

The notebook is self-contained and runs the full pipeline in order:
1. Data loading and per-FST evidence record construction
2. SCP-KG fitting with LEH empirical-Bayes prior
3. CEG and TVD computation
4. IGR candidate generation (Stages 1–4) with ATC safety filtering
5. 12-method benchmark evaluation (5-fold CV)
6. SSCC conformal calibration
7. Table and result file export

***

## Benchmark Details

### Benchmark Scale

| Entity | Count |
|--------|-------|
| Diseases | 49 |
| Drugs | 166 |
| Indication edges | 419 |
| Contraindication edges (safety oracle) | 278 |
| CV folds | 5 (seeds: 42, 142, 242, 342, 442) |
| Methods evaluated | 12 |

### Source Datasets

| Dataset | Description | Access |
|---------|-------------|--------|
| **PrimeKG** | Relational backbone (8.1M edges, 90K entities) | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM) |
| **Fitzpatrick17k** | 16,577 dermoscopic images with FST I–VI labels | [GitHub](https://github.com/mattgroh/fitzpatrick17k) |
| **DermaCon-IN** | 5,450 records, Indian cohort, FST III–VI predominance, Monk Skin Tone annotations | [DermaCon](https://derma-con.github.io/) |

### Evaluated Methods (12)

| Family | Methods |
|--------|---------|
| Trivial baselines | Random, Frequency |
| Retrieval | BM25-symptom, Jaccard-ATC, Degree-centrality |
| KG Embedding | TransE, DistMult, ComplEx, RotatE |
| Posterior-based | SCP-KG-Prior, SCP-KG-Posterior (LEH), SCP-KG-Posterior+SSCC |

***

## Diagnostic Suite

### SCP-KG with LEH (§4.1)
Subgroup-conditional Beta posteriors with empirical-Bayes prior pooling. Fitted prior: **Beta(1.10, 0.50)** over 838 (edge, group) pairs.

### Counterfactual Equity Gap — CEG (§4.2)
`CEG(e) = KL(θ_{e,g_maj} ∥ θ_{e,g_min})`. Decomposes into `Δ_mean` (signal-driven gap) and `Δ_prec` (precision-driven gap).

### Topological Void Detection — TVD (§4.3)
Identifies 55 structural voids using 0- and 1-dimensional persistence on weighted subgroup graphs. Most are 1-cycles among related drug classes whose connectivity collapses under FST IV–VI filtration.

### Inverse Graph Reasoning — IGR (§4.4)
4-stage framework: disease gap detection → missing-edge proposal (Types A/B/C) → BED-IGR unified scoring → Pareto ranking. Outputs 4 Pareto candidates and 9 Type-A quick wins.

### ATC Safety Filter (§4.5)
Rejects clinically incompatible candidates using a ~150-drug seed map across 7 disease domains. Rejected 96 of 463 candidates.

### SSCC (§4.6)
Subgroup-stratified conformal calibration correcting for the majority-to-minority distribution shift at calibration time via importance-ratio weighting.

***
<!--
## Reproducibility

The benchmark, pre-built Evidence Record JSONs, fold splits, evaluation harness, and Croissant metadata are released at:

> **https://anonymous.4open.science/r/dermakg-bench-XXXX** *(anonymised for review)*

Croissant schema document: Appendix E of the paper.
Runtime profile and 5-fold raw numbers: `pipeline_metrics.json` and Appendix C.

***

## Citation

```bibtex
@inproceedings{dermakg2026,
  title     = {{DermaKG-Bench}: A {Fitzpatrick}-Stratified Benchmark and Empirical-{Bayes}
               Diagnostic Suite for Equitable Dermatology Drug Recommendation},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Under review}
}
```

***
-->
## License

To be released upon de-anonymisation. All source datasets (PrimeKG, Fitzpatrick17k, DermaCon-IN) retain their original licenses.
