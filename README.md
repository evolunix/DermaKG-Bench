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

<pre>
dermakg-bench/
│
├── <a href="README.md">README.md</a>                               ← This file
│
├── code/
│   ├── <a href="code/dermakg_kaggle_all_in_one.py">dermakg_kaggle_all_in_one.py</a>        ← Full end-to-end pipeline (data construction, SCP-KG fitting, 
│   |                                         CEG/TVD/IGR, baseline eval)
│   ├── <a href="code/dermakg_baseline_comparison_cell.py">dermakg_baseline_comparison_cell.py</a> ← Baseline evaluation: 12-method FST-stratified Hits@1/5/10, 
│   |                                         MRR, NDCG@10 across 5 CV folds
│   ├── <a href="code/dermakg_igr_ablation_cell.py">dermakg_igr_ablation_cell.py</a>        ← IGR ablation: BED-IGR vs. equity_gain/cost heuristic 
│   |                                         correlation analysis (Appendix B)
│   ├── <a href="code/dermakg_disagreement_table_cell.py">dermakg_disagreement_table_cell.py</a>  ← Disagreement analysis between ranking methods
│   └── <a href="code/Kaggle_reproduction.ipynb">Kaggle_reproduction.ipynb</a>           ← Kaggle-ready reproduction notebook (calls the above 
│                                             modules in order)
│
├── <a href="pipeline_metrics.json">pipeline_metrics.json</a>                   ← End-to-end pipeline run metrics & timing profile
|
├── <a href="croissant.json">croissant.json</a>                    ← MLCommons Croissant dataset metadata and schema definition
│
├── data/
│   ├── <a href="data/evidence_records.json">evidence_records.json</a>               ← Pre-built per-FST Evidence Records with log-scaled 
│   |                                         weights for all 1,874 (edge, group) pairs
│   ├── <a href="data/folds.json">folds.json</a>                          ← 5-fold CV split indices (seeds: 42, 142, 242, 342, 442)
│   ├── <a href="data/contraindications.json">contraindications.json</a>              ← 278 PrimeKG contraindication edges (safety oracle)
│   ├── <a href="data/atc_seed_map.csv">atc_seed_map.csv</a>                    ← ~150-drug seed map across 7 disease domains
│   └── <a href="data/atc_domain_constraints.json">atc_domain_constraints.json</a>         ← ATC class allow-lists and block-lists per domain
│
├── results/
│   ├── <a href="results/paper_table_main.csv">paper_table_main.csv</a>                ← Main results table (paper Table 2 / Table 3)
│   ├── <a href="results/paper_table4_top10.csv">paper_table4_top10.csv</a>              ← Top-10 IGR candidates (paper Table 4)
│   ├── <a href="results/paper_table4.tex">paper_table4.tex</a>                    ← LaTeX source for paper Table 4
│   ├── <a href="results/paper_table4_summary.txt">paper_table4_summary.txt</a>            ← Human-readable Table 4 summary
│   ├── <a href="results/paper_table_disagreement.csv">paper_table_disagreement.csv</a>        ← Disagreement cases between methods
│   │
│   ├── ablation/
│   │   ├── <a href="results/ablation/ablation_igr_correlations.csv">ablation_igr_correlations.csv</a>              ← IGR ablation: BED-IGR vs. equity_gain/
│   |   |                                                cost correlation
│   │   ├── <a href="results/ablation/ablation_igr_disagreement_cases.csv">ablation_igr_disagreement_cases.csv</a>        ← Ablation cases where IGR & heuristic diverge
│   │   ├── <a href="results/ablation/ablation_igr_per_disease_correlations.csv">ablation_igr_per_disease_correlations.csv</a>  ← Per-disease Spearman ρ (Appendix B)
│   │   └── <a href="results/ablation/ablation_igr_top_k_overlap.csv">ablation_igr_top_k_overlap.csv</a>             ← Top-k overlap between IGR variants
│   │
│   ├── benchmark/
│   │   ├── <a href="results/benchmark/skin_stats_v5_5.csv">skin_stats_v5_5.csv</a>             ← Per-FST disease cohort statistics (Fitzpatrick17k + 
│   |   |                                     DermaCon-IN)
│   │   └── <a href="results/benchmark/structural_voids.csv">structural_voids.csv</a>            ← TVD-detected topological voids (55 voids)
│   │
│   ├── scp_kg/
│   │   ├── <a href="results/scp_kg/scp_all_posteriors.csv">scp_all_posteriors.csv</a>          ← All Beta posteriors per (edge, FST subgroup)
│   │   ├── <a href="results/scp_kg/scp_eb_prior.json">scp_eb_prior.json</a>               ← Fitted empirical-Bayes LEH prior (α₀, β₀)
│   │   └── <a href="results/scp_kg/scp_kg_summary.txt">scp_kg_summary.txt</a>              ← SCP-KG construction summary & statistics
│   │
│   ├── igr/
│   │   ├── <a href="results/igr/igr_all_candidates.csv">igr_all_candidates.csv</a>          ← All 463 IGR candidates (Type A/B/C) pre-safety filter
│   │   ├── <a href="results/igr/igr_disease_gaps.csv">igr_disease_gaps.csv</a>            ← Per-disease directional equity scores (Stage 1)
│   │   ├── <a href="results/igr/igr_pareto_frontier.csv">igr_pareto_frontier.csv</a>         ← 4 Pareto-optimal candidates (Stage 4)
│   │   ├── <a href="results/igr/igr_quick_wins.csv">igr_quick_wins.csv</a>              ← 9 Type-A quick wins (acquisition cost ≤ 5)
│   │   └── <a href="results/igr/safety_rejected.csv">safety_rejected.csv</a>             ← 96 candidates rejected by ATC safety filter
│   │
│   └── eval/
│       ├── <a href="results/eval/baseline_comparison.csv">baseline_comparison.csv</a>         ← Hits@1/5/10, MRR, NDCG@10 across 12 methods (FST-stratified)
│       ├── <a href="results/eval/ceg_top100.csv">ceg_top100.csv</a>                  ← Top-100 edges by Counterfactual Equity Gap (CEG)
│       └── <a href="results/eval/paper_table_fairness.csv">paper_table_fairness.csv</a>        ← FST fairness gap metrics (paper Table — fairness)
│
└── <a href="LICENSE">LICENSE</a>
</pre>

***

## Code Modules

| File | Description |
|------|-------------|
| `dermakg_kaggle_all_in_one.py` | Full end-to-end pipeline: data ingestion → SCP-KG fitting → CEG/TVD computation → IGR → baseline evaluation → table export |
| `dermakg_baseline_comparison_cell.py` | Runs all 12 methods over 5-fold CV; outputs FST-stratified Hits@1/5/10, MRR, NDCG@10 |
| `dermakg_igr_ablation_cell.py` | Ablation study comparing BED-IGR to the equity_gain/cost heuristic; computes per-disease Spearman ρ (Appendix B) |
| `dermakg_disagreement_table_cell.py` | Identifies and tabulates cases where method rankings diverge |
| `Kaggle_reproduction.ipynb` | Kaggle-ready notebook that imports and runs the above modules sequentially for full reproduction |

### Running the Pipeline

```bash
# Clone the repository
git clone https://github.com/evolunix/DermaKG-Bench.git
cd dermakg-bench

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python code/dermakg_kaggle_all_in_one.py

# Or run individual components
python code/dermakg_baseline_comparison_cell.py
python code/dermakg_igr_ablation_cell.py
python code/dermakg_disagreement_table_cell.py
```

Alternatively, open `code/Kaggle_reproduction.ipynb` on Kaggle or locally for a step-by-step walkthrough.

***

## File Reference

### Data Files

| File | Description |
|------|-------------|
| `evidence_records.json` | Pre-built per-FST Evidence Records with log-scaled weights for all 1,874 (edge, group) pairs |
| `folds.json` | 5-fold CV split indices (seeds: 42, 142, 242, 342, 442) over 419 indication edges |
| `contraindications.json` | 278 PrimeKG contraindication edges held back as independent safety oracle |
| `atc_seed_map.csv` | ~150-drug seed map across 7 disease domains used by the ATC safety filter |
| `atc_domain_constraints.json` | ATC class allow-lists and block-lists per disease domain |
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
| `pipeline_metrics.json` | Runtime profile and fold-level raw numbers (Appendix C equivalent) |

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
