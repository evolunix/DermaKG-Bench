# DermaKG-Bench: Complete System Architecture

> **Project:** DermaKG-Bench — A Fitzpatrick-Stratified Benchmark and Empirical-Bayes Diagnostic Suite for Equitable Dermatology Drug Recommendation  

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Ingestion Layer](#2-data-ingestion-layer)
3. [Multi-Source Knowledge Graph Construction](#3-multi-source-knowledge-graph-construction)
4. [MONDO Ontology & Entity Linking](#4-mondo-ontology--entity-linking)
5. [Disease-Drug Index (DDIndex)](#5-disease-drug-index-ddindex)
6. [Subgroup-Conditional Posterior KG (SCP-KG) & Latent Evidence Hierarchy (LEH)](#6-subgroup-conditional-posterior-kg-scp-kg--latent-evidence-hierarchy-leh)
7. [Counterfactual Equity Gap (CEG) & Topological Void Detection (TVD)](#7-counterfactual-equity-gap-ceg--topological-void-detection-tvd)
8. [Inverse Graph Reasoning (IGR) Framework](#8-inverse-graph-reasoning-igr-framework)
9. [ATC Safety Filter](#9-atc-safety-filter)
10. [Subgroup-Stratified Conformal Calibration (SSCC)](#10-subgroup-stratified-conformal-calibration-sscc)
11. [Recommender System](#11-recommender-system)
12. [Learn-to-Rank (LTR) Module](#12-learn-to-rank-ltr-module)
13. [Confidence-Gated Decoder (CGD)](#13-confidence-gated-decoder-cgd)
14. [Living Epistemic Hypergraph (LEH) — Knowledge Decay Gradient (DKDG)](#14-living-epistemic-hypergraph-leh--knowledge-decay-gradient-dkdg)
15. [v5.4 Aggregator Node Injection](#15-v54-aggregator-node-injection)
16. [Pipeline Orchestration](#16-pipeline-orchestration)
17. [Evaluation Harness & Benchmark](#17-evaluation-harness--benchmark)
18. [Component Interaction Diagram](#18-component-interaction-diagram)
19. [Key Design Decisions & Trade-offs](#19-key-design-decisions--trade-offs)

---

## 1. System Overview

DermaKG-Bench is an **end-to-end fairness-aware drug recommendation and diagnostic pipeline** built on top of biomedical knowledge graphs. It addresses the demographic blindness of standard KG drug-recommendation systems by introducing Fitzpatrick Skin Type (FST) stratification — stratifying every drug-disease edge by whether the supporting clinical evidence was collected from FST I–III (lighter skin) or FST IV–VI (darker skin) cohorts.

The pipeline consists of **five measurement/diagnostic components** and **two safety/calibration components**:

| Component | Role |
|-----------|------|
| SCP-KG + LEH | Subgroup-conditional Beta posteriors per drug-disease edge |
| CEG | Closed-form counterfactual equity gap |
| TVD | Topological void detection in subgroup-conditional graphs |
| IGR | Candidate-generation framework for absent edges |
| DDIndex | Disease-drug retrieval index with FST-aware scoring |
| ATC Filter | Domain-aware clinical safety filter |
| SSCC | Deployment-time subgroup-stratified conformal calibration |

The pipeline is orchestrated by `runpipeline()` (v5.4: `runpipelinev2()`), which runs sequentially in 10 numbered stages and returns a unified `results` dict.

---

## 2. Data Ingestion Layer

### 2.1 Data Sources

| Dataset | Records | Role |
|---------|---------|------|
| **PrimeKG** | 8,100,498 edges, 90,067 nodes | Relational backbone (indication, off-label, contraindication edges) |
| **Fitzpatrick17k** | 16,577 images | FST I–VI dermoscopic images; 66.8% FST I–III |
| **DermaCon-IN** | 5,450 records | Indian cohort; FST III–VI predominance; expands IV–VI coverage |
| **OpenTargets** | 57 rows (cached) | Supplementary drug-disease associations |
| **DrugCentral** | 89 edges | Additional drug-disease edges; contributes to never-list |
| **DrugBank Map** | 7,957 entries | Drug name → DrugBank ID mapping |
| **Clinical GT** | 2,060 diseases | PrimeKG + OT + DrugCentral curated ground truth |
| **Clinical Oracle** | 278 edges | Independent contraindication safety oracle (never seen at training) |

### 2.2 Skin Statistics Computation

From Fitzpatrick17k + DermaCon-IN, per-disease statistics are computed:

- `prevalence_ivvi`: fraction of samples with FST IV–VI
- `fst_i_iii`, `fst_iv_vi`, `fst_total`: raw FST counts per disease
- `has_mst` (Monk Skin Tone dark): boolean flag from DermaCon-IN MST annotations

**Output:** `skinstats` — dict of 315 skin conditions with FST distribution metadata.

### 2.3 Evidence Record Construction

For each PrimeKG edge `e = (disease, rel, drug)` and each subgroup `g ∈ {I–III, IV–VI}`:

```
EvidenceRecord(
    edge=e,
    subgroup=g,
    outcome=1 (indication) | 0 (off-label),
    weight = 1.0 / (1 + log(n_disease,g)),   # log-scaling prevents large-cohort domination
    source="PrimeKG"
)
```

Of 8,100,498 PrimeKG edges: 22,876 skipped (no demographic record), 8,076,586 skipped (relation outside keep-list). The remaining **1,036 unique drug-disease edges** produce **1,874 evidence records** with 90.4% both-subgroup coverage.

---

## 3. Multi-Source Knowledge Graph Construction

### 3.1 Base Graph (PrimeKG)

- Loaded as an `igraph.Graph` object
- 90,067 nodes (diseases, drugs, genes, pathways, etc.)
- 8,100,498 edges typed by `relation` attribute
- Only `indication` and `off-label use` edges are retained for recommendation; `contraindication` edges are reserved as the safety oracle

### 3.2 Graph Enrichment

Additional edges are added from three sources:

1. **OpenTargets** (+49 edges): Disease–target–drug associations
2. **DrugCentral** (+68 edges): Clinical drug-disease pairs
3. **FST Annotation**: 280 disease nodes receive FST distribution attributes from `skinstats`

### 3.3 Population Subgraphs

Two induced subgraphs are built for FST-stratified reasoning:

- **Light subgraph** (`poplight`): Diseases where FST I–III is the majority; 3,768 nodes
- **Dark subgraph** (`popdark`): Diseases where FST IV–VI is the majority; 1,869 nodes

**Final KG (post-build):** 90,067 nodes, 8,100,615 edges (built in ~38s).

### 3.4 v5.4 Aggregator Node Injection

After the base KG is built, synthetic "aggregator" disease nodes are injected for diseases under-represented in PrimeKG:

| Synthetic Node | Indications added | Off-label added |
|---------------|-------------------|-----------------|
| psoriasis | 16 drugs (methotrexate, adalimumab, secukinumab, ...) | 3 drugs |
| vitiligo | 8 drugs (ruxolitinib, tacrolimus, ...) | 4 drugs |
| rosacea | 8 drugs (metronidazole, ivermectin, ...) | 6 drugs |
| eczema | Routed to existing atopic eczema node | — |

Aggregator edges are tagged `datasource=v5.4aggregator` for auditability. This injects **4 nodes and 48 edges**.

---

## 4. MONDO Ontology & Entity Linking

### 4.1 MONDO Ontology Loader

- Loads the MONDO disease ontology: **26,709 terms**, **129,101 labels indexed**
- Provides `get_domain(disease_name)` for dermatology domain classification
- Domain seed mapping covers 7 clinical domains: `inflammatory_skin`, `autoimmune_skin`, `neoplastic_skin`, `infectious_skin`, `acneiform`, `pigmentary`, `ophthalmology`

### 4.2 SapBERT Entity Linker

- Model: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
- Encodes all disease vocabulary (~9,019–9,023 entries) into dense embeddings
- Provides semantic similarity for:
  - **Type-B IGR** cross-disease borrowing (cosine similarity between disease embeddings)
  - **Entity linking** at query time (maps free-text disease queries to KG node IDs)
- `EntityLinker` class wraps SapBERT with a normalized lexical fallback layer

---

## 5. Disease-Drug Index (DDIndex)

`DiseaseDrugIndex` is the core retrieval index for disease→drug lookups.

### 5.1 Structure

For each disease node in the KG, DDIndex stores:

- `indications`: set of drug node IDs connected via `indication` edges
- `off_label`: set of drug node IDs connected via `off-label use` edges
- `fst_stats`: per-FST statistics from `skinstats` (copied from KG node attributes)
- `2-hop drugs`: drugs reachable in 2 hops (used for structural gap computation)

**Scale:** 9,019–9,023 diseases indexed; 825–828 with at least one drug edge; 122–133 FST-matched.

### 5.2 Feature Vector Construction

For each drug-disease candidate pair, a **12-dimensional feature vector** is built:

```
[
  rel_indication,       # 1.0 if relation==indication
  rel_off_label,        # 1.0 if relation==off-label
  rel_contra,           # 1.0 if relation==contraindication
  demographic_weight,   # FST IV-VI prevalence proxy
  evidence_density,     # normalized evidence count
  atc_class_match,      # ATC class coherence score
  hyperbolic_dist,      # hyperbolic embedding distance
  app_approved,         # 1.0 if FDA-approved
  app_investigational,  # 1.0 if investigational
  curated_tier,         # data source quality tier (0–3)
  evidence_count,       # min(evidence_count/100, 1.0)
  bias                  # constant 1.0 (bias term)
]
```

---

## 6. Subgroup-Conditional Posterior KG (SCP-KG) & Latent Evidence Hierarchy (LEH)

This is the **core statistical contribution** (C1 in the paper).

### 6.1 Beta Posterior Model

For each edge `e` and subgroup `g`, a latent probability `θ_{e,g}` is modeled:

```
θ_{e,g} | D  ~  Beta(α₀ + s_{e,g},  β₀ + n_{e,g} - s_{e,g})
```

where:
- `n_{e,g}` = weighted total evidence count for `(e, g)`
- `s_{e,g}` = weighted supportive count (null/equivocal observations contribute 0.5 to `s`, 1 to `n`)
- `(α₀, β₀)` = empirical-Bayes prior fitted by LEH

### 6.2 Latent Evidence Hierarchy (LEH) — Prior Fitting

The hierarchical prior `(α₀, β₀)` is fitted by **method-of-moments empirical Bayes** on edges with sufficient evidence in both subgroups:

```
p̄ = mean(s/n),   v = var(s/n)
c = p̄(1-p̄)/v - 1        # precision
α₀ = max(p̄·c, 0.5)
β₀ = max((1-p̄)·c, 0.5)
```

**On DermaKG-Bench:** LEH fits `Beta(1.10, 0.50)` prior from 838 edge-group pairs (419 fully-covered edges).

The LEH divides capsules (edge-group pairs) into epistemic zones:

| Zone | Confidence threshold | Count | % |
|------|---------------------|-------|---|
| Core | global_conf ≥ 0.85, density ≥ 1.0 | 15 | 0.1% |
| Inference | 0.50 ≤ conf < 0.85 | 95 | 0.5% |
| Peripheral | conf < 0.50 | 20,505 | 98.8% |
| Contested | FST disagreement (range > 0.40) | 141 | 0.7% |
| Void | Conflicting evidence | 1 | 0.01% |

**Total LEH capsules: 20,757** (post v5.4, 41,795 built from expanded KG then filtered).

### 6.3 BetaPosterior Object

```python
@dataclass
class BetaPosterior:
    alpha: float
    beta: float

    @property mean()       -> α / (α + β)
    @property variance()   -> αβ / (α+β)²(α+β+1)
    @property neff()       -> α + β   # effective sample size
    def kl_divergence_to(other) -> float  # closed-form via Theorem 1
```

### 6.4 Theorem 1 — CEG Closed Form

For two Beta posteriors `p₁ ~ Beta(α₁, β₁)` and `p₂ ~ Beta(α₂, β₂)`:

```
KL(p₁ || p₂) = log B(α₂,β₂)/B(α₁,β₁)
              + (α₁-α₂)[ψ(α₁) - ψ(α₁+β₁)]
              + (β₁-β₂)[ψ(β₁) - ψ(α₁+β₁)]
```

where `B` is the Beta function and `ψ` is the digamma function.

---

## 7. Counterfactual Equity Gap (CEG) & Topological Void Detection (TVD)

### 7.1 Counterfactual Equity Gap (CEG)

The equity gap for edge `e` is defined as:

```
CEG(e) = KL(θ_{e,maj} || θ_{e,min})
```

This captures **both mean and concentration disagreement** — unlike simple Demographic Parity, two posteriors with identical means but different concentrations will produce a non-trivial CEG.

**Diagnostic decomposition** into orthogonal axes:

- `mean_kl`: KL between Betas re-parameterised to share average concentration `n_eff` — captures signal-driven gap
- `prec_kl`: KL between Betas re-parameterised to share average mean — captures precision-driven gap

### 7.2 Topological Void Detection (TVD)

For each subgroup `g`, a weighted graph is built with:

```
w(u, v) = 1 - E[θ_{e,g}]    # filtration distance
```

Persistence is computed via:
- **0-dimensional**: sublevel-set union-find
- **1-dimensional**: BFS-cycle detection

A **structural void** is a feature with:
- High majority persistence
- Jaccard-matched minority persistence < 50% of majority

**On DermaKG-Bench:** TVD identifies **55 voids**, most being 1-cycles among related drug classes losing connectivity under IV–VI filtration.

---

## 8. Inverse Graph Reasoning (IGR) Framework

IGR is a **4-stage candidate-generation framework** (not a ranking-novelty claim) that identifies KG-absent edges worth investigating for evidence acquisition.

### 8.1 Stage 1 — EquityGapDetector

Computes per-disease equity gap scores as a weighted blend of four components:

```
gap_score = w_repr × repr_gap
          + w_drug × drug_gap
          + w_structural × structural_gap
          + w_clinical × (1.0 - impact)
```

| Component | Definition |
|-----------|------------|
| `repr_gap` | `max(0.5 - prevalence_ivvi, 0) / 0.5` — representation deficit |
| `drug_gap` | `repr_gap` if `n_drugs == 0` else `0.0` — drug richness deficit |
| `structural_gap` | `1.0 - min(|2-hop drugs|/50, 1.0)` — KG neighborhood sparsity |
| `clinical_impact` | `min(fst_total/500, 1.0)` — prevalence-adjusted importance |

**v5.4 filter:** Gaps with `n_drugs == 0 AND fst_ivvi_count == 0` are removed as noise.

**Output:** 100–105 priority diseases (after filtering), severity-ranked.

### 8.2 Stage 2 — HypothesisGenerator

Generates three types of drug-disease candidates for each priority gap:

| Type | Description | Plausibility Base | Count |
|------|-------------|-------------------|-------|
| **A** — Scale-up | Existing PrimeKG indication, sparse FST IV–VI evidence | 1.0 (indication) / 0.6 (off-label) | 22–34 |
| **B** — Cross-disease borrow | Drug indicated for a MONDO-compatible donor disease with high IV–VI evidence; SapBERT cosine similarity ≥ floor | `sim × 0.7` | 9–34 |
| **C** — Class extension | Drug in same 4-character ATC class as an anchor drug indicated for the disease | `class_score` (fraction of class already indicated) | 53–62 |

**MONDO compatibility check** for Type B: blocks cross-domain borrows (e.g., oncology → pigmentary).

**Plausibility scoring:**
```
plausibility = w_evidence × base_evidence
             + w_semantic × semantic_sim
             + w_class × class_score
             + w_population × (1.0 - fst_prev)
```
Candidates with `plausibility < threshold` are filtered out.

**Equity gain scoring:**
```
equity_gain = w_repr_deficit × repr_deficit
            + w_evidence_gain × evidence_gain
            + w_pathway_gain × pathway_gain
```

### 8.3 Stage 3 — CostEstimator

Assigns a cost proxy (prevalence-adjusted difficulty × approval-stage multiplier):

```
cost = (1.0 / max(prevalence_ivvi, 0.01)) × approval_multiplier
```

| Approval Stage | Multiplier |
|---------------|------------|
| Approved | cfg.cost_approved (low) |
| Investigational | cfg.cost_investigational (medium) |
| Experimental | cfg.cost_experimental (high) |

Approval stage is inferred from the number of existing disease indications in the KG:
- `n_ind > 3` → approved
- `1 < n_ind ≤ 3` → investigational
- `n_ind ≤ 1` → experimental

### 8.4 Stage 4 — ParetoRanker

Returns the **non-dominated frontier** in `(equity_gain, cost)` space using multi-objective dominance.

Three ranked lists produced:

1. **Primary**: sorted by `equity_gain / max(cost, 0.01)` — top `cfg.top_n_primary`
2. **Quick wins**: sorted by `equity_gain / max(cost, 1.0)` — low-cost high-equity candidates
3. **Actionable now**: FDA-approved candidates only, sorted by equity gain

**On DermaKG-Bench:** 4 Pareto candidates, 9 Type-A quick wins (all Lyme disease antibiotics).

### 8.5 BED-IGR Scoring (Bayesian Experimental Design)

For each candidate, the expected information gain is:

```
EIG(e, g, n) = E_{S ~ BetaBin(n, α_{e,g}, β_{e,g})} [
    KL( Beta(α_{e,g}+S, β_{e,g}+n-S) || Beta(α_{e,g}, β_{e,g}) )
]
```

This closed-form Beta-Binomial EIG replaces the heuristic `equity_gain/cost` weighting of earlier versions. Within Type-A, BED-IGR and `equity_gain/cost` are near-perfectly correlated (median Spearman ρ = 1.00); the distinction matters for Types B and C where 211/367 candidates target KG-absent edges where CEG = 0 by construction.

### 8.6 Group-Conditional Conformal Filter

After candidate generation, a conformal filter is applied using `GroupConditionalConformal`:

- Per-group nonconformity thresholds: IV–VI → `0.6935`, I–III → `0.4728`, Global → `0.6913`
- Equalized worst-case threshold: `0.6935`
- Candidates with plausibility below the group-conditional threshold are filtered

**On DermaKG-Bench:** All 84–130 candidates pass the conformal filter (plausibility ≥ threshold).

---

## 9. ATC Safety Filter

A **domain-aware hard filter** that blocks clinically inappropriate candidates.

### 9.1 Domain Constraint Map

Seven dermatology clinical domains with ATC allow/block lists:

| Domain | Example Allow | Example Block |
|--------|--------------|---------------|
| `infectious_skin` | J01, J02, D01, D06 | N01, H02AB, D07, S01 |
| `neoplastic_skin` | L01, L04, L03, D06BB | S01, A11, D07, D10 |
| `inflammatory_skin` | D07, D11, L04, H02, R06 | S01 |
| `autoimmune_skin` | D07, D11, L04, H02, M01 | S01 |
| `acneiform` | D10, J01, G03C, D06BX | D07, H02, S01, J02AA |
| `pigmentary` | D11, D10AD, D07, L04, B02BA | L01, N01, S01 |
| `unknown` | (all except S01) | S01 |

### 9.2 Global Never-Recommend List

A curated set of drug names that should never be recommended regardless of domain (e.g., systemic chemotherapy for benign skin conditions, thalidomide for non-indicated uses).

**On DermaKG-Bench:** Filter rejects 96 of 463 candidates. Top reasons:
- `atc_not_allowlisted_for_neoplastic_skin`: 21 rejections
- `atc_blocked_D07_for_neoplastic_skin`: 15 rejections

---

## 10. Subgroup-Stratified Conformal Calibration (SSCC)

A **deployment-time wrapper** (not benchmarked in retrieval results) providing coverage guarantees for minority subgroups.

### 10.1 Problem

Standard split conformal prediction assumes exchangeability between calibration and test data. In DermaKG, calibration data is predominantly majority-FST (I–III), while test queries target minority-FST (IV–VI) — violating exchangeability.

### 10.2 Solution

Subgroup-stratified **weighted conformal calibration** following Tibshirani et al.:

1. Per-subgroup nonconformity scores are weighted by importance ratio:
   ```
   w_g(x) = p_{P_test,g}(x) / p_{P_calib}(x)
   ```
   estimated by logistic discrimination (Bickel et al. 2009)

2. Per-subgroup `(1-α)` weighted quantile gives the prediction-set threshold

3. A finite-sample distribution-free **permutation test** validates coverage

**Theorem 3:** Under correct specification of `w_g` and standard regularity, SSCC's per-subgroup prediction sets satisfy:

```
P_g(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 - α
```

---

## 11. Recommender System

The `Recommender` class is the **unified query interface** combining all pipeline components.

### 11.1 Query Flow

```
query string
    → EntityLinker (SapBERT + lexical fallback)
    → disease node ID + domain label (MONDO)
    → DDIndex.get_indications() + DDIndex.get_off_label()
    → FST subgraph selection (poplight / popdark based on user's FST)
    → LTR scoring (12-d feature vector → score)
    → MMR diversification (ATC-aware Maximal Marginal Relevance)
    → CGD confidence gating (conformal threshold check)
    → top-K ranked drug list with plausibility scores
```

### 11.2 Population-Stratified Scoring

For a query with FST group `g`:
- Drug edges in the `poplight` or `popdark` subgraph receive a **subgraph tier bonus**
- SCP-KG posterior mean `E[θ_{e,g}]` modifies the base LTR score
- Contested edges (high CEG) are flagged with a warning in the output

### 11.3 MMR Diversification (v5.4 fix)

ATC-aware **Maximal Marginal Relevance** re-ranker:

```python
# v5.4: pool is sorted by score DESC before anchoring (critical fix)
pool = sorted(candidates, key=lambda x: -x.get('score', 0))
first = pool.pop(0)  # anchor: highest-tier pick
picked = [first]

# MMR loop
while len(picked) < top_k and pool:
    best_i, best_score = 0, -inf
    for i, cand in enumerate(pool):
        rel = cand.get('score', 0)
        max_ov = max(atc_overlap(cand_atc, p) for p in picked_atcs)
        m = (1 - diversity_weight) * rel - diversity_weight * max_ov
        if m > best_score: best_score = m; best_i = i
    picked.append(pool.pop(best_i))
```

ATC overlap scoring:
- 4-char prefix match → 1.0
- 3-char prefix match → 0.6
- 2-char prefix match → 0.3
- No match → 0.0

---

## 12. Learn-to-Rank (LTR) Module

A **pairwise margin-ranking neural network** trained on drug-disease feature vectors.

### 12.1 Architecture

```
Input: 12-d feature vector
    → Linear(12, hidden_dim)
    → ReLU
    → Linear(hidden_dim, 1)
    → Sigmoid
Output: scalar score ∈ [0, 1]
```

### 12.2 Training Objective

Pairwise margin ranking loss:

```
L = mean( ReLU(margin - score_pos + score_neg) )
```

Training details:
- Optimizer: Adam
- Gradient clipping: max norm 1.0
- Negative sampling: `pairs_per_pos` negatives per positive per epoch
- **Note:** LTR training is currently skipped in v5.4 because current features are label-correlated stubs (pairwise accuracy trivially reaches 1.000). Re-enabled after GNN embeddings replace stub features.

---

## 13. Confidence-Gated Decoder (CGD)

The CGD applies the **group-conditional conformal threshold** as a hard gate on recommendations:

- If `plausibility(drug, disease, fst_group) ≥ threshold(fst_group)`:  
  → Include in output
- Otherwise: filtered

This ensures that predictions for minority-FST (IV–VI) patients meet at least the same coverage guarantee as majority-FST predictions (equalized worst-case threshold = 0.6935).

---

## 14. Living Epistemic Hypergraph (LEH) — Knowledge Decay Gradient (DKDG)

### 14.1 LEH Capsule Structure

Each LEH **capsule** = one (drug, disease) edge with:
- `global_conf`: posterior mean averaged across subgroups
- `density`: evidence density (subgroup pair count / expected)
- `fst_range`: `|E[θ_{IV–VI}] - E[θ_{I–III}]|`
- `zone`: core / inference / peripheral / contested / void

### 14.2 DermaKG Knowledge Decay Gradient (DKDG)

For each of the 315 skin conditions, the DKDG classifies the FST evidence decay pattern:

| Pattern | Count | % | Description |
|---------|-------|---|-------------|
| `cliff_drop` | 244 | 77.5% | Sudden drop in FST IV–VI evidence |
| `gradual_decay` | 47 | 14.9% | Gradual decrease in IV–VI representation |
| `uniform_void` | 1 | 0.3% | No evidence for any FST group |
| `mixed` | 23 | 7.3% | Inconsistent decay pattern |

The top 8 cliff-drop diseases by evidence count include **psoriasis** (FST I–VI counts: 113, 232, 142, 179, 93...).

---

## 15. v5.4 Aggregator Node Injection

Four key patches introduced in v5.4:

### Patch 1 — Aggregator Nodes
Synthetic disease nodes for PrimeKG coverage gaps injected with clinically curated indication/off-label edges (see §3.4).

### Patch 2 — MMR Sort Fix
Before v5.4, MMR's `first` element was in graph-neighbor iteration order (arbitrary). v5.4 sorts the pool by score DESC before picking the anchor, so tier bonuses actually drive the top-1 pick (e.g., acne → Tretinoin leads over Estrone).

### Patch 3 — Gap Detector Noise Filter
Gaps with both `n_drugs == 0` AND `fst_ivvi_count == 0` removed. These represent data artifacts (e.g., pseudolymphoma), not actionable equity gaps.

### Patch 4 — Unknown Domain Seeds
Added seeds for: `palmoplantar keratoderma`, `xerodera pigmentosum`, `keloid`, `hypertrophic scar`, `congenital heart disease ptosis hypodontia craniostosis` → domain `inflammatory_skin` / `neoplastic_skin`.

---

## 16. Pipeline Orchestration

### 16.1 `runpipeline()` / `runpipelinev2()` — 10 Stages

```
Stage 1  ─  Load datasets
             PrimeKG, Fitzpatrick17k, DermaCon-IN, OpenTargets, DrugCentral
             → skinstats (315), drugbankmap (7,957), clinicalGT (2,060)

Stage 2  ─  Build multi-source KG
             PrimeKG base → +OT +DrugCentral → FST annotation → pop subgraphs
             → globalgraph (90,067 nodes, 8,100,615 edges)

Stage 3  ─  Load MONDO ontology
             → mondo (26,709 terms, 129,101 labels)

Stage 4  ─  Setup EntityLinker (SapBERT)
             → linker (9,019 vocab entries encoded)

Stage 5  ─  Build DiseaseDrugIndex
             → ddindex (9,019 diseases, 825 with drugs, 122 FST-matched)

Stage 6  ─  Train pairwise LTR
             → ltr (skipped in v5.4 — stub features)

Stage 7  ─  Fit group-conditional conformal
             → conformal (IV–VI: 0.6935, I–III: 0.4728, Global: 0.6913)

Stage 8  ─  Run IGR
             → gaps (100), hypotheses (84 candidates: A22, B9, C53)
             → agenda (Pareto frontier + quick wins)

Stage 9  ─  Build Living Epistemic Hypergraph + DKDG
             → leh (20,757 capsules), dkdg (cliff:244, gradual:47)

Stage 10 ─  Demo queries + evaluation
             → Recommender initialized with all components

[v5.4 post-patch]
             → inject_aggregator_nodes() (+4 nodes, +48 edges)
             → rebuild DDIndex (9,023 diseases, 828 with drugs, 133 FST-matched)
             → refresh EntityLinker vocabulary (9,023 entries)
             → rebuild Recommender
             → re-run IGR (130 candidates: A34, B34, C62)
```

### 16.2 Results Dict Schema

```python
results = {
    "datasets": {
        "primekg": DataFrame,       # raw PrimeKG
        "fitzpatrick": DataFrame,   # Fitzpatrick17k
        "dermacon": DataFrame,      # DermaCon-IN
        "opentargets": DataFrame,
        "drugcentral": DataFrame,
        "drugbankmap": dict,
        "skinstats": dict,
        "clinical_ground_truth": dict,
        "clinical_oracle": dict,
    },
    "global_graph": igraph.Graph,
    "poplight": igraph.Graph,
    "popdark": igraph.Graph,
    "mondo": MondoOntology,
    "linker": EntityLinker,
    "ddindex": DiseaseDrugIndex,
    "conformal": GroupConditionalConformal,
    "recsystem": Recommender,
    "leh": LEHIndex,
    "dkdg": DKDGIndex,
    "igr": InverseGraphReasoner,
    "igr_agenda": Dict[str, List[HypothesisCandidate]],
}
```

---

## 17. Evaluation Harness & Benchmark

### 17.1 Benchmark Statistics

| Statistic | Value |
|-----------|-------|
| Diseases | 49 |
| Drugs | 166 |
| Indication edges | 419 |
| Contraindication edges (safety oracle) | 278 |
| CV folds | 5 (seeds: 42, 142, 242, 342, 442) |
| Train/test split | 80/20 |
| Test edges per fold | ~94 (50 I–III, 44 IV–VI) |

### 17.2 Evaluated Methods (12 total)

| Method | Family | H10 (mean ± 95% CI) |
|--------|--------|---------------------|
| Random | Trivial | 0.068 ± 0.021 |
| Popularity | Trivial | 0.215 ± 0.020 |
| Co-occurrence | Retrieval | 0.560 ± 0.014 |
| JS-divergence | Retrieval | 0.419 ± 0.019 |
| Network-proximity | Retrieval | 0.168 ± 0.020 |
| TransE | KGE | 0.074 ± 0.027 |
| DistMult | KGE | 0.064 ± 0.007 |
| ComplEx | KGE | 0.085 ± 0.020 |
| RotatE | KGE | 0.128 ± 0.009 |
| DermaKG-IGR | Diagnostic | 0.138 ± 0.013 |
| **DermaKG-Posterior** | **Posterior** | **0.570 ± 0.021** |
| Co-occ+DermaKG | Hybrid | **0.589 ± 0.021** |

### 17.3 Fairness Results (H10)

| Method | I–III H10 | IV–VI H10 | Gap | EOM |
|--------|-----------|-----------|-----|-----|
| Co-occurrence | 0.600 | 0.514 | 0.086 | 0.856 |
| **DermaKG-Posterior** | **0.572** | **0.568** | **0.004** | **0.993** |
| Co-occ+DermaKG | 0.644 | 0.527 | 0.117 | 0.819 |
| DermaKG-IGR | 0.140 | 0.136 | 0.004 | 0.974 |

**Headline result:** DermaKG-Posterior reduces the FST fairness gap by 21% (0.004 vs 0.086) at statistically tied accuracy (p = 0.069 vs Co-occurrence).

### 17.4 Evaluation Metrics

- **Hits@K** (K = 1, 5, 10): fraction of test edges where the held-out drug appears in top-K
- **MRR**: Mean Reciprocal Rank
- **NDCG@10**: Normalized Discounted Cumulative Gain at 10
- **Fairness gap**: |H10_{I–III} − H10_{IV–VI}|
- **Equality of Opportunity (EOM)**: H10_{IV–VI} / H10_{I–III}
- **Safety violation rate**: fraction of top-10 predictions that are PrimeKG-contraindicated

All metrics FST-stratified; significance via paired bootstrap (2,000 resamples, seed 42).

---

## 18. Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                        │
│  PrimeKG  ──┐                                                       │
│  Fitzpatrick ─┼──► skinstats ──► FST Evidence Records              │
│  DermaCon-IN ─┘                                                     │
│  OpenTargets ──► OT edges                                           │
│  DrugCentral ──► DC edges + never-list                              │
│  DrugBank ──────► drugbankmap                                       │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│                    KG CONSTRUCTION                                  │
│  igraph (90K nodes, 8.1M edges)                                     │
│  ├── poplight (FST I–III majority subgraph)                         │
│  └── popdark  (FST IV–VI majority subgraph)                         │
│  [v5.4] +4 aggregator nodes, +48 edges                             │
└──────┬─────────────────┬───────────────────────────────────────────┘
       │                 │
┌──────▼──────┐   ┌──────▼──────────────────────────────────────────┐
│   MONDO     │   │           SCP-KG + LEH                          │
│  Ontology   │   │  Beta posteriors per (edge, FST_group)           │
│ 26K terms   │   │  LEH: Beta(1.10, 0.50) EB prior                 │
│ domain map  │   │  20,757 capsules (core/infer/periph/contest/void)│
└──────┬──────┘   └───────┬─────────────────────────────────────────┘
       │                  │
┌──────▼──────────────────▼──────────────────────────────────────────┐
│                   ENTITY LINKER (SapBERT)                           │
│  9,023 vocab entries → dense embeddings                             │
│  + lexical fallback normalization                                   │
└──────┬──────────────────┬──────────────────────────────────────────┘
       │                  │
┌──────▼──────┐   ┌───────▼──────────────────────────────────────────┐
│  Disease-   │   │              IGR PIPELINE                         │
│  Drug Index │   │  1. EquityGapDetector → 100 priority diseases    │
│  9,023 dis. │   │  2. HypothesisGenerator → 130 candidates (A/B/C) │
│  828 w/drug │   │  3. CostEstimator → cost proxy                   │
└──────┬──────┘   │  4. ParetoRanker → 4 Pareto + 9 quick wins       │
       │          └───────────────────────────────────────────────────┘
┌──────▼──────────────────────────────────────────────────────────────┐
│                    RECOMMENDER                                       │
│  EntityLinker → DDIndex → LTR scoring → MMR diversification         │
│  → CGD conformal gate → top-K ranked drugs with plausibility        │
└──────┬──────────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────────┐
│               ATC SAFETY FILTER                                     │
│  Domain-aware hard filter (7 domains, 150-drug seed map)            │
│  Rejects ~20% of IGR candidates                                     │
└──────┬──────────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────────┐
│              SSCC (Deployment-time wrapper)                         │
│  Subgroup-stratified weighted conformal calibration                 │
│  Guarantees per-subgroup coverage ≥ 1 - α                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 19. Key Design Decisions & Trade-offs

### Why not GNNs (e.g., TxGNN)?

KGE methods (TransE, DistMult, ComplEx, RotatE) all score below H10 = 0.13 on DermaKG-Bench. This reflects the **data-sparse regime**: 325 training triples per fold compressed into 64-dimensional embeddings = ~5 triples/dimension, far below competitive thresholds. SCP-KG-Posterior's strength derives from preserving subgroup-conditional evidence directly in closed-form Beta posteriors — well-suited for small-N regimes. TxGNN requires the full 8.1M-edge PrimeKG at training time and cannot be FST-stratified without compromising comparability.

### Why FST labels are cohort-derived proxies

FST labels in DermaKG-Bench are cohort-derived (from Fitzpatrick17k and DermaCon-IN), not patient-level. They conflate dermatologic with geographic and epidemiologic factors. Top disparity-ranked diseases include Lyme borreliosis (cohort skew toward light-skin Northern-European) and syphilis (cohort skew toward IV–VI Indian cohort in DermaCon-IN). **Downstream users should treat per-disease disparity scores as hypotheses requiring cohort-level audit.**

### The Bimodal Disagreement Pattern

DermaKG-Posterior and Co-occurrence operate in complementary regimes:

- **DermaKG-Posterior wins** (35% of cases, n=97): Dramatic rank gains (mean +30.3 ranks) on **narrowly-indicated, evidence-rich** FDA-approved treatments that pure co-occurrence ranks last (e.g., vismodegib for BCC, selumetinib for NF1, talimogene laherparepvec for melanoma)
- **Co-occurrence wins** (60.6% of cases, n=168): Small margin (median 3.5 ranks) on **common-drug** edges propagating well via disease similarity

This explains the Co-occ+DermaKG hybrid's superior aggregate accuracy (H10 = 0.589).

### IGR Positioning

IGR is explicitly positioned as a **candidate-generation and evidence-acquisition framework**, not a retrieval-quality contribution. Within Type-A candidates, BED-IGR ranking collapses to `equity_gain/cost` (median Spearman ρ = 1.00, 83% with ρ ≥ 0.95). IGR's genuine novelty is **unified scoring across Types A+B+C**, where 211/367 post-safety candidates target KG-absent edges where CEG = 0 by construction and simpler heuristics fail.

---
