#!/usr/bin/env python3
# =============================================================================
# DermaKG-Bench — DISAGREEMENT TABLE GENERATOR
# =============================================================================
# Run AFTER the main pipeline cell AND the comparison cell.
# Reuses primekg_df, skin_stats, OUTPUT_DIR, TARGET_SUBGROUP from main cell's
# namespace.
#
# PURPOSE: Replace the fabricated Table 4 in Appendix G of the paper with
# real per-edge rank comparisons between DermaKG-Posterior and Co-occurrence,
# computed across all 5 CV folds.
#
# OUTPUT:
#   /kaggle/working/dermakg_results/
#     paper_table_disagreement.csv     — full disagreement table (all edges
#                                         where |rank gain| ≥ 3)
#     paper_table4_top10.csv           — top-10 most-informative cases for
#                                         the paper's Appendix G Table 4
#     paper_table4_summary.txt         — paper-ready text snippet
#
# METHODOLOGY (defensive against reviewer scrutiny):
#   For each test edge (disease, held-out drug, FST group) across all folds:
#     1. Compute Co-occurrence's rank of the held-out drug
#     2. Compute DermaKG-Posterior's rank of the held-out drug
#     3. Compute rank_gain = co_rank − dermakg_rank
#        (positive = DermaKG-Posterior ranks the truth higher)
#   Aggregate across folds: report MEAN rank gain per (disease, drug) pair
#     where the same edge appears in multiple folds, OR include each
#     (disease, drug, fold) tuple separately and select most-extreme.
#   Filter: keep edges where mean |rank gain| ≥ 3.
#   Top-10 selection: top 10 by mean rank_gain (positive only, since we
#     want cases where DermaKG-Posterior wins — those are most informative
#     for clinician review per §6.2 of the paper).
# =============================================================================

import os, json, math, random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dermakg_kaggle_all_in_one import OUTPUT_DIR, primekg_df, skin_stats, TARGET_SUBGROUP

_REQUIRED = ["primekg_df", "skin_stats", "OUTPUT_DIR", "TARGET_SUBGROUP"]
_missing = [n for n in _REQUIRED if n not in dir()]
if _missing:
    raise RuntimeError(
        f"Disagreement table cell expects main pipeline cell to have run. "
        f"Missing: {_missing}"
    )

OUT = OUTPUT_DIR
SEED = 42
N_FOLDS = 5
MIN_RANK_GAIN = 3                  # threshold for "interesting" disagreement
TOP_N_FOR_TABLE = 10               # how many to print in paper's Table 4

print("=" * 78)
print("DISAGREEMENT TABLE GENERATOR — DermaKG-Posterior vs Co-occurrence")
print("=" * 78)

# -----------------------------------------------------------------------------
# §1  Load required artifacts
# -----------------------------------------------------------------------------

_post_csv = os.path.join(OUT, "scp_all_posteriors.csv")
_prior_json = os.path.join(OUT, "scp_eb_prior.json")

if not os.path.exists(_post_csv):
    raise RuntimeError(
        f"Cannot find {_post_csv}. The main pipeline cell must have produced "
        f"this file (post-patch). Re-run the main pipeline first."
    )

_post_df = pd.read_csv(_post_csv)

if os.path.exists(_prior_json):
    with open(_prior_json) as f:
        _eb_prior = json.load(f)
else:
    _eb_prior = {"alpha0": 1.0, "beta0": 1.0}

print(f"  loaded {len(_post_df)} SCP posteriors")
print(f"  EB prior: Beta({_eb_prior['alpha0']:.3f}, {_eb_prior['beta0']:.3f})")

# -----------------------------------------------------------------------------
# §2  Reconstruct ground truth and FST stratification (must match main eval)
# -----------------------------------------------------------------------------

def _norm(s): return str(s).lower().strip()

gt_indications: Dict[str, set] = defaultdict(set)
disease_fst: Dict[str, str] = {}

for name, val in skin_stats.items():
    if not isinstance(val, dict):
        continue
    n_iii = float(val.get("fst_i_iii", 0))
    n_ivvi = float(val.get("fst_iv_vi", 0))
    if n_iii + n_ivvi == 0:
        continue
    disease_fst[_norm(name)] = "IV-VI" if n_ivvi >= n_iii else "I-III"

df_pkg = primekg_df.rename(columns={c: str(c).lower().strip()
                                    for c in primekg_df.columns})
for row in df_pkg.itertuples(index=False):
    rel = _norm(getattr(row, "relation", ""))
    if rel != "indication":
        continue
    x_t = _norm(getattr(row, "x_type", ""))
    y_t = _norm(getattr(row, "y_type", ""))
    if x_t == "disease" and y_t == "drug":
        d, dr = _norm(row.x_name), _norm(row.y_name)
    elif y_t == "disease" and x_t == "drug":
        d, dr = _norm(row.y_name), _norm(row.x_name)
    else:
        continue
    if d in disease_fst:
        gt_indications[d].add(dr)

ALL_DRUGS = sorted({dr for drs in gt_indications.values() for dr in drs})
print(f"  ground truth: {len(gt_indications)} diseases, "
      f"{sum(len(v) for v in gt_indications.values())} indication edges, "
      f"{len(ALL_DRUGS)} drugs")

# -----------------------------------------------------------------------------
# §3  Build the two rankers exactly as in the comparison cell
# -----------------------------------------------------------------------------

def make_cooccurrence_ranker(train_indications):
    drug_set = {d: drs for d, drs in train_indications.items()}
    pop = Counter()
    for drs in drug_set.values():
        for dr in drs:
            pop[dr] += 1
    pop_fallback = [d for d, _ in sorted(pop.items(), key=lambda x: -x[1])]
    pop_fallback += [dr for dr in ALL_DRUGS if dr not in pop]

    def ranker(disease):
        target = drug_set.get(disease, set())
        if not target:
            return list(pop_fallback)
        sims = []
        for other_d, other_drs in drug_set.items():
            if other_d == disease or not other_drs:
                continue
            j = len(target & other_drs) / max(len(target | other_drs), 1)
            if j > 0:
                sims.append((other_d, j))
        sims.sort(key=lambda x: -x[1])
        score = Counter()
        for other_d, j in sims[:20]:
            for dr in drug_set[other_d]:
                if dr not in target:
                    score[dr] += j
        ranked = [d for d, _ in sorted(score.items(), key=lambda x: -x[1])]
        seen = set(ranked)
        ranked += [dr for dr in pop_fallback if dr not in seen]
        return ranked
    return ranker

def make_dermakg_posterior_ranker(train_indications):
    target = TARGET_SUBGROUP
    mean_col = "post_mean_ivvi" if target == "IV-VI" else "post_mean_iii"

    lookup: Dict[Tuple[str, str], float] = {}
    for _, row in _post_df.iterrows():
        d = _norm(row["disease"])
        dr = _norm(row["drug"])
        rel = _norm(row.get("relation", ""))
        if rel not in ("indication", "off-label use", "indicated_for"):
            continue
        lookup[(d, dr)] = float(row[mean_col])

    prior_mean = _eb_prior["alpha0"] / (_eb_prior["alpha0"] + _eb_prior["beta0"])

    pop = Counter()
    for drs in train_indications.values():
        for dr in drs:
            pop[dr] += 1

    def ranker(disease):
        scores = []
        for dr in ALL_DRUGS:
            m = lookup.get((disease, dr), prior_mean)
            score = m + 1e-3 * pop.get(dr, 0)   # popularity tie-breaker
            scores.append((dr, score))
        return [d for d, _ in sorted(scores, key=lambda x: -x[1])]
    return ranker

# -----------------------------------------------------------------------------
# §4  Replay all 5 CV folds and collect per-edge ranks for both methods
# -----------------------------------------------------------------------------

def make_fold_split(fold_seed):
    rng = random.Random(fold_seed)
    test_edges = []
    train_indications = defaultdict(set)
    for d, drugs in gt_indications.items():
        drugs_list = sorted(drugs)
        rng.shuffle(drugs_list)
        n_test = max(1, int(0.2 * len(drugs_list)))
        test_drugs = set(drugs_list[:n_test])
        for dr in test_drugs:
            test_edges.append((d, dr, disease_fst[d]))
        train_indications[d] = drugs - test_drugs
    return dict(train_indications), test_edges

# {(disease, drug): [(fold_idx, fst_group, co_rank, derm_rank), ...]}
edge_records: Dict[Tuple[str, str], List] = defaultdict(list)

print("\n" + "=" * 78)
print(f"§4  REPLAYING {N_FOLDS} FOLDS")
print("=" * 78)

for fold_idx in range(N_FOLDS):
    fold_seed = SEED + fold_idx * 100
    train_indications, test_edges = make_fold_split(fold_seed)
    co_ranker = make_cooccurrence_ranker(train_indications)
    derm_ranker = make_dermakg_posterior_ranker(train_indications)

    co_cache = {}
    derm_cache = {}

    for disease, drug, group in test_edges:
        if disease not in co_cache:
            co_cache[disease] = co_ranker(disease)
        if disease not in derm_cache:
            derm_cache[disease] = derm_ranker(disease)
        co_ranked = co_cache[disease]
        derm_ranked = derm_cache[disease]

        try:
            co_rank = co_ranked.index(drug) + 1
        except ValueError:
            co_rank = len(co_ranked) + 1
        try:
            derm_rank = derm_ranked.index(drug) + 1
        except ValueError:
            derm_rank = len(derm_ranked) + 1

        edge_records[(disease, drug)].append(
            (fold_idx, group, co_rank, derm_rank))

    print(f"  fold {fold_idx+1}: scored {len(test_edges)} edges")

print(f"\n  unique (disease, drug) edges across folds: {len(edge_records)}")

# -----------------------------------------------------------------------------
# §5  Aggregate per (disease, drug): mean ranks across folds
# -----------------------------------------------------------------------------

agg_rows = []
for (disease, drug), records in edge_records.items():
    co_ranks = [r[2] for r in records]
    derm_ranks = [r[3] for r in records]
    groups = [r[1] for r in records]
    n_folds_appeared = len(records)

    co_mean = float(np.mean(co_ranks))
    derm_mean = float(np.mean(derm_ranks))
    rank_gain = co_mean - derm_mean
    fst_group = groups[0]   # all folds same FST stratum since deterministic
    agg_rows.append(dict(
        disease=disease,
        drug=drug,
        fst_stratum=fst_group,
        n_folds_appeared=n_folds_appeared,
        co_rank_mean=round(co_mean, 1),
        dermakg_rank_mean=round(derm_mean, 1),
        rank_gain=round(rank_gain, 1),
        co_rank_per_fold=";".join(str(r) for r in co_ranks),
        dermakg_rank_per_fold=";".join(str(r) for r in derm_ranks),
    ))

agg_df = pd.DataFrame(agg_rows)
agg_df = agg_df.sort_values("rank_gain", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------------------------
# §6  Filter and write outputs
# -----------------------------------------------------------------------------

interesting = agg_df[agg_df["rank_gain"].abs() >= MIN_RANK_GAIN].copy()
interesting.to_csv(os.path.join(OUT, "paper_table_disagreement.csv"),
                   index=False)
print(f"\n  → {OUT}/paper_table_disagreement.csv ({len(interesting)} edges "
      f"with |rank gain| ≥ {MIN_RANK_GAIN})")

# Top-N for the paper's Appendix G Table 4
positive = agg_df[agg_df["rank_gain"] > 0].copy()
top_n = positive.head(TOP_N_FOR_TABLE).copy()
top_n_paper = top_n[["disease", "drug", "fst_stratum",
                     "co_rank_mean", "dermakg_rank_mean", "rank_gain"]].copy()
top_n_paper.to_csv(os.path.join(OUT, "paper_table4_top10.csv"), index=False)
print(f"  → {OUT}/paper_table4_top10.csv (paper's Table 4, top-{TOP_N_FOR_TABLE})")

# -----------------------------------------------------------------------------
# §7  Print Table 4 for the paper + summary text
# -----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("§7  TABLE 4 FOR APPENDIX G OF PAPER")
print("=" * 78)
print("\nThe top-10 cases where DermaKG-Posterior ranks the held-out indication")
print("substantially higher than Co-occurrence does (averaged across 5 CV folds):\n")

if len(top_n_paper) == 0:
    print("  ⚠ No cases with positive rank_gain ≥ 3. This means DermaKG-Posterior")
    print("    and Co-occurrence agree closely on all test edges. Reconsider")
    print("    whether the disagreement table belongs in the paper at all.")
else:
    print(top_n_paper.to_string(index=False))

# Distribution summary
n_dermakg_wins = (agg_df["rank_gain"] > 0).sum()
n_coocc_wins = (agg_df["rank_gain"] < 0).sum()
n_ties = (agg_df["rank_gain"] == 0).sum()
n_strong_dermakg = (agg_df["rank_gain"] >= MIN_RANK_GAIN).sum()
n_strong_coocc = (agg_df["rank_gain"] <= -MIN_RANK_GAIN).sum()

summary_text = f"""
DISAGREEMENT TABLE — SUMMARY STATISTICS (for paper §6.2 / Appendix G):

  Total unique (disease, drug) edges across {N_FOLDS} folds: {len(agg_df)}
  DermaKG-Posterior ranks higher: {n_dermakg_wins} edges ({100*n_dermakg_wins/len(agg_df):.1f}%)
  Co-occurrence ranks higher:     {n_coocc_wins} edges ({100*n_coocc_wins/len(agg_df):.1f}%)
  Tied:                            {n_ties} edges ({100*n_ties/len(agg_df):.1f}%)

  Strong disagreements (|rank gain| ≥ {MIN_RANK_GAIN}):
    DermaKG-Posterior wins: {n_strong_dermakg}
    Co-occurrence wins:     {n_strong_coocc}

  Mean rank gain (DermaKG-Posterior − Co-occurrence): {agg_df['rank_gain'].mean():+.2f}
  Median rank gain:                                    {agg_df['rank_gain'].median():+.2f}

For paper Appendix G Table 4: see paper_table4_top10.csv
For full disagreement edges (clinician adjudication subset):
                              see paper_table_disagreement.csv
"""

print(summary_text)
with open(os.path.join(OUT, "paper_table4_summary.txt"), "w") as f:
    f.write(summary_text)

# LaTeX for direct paper insertion
latex_lines = [
    "% Top-10 disagreement cases — auto-generated from real CV runs",
    "% Replace fabricated Table 4 in Appendix G with this.",
    "\\begin{table}[h]",
    "\\centering",
    "\\caption{Disagreement cases between DermaKG-Posterior and Co-occurrence: "
    f"top-{TOP_N_FOR_TABLE} test edges where DermaKG-Posterior ranks the "
    "held-out indication substantially higher than Co-occurrence. Per-disease "
    "majority FST in parentheses. Rank gain = Co-occurrence rank "
    "$-$ DermaKG-Posterior rank; positive values indicate DermaKG-Posterior "
    "is more accurate on that edge. Values aggregated across all 5 folds.}",
    "\\label{tab:disagreement}",
    "\\small",
    "\\begin{tabular}{llrrr}",
    "\\toprule",
    "Disease (FST stratum) & Drug & Co-occ rank & DermaKG-P rank & Rank gain \\\\",
    "\\midrule",
]
for _, r in top_n_paper.iterrows():
    latex_lines.append(
        f"{r['disease'].title()} ({r['fst_stratum']}) & "
        f"{r['drug']} & "
        f"{r['co_rank_mean']:.1f} & "
        f"{r['dermakg_rank_mean']:.1f} & "
        f"+{r['rank_gain']:.1f} \\\\"
    )
latex_lines.extend([
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}",
])

latex_path = os.path.join(OUT, "paper_table4.tex")
with open(latex_path, "w") as f:
    f.write("\n".join(latex_lines))
print(f"  → {latex_path} (paste into Appendix G to replace the fabricated table)")

print("\n" + "=" * 78)
print("DONE. Use paper_table4_top10.csv as the data source for Appendix G")
print("Table 4. Values are real, computed across all 5 CV folds.")
print("=" * 78)