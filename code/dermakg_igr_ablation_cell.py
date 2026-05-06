#!/usr/bin/env python3
# =============================================================================
# IGR ABLATION CELL
# =============================================================================
# Run AFTER the main pipeline cell (dermakg_kaggle_all_in_one.py).
# Reuses primekg_df, skin_stats, OUTPUT_DIR from the main cell's namespace.
# Reads from disk: igr_all_candidates.csv, scp_all_posteriors.csv,
#                  ceg_top100.csv, scp_eb_prior.json.
#
# QUESTION: is BED-IGR's expected-information-gain ranking different from
# simpler heuristics that any reviewer would propose as the "obvious" baseline?
#
# Specifically, we test BED-IGR against five alternatives, all of which a
# reviewer might suggest replace it:
#
#   H1. CEG / cost                    — sort candidates by CEG ÷ cost
#   H2. CEG only                      — sort by CEG, ignore cost
#   H3. equity_gain / cost            — sort by directional equity ÷ cost
#   H4. uncertainty / cost            — sort by minority posterior variance
#                                       ÷ cost (variance reduction proxy)
#   H5. n_minority deficit / cost     — sort by (n_eff_iii − n_eff_ivvi) / cost
#
# If BED-IGR's ranking strongly correlates (Spearman ρ > 0.95) with ALL of
# these, IGR's novelty as a Bayesian experimental design framework is hard
# to defend. If correlations are moderate (ρ in [0.3, 0.85]), IGR captures
# something distinct from each heuristic — which is the result we need.
#
# Spearman is the right metric: we care about RANKING agreement, not raw
# score agreement. A small constant-multiple difference in EIG values would
# preserve ranking but Pearson would miss it.
#
# Outputs:
#   ablation_igr_correlations.csv     — per-disease correlations + summary
#   ablation_igr_top10_overlap.csv    — top-10 set overlap (Jaccard) per heuristic
#   ablation_igr_disagreement_cases.csv — cases where BED-IGR ranks differently
# =============================================================================

import os, json, math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from dermakg_kaggle_all_in_one import OUTPUT_DIR, primekg_df, skin_stats

_REQUIRED = ["OUTPUT_DIR"]
_missing = [n for n in _REQUIRED if n not in dir()]
if _missing:
    raise RuntimeError(
        f"IGR ablation expects main pipeline cell to have run. Missing: {_missing}"
    )

OUT = OUTPUT_DIR

# -----------------------------------------------------------------------------
# §1  LOAD ARTIFACTS
# -----------------------------------------------------------------------------

print("=" * 78)
print("§1  IGR ABLATION — BED-IGR vs heuristic ranking strategies")
print("=" * 78)

_cand = pd.read_csv(os.path.join(OUT, "igr_all_candidates.csv"))
_post = pd.read_csv(os.path.join(OUT, "scp_all_posteriors.csv"))
with open(os.path.join(OUT, "scp_eb_prior.json")) as f:
    _prior = json.load(f)

print(f"  IGR candidates loaded   : {len(_cand)}")
print(f"  SCP posteriors loaded   : {len(_post)}")
print(f"  EB prior                : Beta({_prior['alpha0']:.3f}, "
      f"{_prior['beta0']:.3f})")

def _norm(s): return str(s).lower().strip()

# Build (disease, drug) -> posterior parameters
post_lookup: Dict[Tuple[str, str], Dict] = {}
for _, r in _post.iterrows():
    key = (_norm(r["disease"]), _norm(r["drug"]))
    post_lookup[key] = {
        "alpha_iii": float(r["alpha_iii"]), "beta_iii": float(r["beta_iii"]),
        "post_mean_iii": float(r["post_mean_iii"]),
        "n_eff_iii": float(r["n_eff_iii"]),
        "alpha_ivvi": float(r["alpha_ivvi"]),
        "beta_ivvi": float(r["beta_ivvi"]),
        "post_mean_ivvi": float(r["post_mean_ivvi"]),
        "n_eff_ivvi": float(r["n_eff_ivvi"]),
        "ceg": float(r["ceg"]),
    }

# -----------------------------------------------------------------------------
# §2  COMPUTE HEURISTIC SCORES FOR EACH IGR CANDIDATE
# -----------------------------------------------------------------------------
# For each IGR candidate, we compute six scores: BED-IGR (as already produced
# by the main pipeline) plus five heuristic alternatives. Each score gets
# inverted as needed so HIGHER = BETTER (more worth investigating).

print("\n" + "=" * 78)
print("§2  COMPUTING HEURISTIC SCORES")
print("=" * 78)

def _beta_var(a: float, b: float) -> float:
    """Variance of Beta(a, b) — proxy for minority-side uncertainty."""
    s = a + b
    return (a * b) / (s * s * (s + 1.0)) if s > 0 else 0.0

ablation_rows = []
n_skipped = 0

for _, r in _cand.iterrows():
    d = _norm(r["disease"])
    dr = _norm(r["drug"])
    cost = float(r["cost"])
    bed_eig = float(r["expected_information_gain"])
    equity_gain = float(r.get("equity_gain", 0.0))

    post = post_lookup.get((d, dr))
    if post is None:
        # Type B/C candidates may target edges not in the original SCP-KG;
        # their posterior is implicitly the prior. Use the prior here.
        post = {
            "alpha_iii": _prior["alpha0"], "beta_iii": _prior["beta0"],
            "post_mean_iii": _prior["alpha0"] / (_prior["alpha0"] + _prior["beta0"]),
            "n_eff_iii": _prior["alpha0"] + _prior["beta0"],
            "alpha_ivvi": _prior["alpha0"], "beta_ivvi": _prior["beta0"],
            "post_mean_ivvi": _prior["alpha0"] / (_prior["alpha0"] + _prior["beta0"]),
            "n_eff_ivvi": _prior["alpha0"] + _prior["beta0"],
            "ceg": 0.0,
        }

    ceg = post["ceg"]
    minority_var = _beta_var(post["alpha_ivvi"], post["beta_ivvi"])
    n_deficit = max(0.0, post["n_eff_iii"] - post["n_eff_ivvi"])

    ablation_rows.append({
        "disease": r["disease"],
        "drug": r["drug"],
        "type": r["type"],
        "cost": cost,
        # The thing we're testing
        "score_bed_igr": bed_eig,                            # BED-IGR EIG
        # Five heuristics a reviewer might propose
        "score_h1_ceg_per_cost": ceg / cost,                 # H1
        "score_h2_ceg_only":     ceg,                        # H2
        "score_h3_equity_per_cost": equity_gain / cost,      # H3
        "score_h4_var_per_cost": minority_var / cost,        # H4
        "score_h5_ndef_per_cost": n_deficit / cost,          # H5
    })
    if post.get("ceg", 0.0) == 0.0 and ceg == 0.0:
        n_skipped += 1

ab_df = pd.DataFrame(ablation_rows)
print(f"  scored {len(ab_df)} candidates "
      f"({n_skipped} fell back to prior — Type B/C)")

# -----------------------------------------------------------------------------
# §3  GLOBAL CORRELATION BETWEEN BED-IGR AND EACH HEURISTIC
# -----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("§3  GLOBAL RANK CORRELATION (Spearman ρ + Kendall τ)")
print("=" * 78)
print("  H1: CEG / cost      H2: CEG only        H3: equity_gain / cost")
print("  H4: minority Var / cost                  H5: n-deficit / cost\n")

heuristic_cols = [
    ("H1: CEG / cost",     "score_h1_ceg_per_cost"),
    ("H2: CEG only",       "score_h2_ceg_only"),
    ("H3: equity / cost",  "score_h3_equity_per_cost"),
    ("H4: Var / cost",     "score_h4_var_per_cost"),
    ("H5: ndef / cost",    "score_h5_ndef_per_cost"),
]

global_corr_rows = []
print(f"  {'heuristic':<25} {'spearman ρ':>12} {'kendall τ':>12} "
      f"{'top-10 Jaccard':>16}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*16}")
for label, col in heuristic_cols:
    rho, p_rho = spearmanr(ab_df["score_bed_igr"], ab_df[col])
    tau, p_tau = kendalltau(ab_df["score_bed_igr"], ab_df[col])
    # Top-10 set overlap (Jaccard)
    bed_top10 = set(ab_df.nlargest(10, "score_bed_igr").index)
    h_top10 = set(ab_df.nlargest(10, col).index)
    jaccard = (len(bed_top10 & h_top10) /
               len(bed_top10 | h_top10) if (bed_top10 | h_top10) else 0.0)
    print(f"  {label:<25} {rho:>+12.3f} {tau:>+12.3f} {jaccard:>16.2f}")
    global_corr_rows.append(dict(
        heuristic=label, spearman=rho, p_spearman=p_rho,
        kendall=tau, p_kendall=p_tau, top10_jaccard=jaccard,
    ))

corr_df = pd.DataFrame(global_corr_rows)
corr_df.to_csv(os.path.join(OUT, "ablation_igr_correlations.csv"), index=False)
print(f"\n  → {OUT}/ablation_igr_correlations.csv")

# -----------------------------------------------------------------------------
# §4  PER-DISEASE CORRELATION (more rigorous: control for disease-level
#     EIG variance, since BED-IGR scaling depends on n_per_trial)
# -----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("§4  PER-DISEASE CORRELATION DISTRIBUTION")
print("=" * 78)
print("  reports median ± IQR of within-disease Spearman ρ across diseases")
print("  with ≥3 candidates, controlling for disease-level scale effects.\n")

per_disease_rows = []
for label, col in heuristic_cols:
    rhos = []
    for disease, grp in ab_df.groupby("disease"):
        if len(grp) < 3:
            continue
        if grp["score_bed_igr"].std() < 1e-9 or grp[col].std() < 1e-9:
            continue
        rho, _ = spearmanr(grp["score_bed_igr"], grp[col])
        if not math.isnan(rho):
            rhos.append(rho)
    if rhos:
        rhos = np.array(rhos)
        per_disease_rows.append(dict(
            heuristic=label,
            n_diseases=len(rhos),
            median_spearman=float(np.median(rhos)),
            q25=float(np.quantile(rhos, 0.25)),
            q75=float(np.quantile(rhos, 0.75)),
            min_spearman=float(rhos.min()),
            max_spearman=float(rhos.max()),
            frac_strong=float((np.abs(rhos) >= 0.95).mean()),
        ))

if per_disease_rows:
    print(f"  {'heuristic':<25} {'n diseases':>11} {'median ρ':>11} "
          f"{'IQR':>16} {'frac |ρ|≥.95':>14}")
    print(f"  {'-'*25} {'-'*11} {'-'*11} {'-'*16} {'-'*14}")
    for row in per_disease_rows:
        iqr_str = f"[{row['q25']:+.2f}, {row['q75']:+.2f}]"
        print(f"  {row['heuristic']:<25} {row['n_diseases']:>11d} "
              f"{row['median_spearman']:>+11.3f} {iqr_str:>16} "
              f"{row['frac_strong']:>14.2%}")
    pd.DataFrame(per_disease_rows).to_csv(
        os.path.join(OUT, "ablation_igr_per_disease_correlations.csv"),
        index=False)
    print(f"\n  → {OUT}/ablation_igr_per_disease_correlations.csv")
else:
    print("  ⚠ No diseases have ≥3 candidates — per-disease analysis skipped.")

# -----------------------------------------------------------------------------
# §5  TOP-K AGREEMENT — does BED-IGR's top-K differ from heuristics' top-K?
# -----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("§5  TOP-K SET OVERLAP — does BED-IGR pick the same headlines?")
print("=" * 78)

overlap_rows = []
print(f"  {'heuristic':<25} {'top-5':>8} {'top-10':>8} {'top-20':>8} "
      f"{'top-50':>8}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for label, col in heuristic_cols:
    row = {"heuristic": label}
    for k in (5, 10, 20, 50):
        if k > len(ab_df):
            row[f"top{k}"] = "—"
            continue
        bed_top = set(ab_df.nlargest(k, "score_bed_igr").index)
        h_top = set(ab_df.nlargest(k, col).index)
        jaccard = len(bed_top & h_top) / len(bed_top | h_top)
        row[f"top{k}"] = f"{jaccard:.2f}"
    overlap_rows.append(row)
    print(f"  {label:<25} {row['top5']:>8} {row['top10']:>8} "
          f"{row['top20']:>8} {row['top50']:>8}")

pd.DataFrame(overlap_rows).to_csv(
    os.path.join(OUT, "ablation_igr_top_k_overlap.csv"), index=False)
print(f"\n  → {OUT}/ablation_igr_top_k_overlap.csv")

# -----------------------------------------------------------------------------
# §6  DISAGREEMENT CASES — examples where BED-IGR ranks differently
# -----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("§6  DISAGREEMENT CASES — BED-IGR vs CEG/cost (the strongest heuristic)")
print("=" * 78)

# Compute rank-difference between BED-IGR and the most threatening heuristic
ab_df["rank_bed"] = ab_df["score_bed_igr"].rank(ascending=False, method="min")
ab_df["rank_h1"]  = ab_df["score_h1_ceg_per_cost"].rank(ascending=False, method="min")
ab_df["rank_diff"] = (ab_df["rank_bed"] - ab_df["rank_h1"]).abs()

disagree = ab_df.nlargest(15, "rank_diff")[
    ["disease", "drug", "type", "cost",
     "score_bed_igr", "score_h1_ceg_per_cost",
     "rank_bed", "rank_h1", "rank_diff"]
].copy()

disagree.columns = ["disease", "drug", "type", "cost",
                    "BED-IGR", "CEG/cost",
                    "rank_BED", "rank_H1", "|Δrank|"]

print("\n  Top 15 candidates where BED-IGR and CEG/cost disagree most "
      "(absolute rank diff):\n")
with pd.option_context("display.max_colwidth", 30, "display.width", 200):
    print(disagree.to_string(index=False))

disagree.to_csv(os.path.join(OUT, "ablation_igr_disagreement_cases.csv"),
                index=False)
print(f"\n  → {OUT}/ablation_igr_disagreement_cases.csv")

# -----------------------------------------------------------------------------
# §7  PAPER GUIDANCE — how to interpret + report
# -----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("§7  HOW TO REPORT THIS ABLATION IN THE PAPER")
print("=" * 78)

# Compute headline interpretation
median_rhos = {row["heuristic"]: row["median_spearman"]
               for row in per_disease_rows} if per_disease_rows else {}
strongest_rho = max((row["spearman"] for row in global_corr_rows),
                    key=abs, default=0.0)

if abs(strongest_rho) > 0.95:
    print(f"""
WARNING: Strongest heuristic correlation is ρ = {strongest_rho:+.3f} (|ρ| > 0.95).
BED-IGR's ranking is NEARLY EQUIVALENT to a simple heuristic. This is a
problem for the paper — a reviewer will say BED-IGR's mathematical
machinery (closed-form Beta-Binomial EIG) doesn't add ranking value
beyond CEG/cost. Options:

  (a) Reframe BED-IGR as providing INTERPRETABILITY (closed-form EIG with
      mean/uncertainty decomposition) rather than ranking novelty.
  (b) Move IGR into the appendix as a qualitative case-study tool only.
  (c) Add a clinical-oracle evaluation where BED-IGR's interpretability
      matters, even if its ranking matches simpler methods.
""")
elif abs(strongest_rho) > 0.85:
    print(f"""
MIXED RESULT: Strongest heuristic correlation is ρ = {strongest_rho:+.3f}
(0.85 < |ρ| ≤ 0.95). BED-IGR is HIGHLY but not perfectly correlated with
heuristic ranking. You can defend BED-IGR by reporting the ~5–15% of
candidates where rankings disagree (§6) and arguing that BED-IGR's
disagreements are clinically more sensible. This requires qualitative
analysis of the disagreement cases.

Recommended framing in §4.3 of the paper:
  'BED-IGR's closed-form expected information gain agrees with simple
   heuristics on most candidates (ρ = {strongest_rho:+.2f}) but disagrees
   on N% of cases (Appendix X). On the disagreement set, BED-IGR's
   ranking better reflects [property].'
""")
else:
    print(f"""
DEFENSIBLE: Strongest heuristic correlation is ρ = {strongest_rho:+.3f}
(|ρ| ≤ 0.85). BED-IGR captures ranking signal that NONE of the proposed
heuristics fully recover. This is a clean ablation result.

Recommended framing in §4.3 of the paper:
  'BED-IGR ranks candidates by closed-form Beta-Binomial expected
   information gain divided by acquisition cost. We ablate it against
   five heuristics a reviewer might propose (Table X). The strongest
   correlation (ρ = {strongest_rho:+.2f} with [heuristic]) indicates
   BED-IGR captures distinct ranking signal: information gain depends
   on posterior shape (mean and uncertainty disagreement separately),
   not just CEG magnitude.'

Citation suggestion for BED-IGR's information-gain decomposition:
  Lindley (1956), MacKay (1992), Houlsby et al. (2011) on Bayesian
  active learning by disagreement.
""")

print(f"""
KEY NUMBERS FOR PAPER:
  • Global Spearman ρ (BED-IGR vs each heuristic):
{chr(10).join(f"      {r['heuristic']:<25} ρ = {r['spearman']:+.3f}, τ = {r['kendall']:+.3f}, top-10 Jaccard = {r['top10_jaccard']:.2f}" for r in global_corr_rows)}
  • Per-disease median Spearman across diseases with ≥3 candidates:
{chr(10).join(f"      {r['heuristic']:<25} median ρ = {r['median_spearman']:+.3f} (IQR [{r['q25']:+.2f}, {r['q75']:+.2f}], n={r['n_diseases']} diseases)" for r in per_disease_rows) if per_disease_rows else "      [not computed — too few candidates per disease]"}

PUT THIS IN: §4.3 (BED-IGR description), Appendix B (ablation table),
            §6.3 (limitations — disclose if ρ > 0.85)
""")