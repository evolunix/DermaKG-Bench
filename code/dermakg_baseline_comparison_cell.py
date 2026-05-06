#!/usr/bin/env python3
# =============================================================================
# DermaKG-Causal — BASELINE COMPARISON CELL (NeurIPS-grade revision)
# =============================================================================
# Run AFTER the main pipeline cell (dermakg_kaggle_all_in_one.py).
# Reuses primekg_df, skin_stats, is_safe_recommendation, atc_class_prefix,
# OUTPUT_DIR, TARGET_SUBGROUP from the main cell's namespace.
#
# Reads from disk (written by the main cell):
#   - igr_all_candidates.csv     : IGR-flagged (disease, drug, EIG) candidates
#   - scp_all_posteriors.csv     : full SCP-KG posteriors per edge per subgroup
#   - scp_eb_prior.json          : Empirical-Bayes prior for unseen pairs
#
# REVISION CHANGES (from previous version):
#   - 5-fold cross-validation with mean ± 95% CI per metric
#   - DermaKG-Posterior: stand-alone ranker over ALL drugs using SCP-KG
#     posterior mean (apples-to-apples with TransE / Co-occurrence).
#   - DermaKG-IGR: original IGR-flagged candidates (kept for reference).
#   - Co-occurrence + DermaKG hybrid: re-rank Co-occurrence's top-K with
#     DermaKG-Causal's subgroup-conditional posterior. Tests whether
#     DermaKG adds VALUE as a fairness layer atop a strong retrieval method.
#   - Safety violation rate uses PrimeKG `contraindication` edges as the
#     out-of-distribution oracle, NOT the ATC filter itself (was circular).
#   - Per-fold + aggregate output for paper tables
# =============================================================================

# -----------------------------------------------------------------------------
# §0  CONFIG
# -----------------------------------------------------------------------------

import os, sys, json, math, time, warnings, random
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dermakg_kaggle_all_in_one import OUTPUT_DIR, TARGET_SUBGROUP, primekg_df, skin_stats, is_safe_recommendation, atc_class_prefix
warnings.filterwarnings("ignore")

_REQUIRED = ["primekg_df", "skin_stats",
             "is_safe_recommendation", "atc_class_prefix",
             "OUTPUT_DIR", "TARGET_SUBGROUP"]
_missing = [n for n in _REQUIRED if n not in dir()]
if _missing:
    raise RuntimeError(
        f"Comparison cell expects the main pipeline cell to have run. "
        f"Missing names: {_missing}. Run dermakg_kaggle_all_in_one.py first."
    )

OUT = OUTPUT_DIR
SEED = 42

# Toggles
RUN_KGE_BASELINES   = True   # pykeen — installs on first run, ~30s/model
RUN_TXGNN           = False  # set True if you've installed TxGNN package
RUN_LLM_ZERO_SHOT   = False  # set True + provide API key below
LLM_API_PROVIDER    = "anthropic"
LLM_MODEL           = "claude-sonnet-4-5"
LLM_API_KEY_ENV     = "ANTHROPIC_API_KEY"
LLM_N_DISEASES      = 50
KGE_EMBEDDING_DIM   = 64
KGE_NUM_EPOCHS      = 30
N_FOLDS             = 5      # 5-fold CV with different seeds per fold
HYBRID_TOPK         = 30     # Co-occurrence top-K to re-rank with DermaKG

# -----------------------------------------------------------------------------
# §1  LOAD DERMAKG ARTIFACTS (candidates + full posteriors + EB prior)
# -----------------------------------------------------------------------------

print("=" * 78)
print("§1  LOADING DERMAKG-CAUSAL ARTIFACTS FROM DISK")
print("=" * 78)

def _norm(s):
    return str(s).lower().strip()

_cand_csv = os.path.join(OUT, "igr_all_candidates.csv")
_post_csv = os.path.join(OUT, "scp_all_posteriors.csv")
_prior_json = os.path.join(OUT, "scp_eb_prior.json")

if not os.path.exists(_cand_csv):
    raise RuntimeError(f"Missing {_cand_csv}. Run main pipeline first.")

_cand_df = pd.read_csv(_cand_csv)
print(f"  IGR candidates: {len(_cand_df)} (across "
      f"{_cand_df['disease'].nunique()} diseases)")

# scp_all_posteriors.csv is new — the main pipeline must have been re-run
# AFTER the patch that added it. Detect & explain if it's missing.
have_posteriors = os.path.exists(_post_csv)
if have_posteriors:
    _post_df = pd.read_csv(_post_csv)
    print(f"  SCP posteriors: {len(_post_df)} edges with both subgroups")
else:
    _post_df = pd.DataFrame()
    print(f"  ⚠ {_post_csv} not found. Re-run main pipeline to enable")
    print(f"    DermaKG-Posterior baseline (stand-alone ranker).")

if os.path.exists(_prior_json):
    with open(_prior_json) as f:
        _eb_prior = json.load(f)
    print(f"  EB prior: Beta({_eb_prior['alpha0']:.3f}, {_eb_prior['beta0']:.3f})")
else:
    _eb_prior = {"alpha0": 1.0, "beta0": 1.0}
    print(f"  ⚠ EB prior file missing — using uniform Beta(1,1)")

# -----------------------------------------------------------------------------
# §2  GROUND TRUTH + CONTRAINDICATION ORACLE (for safety eval)
# -----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("§2  BUILDING GROUND TRUTH + CONTRAINDICATION ORACLE")
print("=" * 78)

# 2a — Indications (positive examples for ranking eval)
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

# 2b — Contraindications (oracle for safety violations)
# These are edges that PrimeKG explicitly flags as drugs that should NOT
# be given for the disease. Using them as the safety oracle is independent
# of our ATC filter — no circularity.
contraindications: Dict[str, set] = defaultdict(set)

df = primekg_df.rename(columns={c: str(c).lower().strip()
                                for c in primekg_df.columns})
for row in df.itertuples(index=False):
    rel = _norm(getattr(row, "relation", ""))
    x_t = _norm(getattr(row, "x_type", ""))
    y_t = _norm(getattr(row, "y_type", ""))
    if x_t == "disease" and y_t == "drug":
        d, dr = _norm(row.x_name), _norm(row.y_name)
    elif y_t == "disease" and x_t == "drug":
        d, dr = _norm(row.y_name), _norm(row.x_name)
    else:
        continue
    if rel == "indication" and d in disease_fst:
        gt_indications[d].add(dr)
    elif rel == "contraindication" and d in disease_fst:
        contraindications[d].add(dr)

n_indications = sum(len(v) for v in gt_indications.values())
n_contras = sum(len(v) for v in contraindications.values())
print(f"  indications: {len(gt_indications)} diseases, {n_indications} edges")
print(f"  contraindications: {len(contraindications)} diseases, {n_contras} edges")
print(f"     (used as out-of-distribution oracle for safety violation rate)")

# Universe of drugs for ranking
ALL_DRUGS = sorted({dr for drs in gt_indications.values() for dr in drs})
DRUG_TO_IDX = {dr: i for i, dr in enumerate(ALL_DRUGS)}
print(f"  drug universe: {len(ALL_DRUGS)} unique drugs")

# Drug popularity (used by Popularity baseline + as fallback)
drug_pop_global = Counter()
for d, drugs in gt_indications.items():
    for dr in drugs:
        drug_pop_global[dr] += 1
sorted_by_pop = [d for d, _ in sorted(drug_pop_global.items(),
                                      key=lambda x: -x[1])]
sorted_by_pop += [dr for dr in ALL_DRUGS if dr not in drug_pop_global]

# -----------------------------------------------------------------------------
# §3  EVALUATION HARNESS
# -----------------------------------------------------------------------------

def _ndcg_at_k(ranked: List[str], relevant: set, k: int) -> float:
    dcg = 0.0
    for i, dr in enumerate(ranked[:k]):
        if dr in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2)
                for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def _safety_violation_rate(top10: List[str], disease: str) -> float:
    """Fraction of top-10 predictions that PrimeKG explicitly contraindicates.

    Uses the PrimeKG `contraindication` relation as the OOD oracle — NOT
    our own ATC filter. This avoids circularity (the original eval was
    measuring our filter against itself).
    """
    contras = contraindications.get(disease, set())
    if not top10:
        return 0.0
    n_violations = sum(1 for dr in top10 if dr in contras)
    return n_violations / len(top10)


def evaluate_method(rank_fn, test_edges) -> Dict:
    """Apply rank_fn(disease) → ranked drug list, score against held-out drugs.

    Returns dict of FST-stratified metrics for one fold.
    """
    by_group = {g: defaultdict(list) for g in ("I-III", "IV-VI", "overall")}
    cache: Dict[str, List[str]] = {}

    for disease, drug, group in test_edges:
        if disease not in cache:
            try:
                cache[disease] = rank_fn(disease)
            except Exception:
                cache[disease] = []
        ranked = cache[disease]
        if not ranked:
            continue
        try:
            rank = ranked.index(drug) + 1
        except ValueError:
            rank = len(ranked) + 1

        relevant = {drug}
        for g in (group, "overall"):
            by_group[g]["hits@1"].append(1.0 if rank <= 1 else 0.0)
            by_group[g]["hits@5"].append(1.0 if rank <= 5 else 0.0)
            by_group[g]["hits@10"].append(1.0 if rank <= 10 else 0.0)
            by_group[g]["mrr"].append(1.0 / rank)
            by_group[g]["ndcg@10"].append(_ndcg_at_k(ranked, relevant, 10))
            by_group[g]["safety_violation_rate"].append(
                _safety_violation_rate(ranked[:10], disease))

    out = {}
    for g, ms in by_group.items():
        out[g] = {k: float(np.mean(v)) if v else 0.0 for k, v in ms.items()}
        out[g]["n_test"] = len(ms.get("hits@1", []))
    return out


# -----------------------------------------------------------------------------
# §4  RANKER FACTORIES — produces a ranker given a training set
# -----------------------------------------------------------------------------
# Each factory takes a train-set dict {disease: set(drugs)} and a fold seed,
# returns a callable rank_fn(disease) → ranked list of drugs.
# Abstraction needed for k-fold CV: we re-train each fold.

def make_random_ranker(train_indications, fold_seed):
    rng_local = random.Random(fold_seed)
    def ranker(disease):
        drugs = list(ALL_DRUGS)
        rng_local.shuffle(drugs)
        return drugs
    return ranker

def make_popularity_ranker(train_indications, fold_seed):
    pop = Counter()
    for d, drs in train_indications.items():
        for dr in drs:
            pop[dr] += 1
    sorted_pop = [d for d, _ in sorted(pop.items(), key=lambda x: -x[1])]
    sorted_pop += [dr for dr in ALL_DRUGS if dr not in pop]
    def ranker(disease):
        return list(sorted_pop)
    return ranker

def make_cooccurrence_ranker(train_indications, fold_seed):
    """Disease-similar (Jaccard over drugs) → rank drugs by frequency."""
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

def make_js_divergence_ranker(train_indications, fold_seed):
    """Disease similarity by JS-divergence of drug-distributions."""
    drug_set = {d: drs for d, drs in train_indications.items()}
    pop = Counter()
    for drs in drug_set.values():
        for dr in drs:
            pop[dr] += 1
    pop_fallback = [d for d, _ in sorted(pop.items(), key=lambda x: -x[1])]
    pop_fallback += [dr for dr in ALL_DRUGS if dr not in pop]

    n_d = len(drug_set)
    disease_idx = {d: i for i, d in enumerate(drug_set.keys())}
    M = np.zeros((n_d, len(ALL_DRUGS)))
    for d, drs in drug_set.items():
        for dr in drs:
            if dr in DRUG_TO_IDX:
                M[disease_idx[d], DRUG_TO_IDX[dr]] = 1.0

    def _js(p, q, eps=1e-12):
        ps = p.sum(); qs = q.sum()
        if ps < eps or qs < eps:
            return 1.0
        p = p / ps; q = q / qs
        m = 0.5 * (p + q)
        return 0.5 * np.sum(np.where(p > 0, p * np.log(p / (m + eps) + eps), 0)) + \
               0.5 * np.sum(np.where(q > 0, q * np.log(q / (m + eps) + eps), 0))

    def ranker(disease):
        if disease not in disease_idx:
            return list(pop_fallback)
        target_vec = M[disease_idx[disease]]
        if target_vec.sum() == 0:
            return list(pop_fallback)
        sims = []
        for other_d, oi in disease_idx.items():
            if other_d == disease:
                continue
            other_vec = M[oi]
            if other_vec.sum() == 0:
                continue
            sims.append((other_d, -_js(target_vec, other_vec)))
        sims.sort(key=lambda x: -x[1])
        score = Counter()
        for other_d, sim in sims[:20]:
            for dr in drug_set[other_d]:
                if dr not in drug_set.get(disease, set()):
                    score[dr] += math.exp(sim)
        ranked = [d for d, _ in sorted(score.items(), key=lambda x: -x[1])]
        seen = set(ranked)
        ranked += [dr for dr in pop_fallback if dr not in seen]
        return ranked
    return ranker

def make_network_proximity_ranker(train_indications, fold_seed):
    """TxGNN-style network proximity baseline."""
    drug_set = {d: drs for d, drs in train_indications.items()}
    pop = Counter()
    for drs in drug_set.values():
        for dr in drs:
            pop[dr] += 1
    pop_fallback = [d for d, _ in sorted(pop.items(), key=lambda x: -x[1])]
    pop_fallback += [dr for dr in ALL_DRUGS if dr not in pop]
    drug_to_diseases = defaultdict(set)
    for d, drs in drug_set.items():
        for dr in drs:
            drug_to_diseases[dr].add(d)

    def ranker(disease):
        target = drug_set.get(disease, set())
        if not target:
            return list(pop_fallback)
        target_neighbors = {d for d, drs in drug_set.items() if drs & target}
        score = Counter()
        for dr in ALL_DRUGS:
            if dr in target:
                continue
            co = drug_to_diseases[dr]
            score[dr] = len(co & target_neighbors) / max(len(co), 1)
        ranked = [d for d, _ in sorted(score.items(), key=lambda x: -x[1])]
        seen = set(ranked)
        ranked += [dr for dr in pop_fallback if dr not in seen]
        return ranked
    return ranker

def make_dermakg_posterior_ranker(train_indications, fold_seed):
    """STAND-ALONE DermaKG-Causal ranker.

    For each (disease, drug) pair, returns the SCP-KG posterior mean for
    the IV-VI subgroup (the underrepresented group). Pairs not in the
    KG fall back to the EB prior mean + a tiny popularity tie-breaker.

    This is the apples-to-apples version of DermaKG-Causal — scores ALL
    166 drugs for every disease, exactly like Co-occurrence and TransE.
    """
    if _post_df.empty:
        return make_dermakg_igr_ranker(train_indications, fold_seed)

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
            key = (disease, dr)
            m = lookup.get(key, prior_mean)
            score = m + 1e-3 * pop.get(dr, 0)
            scores.append((dr, score))
        return [d for d, _ in sorted(scores, key=lambda x: -x[1])]
    return ranker

def make_dermakg_igr_ranker(train_indications, fold_seed):
    """ORIGINAL DermaKG-Causal ranker — only scores IGR-flagged candidates."""
    per_disease = defaultdict(list)
    for _, row in _cand_df.iterrows():
        d = _norm(row["disease"])
        dr = _norm(row["drug"])
        eig = float(row["expected_information_gain"])
        per_disease[d].append((dr, eig))
    cache = {}
    for d, lst in per_disease.items():
        lst.sort(key=lambda x: -x[1])
        cache[d] = [dr for dr, _ in lst]

    pop = Counter()
    for drs in train_indications.values():
        for dr in drs:
            pop[dr] += 1
    pop_fallback = [d for d, _ in sorted(pop.items(), key=lambda x: -x[1])]
    pop_fallback += [dr for dr in ALL_DRUGS if dr not in pop]

    def ranker(disease):
        if disease in cache:
            ranked = list(cache[disease])
            seen = set(ranked)
            ranked += [dr for dr in pop_fallback if dr not in seen]
            return ranked
        return list(pop_fallback)
    return ranker

def make_hybrid_ranker(train_indications, fold_seed):
    """Co-occurrence + DermaKG re-rank.

    Take Co-occurrence's top-K, then re-rank by DermaKG-Causal's posterior
    mean for the underrepresented subgroup. Tests whether DermaKG adds
    fairness value as a re-ranking layer atop a strong retrieval method.
    """
    coocc = make_cooccurrence_ranker(train_indications, fold_seed)

    if _post_df.empty:
        return coocc  # No posteriors → no re-ranking to do

    target = TARGET_SUBGROUP
    mean_col = "post_mean_ivvi" if target == "IV-VI" else "post_mean_iii"
    lookup: Dict[Tuple[str, str], float] = {}
    for _, row in _post_df.iterrows():
        d = _norm(row["disease"])
        dr = _norm(row["drug"])
        lookup[(d, dr)] = float(row[mean_col])
    prior_mean = _eb_prior["alpha0"] / (_eb_prior["alpha0"] + _eb_prior["beta0"])

    def ranker(disease):
        coocc_ranked = coocc(disease)
        topk = coocc_ranked[:HYBRID_TOPK]
        rest = coocc_ranked[HYBRID_TOPK:]
        rerank_scores = [(dr, lookup.get((disease, dr), prior_mean))
                         for dr in topk]
        reranked = [dr for dr, _ in sorted(rerank_scores, key=lambda x: -x[1])]
        return reranked + rest
    return ranker

# -----------------------------------------------------------------------------
# §5  KGE BASELINES (pykeen) — must be retrained per fold
# -----------------------------------------------------------------------------

def make_kge_ranker(model_name, train_indications, fold_seed):
    """Train a KGE model on this fold's training set and return a ranker."""
    try:
        import pykeen
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
        import torch
    except ImportError:
        os.system(f"{sys.executable} -m pip install pykeen --quiet "
                  f"--break-system-packages 2>&1 | tail -3")
        import pykeen
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
        import torch

    triples = []
    for d, drs in train_indications.items():
        for dr in drs:
            triples.append((d, "indication", dr))
    if len(triples) < 10:
        return None

    triples_arr = np.array(triples, dtype=str)
    tf = TriplesFactory.from_labeled_triples(triples_arr)

    try:
        res = pipeline(
            training=tf, testing=tf,
            model=model_name,
            model_kwargs=dict(embedding_dim=KGE_EMBEDDING_DIM),
            training_kwargs=dict(num_epochs=KGE_NUM_EPOCHS, batch_size=512),
            random_seed=fold_seed,
            evaluator_kwargs=dict(filtered=True),
        )
    except Exception as exc:
        print(f"    {model_name} failed: {exc}")
        return None

    model = res.model
    ent_to_id = tf.entity_to_id
    rel_id = tf.relation_to_id.get("indication")

    def ranker(disease):
        if disease not in ent_to_id:
            return list(sorted_by_pop)
        d_id = ent_to_id[disease]
        scores = []
        with torch.no_grad():
            for dr in ALL_DRUGS:
                if dr not in ent_to_id:
                    scores.append((dr, -1e9))
                    continue
                triple = torch.tensor([[d_id, rel_id, ent_to_id[dr]]])
                s = float(model.score_hrt(triple).item())
                scores.append((dr, s))
        return [d for d, _ in sorted(scores, key=lambda x: -x[1])]
    return ranker

# -----------------------------------------------------------------------------
# §6  K-FOLD CROSS-VALIDATION DRIVER
# -----------------------------------------------------------------------------

def make_fold_split(fold_seed):
    """Build a (train, test) split for one CV fold."""
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

# Map of method name → ranker factory
RANKER_FACTORIES = {
    "Random":            make_random_ranker,
    "Popularity":        make_popularity_ranker,
    "Co-occurrence":     make_cooccurrence_ranker,
    "JS-divergence":     make_js_divergence_ranker,
    "Network-proximity": make_network_proximity_ranker,
    "DermaKG-IGR":       make_dermakg_igr_ranker,
    "DermaKG-Posterior": make_dermakg_posterior_ranker,
    "Co-occ+DermaKG":    make_hybrid_ranker,
}

print("\n" + "=" * 78)
print(f"§6  RUNNING {N_FOLDS}-FOLD CROSS-VALIDATION")
print("=" * 78)
print(f"  baseline methods : {list(RANKER_FACTORIES.keys())}")
print(f"  KGE methods      : "
      f"{['TransE','DistMult','ComplEx','RotatE'] if RUN_KGE_BASELINES else []}")
print(f"  test edges/fold  : ~{int(0.2 * sum(len(v) for v in gt_indications.values()))}")

# fold_results: {method_name: [fold0_metrics, fold1_metrics, ...]}
fold_results: Dict[str, List[Dict]] = defaultdict(list)

for fold_idx in range(N_FOLDS):
    fold_seed = SEED + fold_idx * 100
    print(f"\n  --- FOLD {fold_idx+1}/{N_FOLDS} (seed={fold_seed}) ---")
    train_indications, test_edges = make_fold_split(fold_seed)

    n_test_iii = sum(1 for _, _, g in test_edges if g == "I-III")
    n_test_ivvi = sum(1 for _, _, g in test_edges if g == "IV-VI")
    print(f"    test: {len(test_edges)} edges (I-III: {n_test_iii}, "
          f"IV-VI: {n_test_ivvi})")

    # Standard methods
    for name, factory in RANKER_FACTORIES.items():
        t0 = time.time()
        ranker = factory(train_indications, fold_seed)
        metrics = evaluate_method(ranker, test_edges)
        fold_results[name].append(metrics)
        print(f"    {name:<22s}: H@10={metrics['overall']['hits@10']:.3f} "
              f"({time.time()-t0:.1f}s)")

    # KGE methods (slow)
    if RUN_KGE_BASELINES:
        for kge_name in ["TransE", "DistMult", "ComplEx", "RotatE"]:
            t0 = time.time()
            ranker = make_kge_ranker(kge_name, train_indications, fold_seed)
            if ranker is None:
                continue
            metrics = evaluate_method(ranker, test_edges)
            fold_results[kge_name].append(metrics)
            print(f"    {kge_name:<22s}: H@10={metrics['overall']['hits@10']:.3f} "
                  f"({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------------
# §7  AGGREGATE: mean ± 95% CI across folds
# -----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("§7  AGGREGATING — mean ± 95% CI across folds")
print("=" * 78)

def agg_folds(fold_metrics: List[Dict]) -> Dict:
    """Compute mean + 95% CI across folds for each (group, metric)."""
    agg = {}
    for g in ("overall", "I-III", "IV-VI"):
        agg[g] = {}
        if not fold_metrics:
            continue
        metric_keys = set(fold_metrics[0].get(g, {}).keys())
        for m in metric_keys:
            vals = [f[g][m] for f in fold_metrics if g in f and m in f[g]]
            if not vals:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci = 1.96 * std / math.sqrt(max(len(vals), 1))
            agg[g][m] = {"mean": mean, "std": std, "ci_95": ci, "n_folds": len(vals)}
    return agg

aggregated = {name: agg_folds(folds) for name, folds in fold_results.items()}

# -----------------------------------------------------------------------------
# §8  TABLES
# -----------------------------------------------------------------------------

def _fmt(stats):
    if not stats:
        return "—"
    return f"{stats['mean']:.3f}±{stats['ci_95']:.3f}"

# Table 2: main accuracy
main_rows = []
for name, agg in aggregated.items():
    row = {"method": name}
    for g in ("overall", "I-III", "IV-VI"):
        for m in ("hits@1", "hits@5", "hits@10", "mrr", "ndcg@10"):
            row[f"{m}_{g}"] = _fmt(agg.get(g, {}).get(m))
    main_rows.append(row)
main_df = pd.DataFrame(main_rows)
main_df.to_csv(os.path.join(OUT, "paper_table_main.csv"), index=False)
print(f"  → {OUT}/paper_table_main.csv ({len(main_df)} rows)")

# Table 3: fairness — fairness gap, EOM, safety
fair_rows = []
for name, agg in aggregated.items():
    iii = agg.get("I-III", {})
    ivvi = agg.get("IV-VI", {})
    if not iii or not ivvi:
        continue
    for metric in ("hits@10", "mrr", "ndcg@10", "safety_violation_rate"):
        v_iii = iii.get(metric, {}).get("mean", float("nan"))
        v_ivvi = ivvi.get(metric, {}).get("mean", float("nan"))
        ci_iii = iii.get(metric, {}).get("ci_95", 0)
        ci_ivvi = ivvi.get(metric, {}).get("ci_95", 0)
        if math.isnan(v_iii) or math.isnan(v_ivvi):
            continue
        gap = abs(v_iii - v_ivvi)
        # EOM: for utility metrics, min/max; for safety (lower=better), invert
        if metric != "safety_violation_rate":
            eom = (min(v_iii, v_ivvi) / max(max(v_iii, v_ivvi), 1e-9))
        else:
            best = 1 - max(v_iii, v_ivvi)
            worst = 1 - min(v_iii, v_ivvi)
            eom = best / max(worst, 1e-9)
        fair_rows.append(dict(
            method=name, metric=metric,
            value_I_III=f"{v_iii:.3f}±{ci_iii:.3f}",
            value_IV_VI=f"{v_ivvi:.3f}±{ci_ivvi:.3f}",
            fairness_gap=f"{gap:.3f}",
            equality_of_opportunity=f"{eom:.3f}" if eom <= 1 else "—",
        ))
fair_df = pd.DataFrame(fair_rows)
fair_df.to_csv(os.path.join(OUT, "paper_table_fairness.csv"), index=False)
print(f"  → {OUT}/paper_table_fairness.csv ({len(fair_df)} rows)")

# Long-form for paper analysis
long_rows = []
for name, agg in aggregated.items():
    for g, ms in agg.items():
        for metric, stats in ms.items():
            long_rows.append({
                "method": name, "stratum": g, "metric": metric,
                "mean": stats["mean"], "std": stats["std"],
                "ci_95": stats["ci_95"], "n_folds": stats["n_folds"],
            })
long_df = pd.DataFrame(long_rows)
long_df.to_csv(os.path.join(OUT, "baseline_comparison.csv"), index=False)
print(f"  → {OUT}/baseline_comparison.csv ({len(long_df)} rows)")

# Pretty-print
print(f"\n--- TABLE 2 (MAIN) — mean ± 95% CI across {N_FOLDS} folds ---")
with pd.option_context("display.max_colwidth", 30, "display.width", 220):
    print(main_df.to_string(index=False))

print("\n--- TABLE 3 (FAIRNESS) — disparity across FST groups ---")
with pd.option_context("display.max_colwidth", 30, "display.width", 220):
    print(fair_df.to_string(index=False))

# -----------------------------------------------------------------------------
# §9  STATISTICAL SIGNIFICANCE — paired bootstrap on per-fold H@10
# -----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("§9  STATISTICAL SIGNIFICANCE — paired bootstrap on per-fold H@10")
print("=" * 78)

def per_fold_h10(method, stratum="overall"):
    return [f[stratum]["hits@10"] for f in fold_results.get(method, [])]

target_methods = ["DermaKG-Posterior", "DermaKG-IGR", "Co-occ+DermaKG"]
baseline_methods = [m for m in fold_results.keys()
                    if m not in target_methods]

for target in target_methods:
    target_h = np.array(per_fold_h10(target))
    if len(target_h) == 0:
        continue
    print(f"\n  {target} vs baselines — H@10 difference (mean ± 95% CI), p (one-sided):")
    for base in baseline_methods:
        base_h = np.array(per_fold_h10(base))
        if len(base_h) == 0:
            continue
        diffs = target_h - base_h
        rng_b = np.random.RandomState(SEED)
        boot_means = [diffs[rng_b.choice(len(diffs), len(diffs), replace=True)].mean()
                      for _ in range(2000)]
        boot_means = np.array(boot_means)
        mean_diff = float(diffs.mean())
        ci = (1.96 * float(diffs.std(ddof=1)) / math.sqrt(len(diffs))
              if len(diffs) > 1 else 0)
        p_val = float((boot_means <= 0).mean())
        print(f"    vs {base:<22s}: Δ = {mean_diff:+.3f} ± {ci:.3f}, p = {p_val:.3f}")

# -----------------------------------------------------------------------------
# §10  PAPER GUIDANCE
# -----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("§10  PAPER GUIDANCE — what your data supports vs what it doesn't")
print("=" * 78)
print(f"""
EVALUATION PROTOCOL (for §3.2 of the paper):

  • Held-out link prediction over PrimeKG `indication` edges (TxGNN protocol)
  • {N_FOLDS}-fold cross-validation; 80/20 split each fold; mean ± 95% CI
  • {len(ALL_DRUGS)} drug universe; methods rank ALL drugs per disease
  • FST stratification: each held-out edge tagged by disease's majority FST
    in DermaCon-IN/Fitzpatrick17k (cohort-derived, NOT patient-level)
  • Safety oracle: PrimeKG `contraindication` relation (n={n_contras} edges,
    independent of the ATC filter to avoid circularity)

WHAT THE EVAL CAN TELL YOU:

  ✓ Recovery of held-out indication edges (relative)
  ✓ FST-stratified disparity between methods
  ✓ Rate of recommending PrimeKG-contraindicated drugs (independent oracle)
  ✓ Statistical significance via paired bootstrap on per-fold metrics

WHAT THE EVAL CANNOT TELL YOU:

  ✗ Whether a recommendation is clinically appropriate beyond what
    PrimeKG already encodes (no clinician oracle)
  ✗ Whether `disease_fst` is a true dermatologic disparity signal vs.
    geographic/cohort-driven artifact (Lyme, syphilis, cutaneous anthrax)

HOW TO READ TABLE 2:

  • DermaKG-Posterior is the apples-to-apples stand-alone version:
    scores ALL drugs via subgroup-conditional posterior mean.
  • DermaKG-IGR is the IGR-flagged version (only candidates from disparity
    diagnosis). Expected to score lower on raw H@K — that's the design.
  • Co-occ+DermaKG is the hybrid: Co-occurrence's top-{HYBRID_TOPK}
    re-ranked by DermaKG. If it preserves Co-occurrence's H@K and
    closes the fairness gap, that's the headline result.

HOW TO FRAME THE CONTRIBUTION:

  Frame 1 (FAIRNESS LAYER, recommended if hybrid wins):
    'A subgroup-conditional posterior re-ranking module that closes the
    FST fairness gap of strong KG retrieval baselines while preserving
    accuracy, validated via {N_FOLDS}-fold CV across {len(gt_indications)}
    dermatologic conditions.'

  Frame 2 (DIAGNOSTIC SYSTEM):
    'Empirical-Bayes posterior diagnostics (CEG, TVD) for evidence-
    distribution disparity in dermatology drug knowledge graphs, with
    case studies on PrimeKG indications stratified by Fitzpatrick skin
    type.'

  AVOID:
    'A drug-discovery system' — IGR's Type B/C extrapolation runs on
    fragile fallbacks. Without SapBERT integration and clinical oracle,
    discovery claims fail review.

VENUES:
    ML4H 2026 workshop (~6 weeks)   — Frame 2
    FAccT 2026 (~10 weeks)          — Frame 1
    NeurIPS 2026 D&B (~12 weeks)    — Either, plus clinical oracle + Croissant
""")