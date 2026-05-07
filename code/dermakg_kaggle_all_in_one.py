#!/usr/bin/env python3
# =============================================================================
# DermaKG-Causal v1.0 — ALL-IN-ONE KAGGLE CELL
# =============================================================================
# Single self-contained file. Paste into one Kaggle code cell and run.
#
# Data loading is lifted verbatim from your v5.5 DataLoader:
#   - PrimeKG:        https://dataverse.harvard.edu/api/access/datafile/6180620
#   - Fitzpatrick17k: https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv
#   - DermaCon-IN:    /kaggle/input/datasets/avishekrauniyar/dermacon-in-dataset-release-v1-0/METADATA/Skin_Metadata.csv
#                     (with two fallback paths + kagglehub)
#
# Cache directory is /kaggle/working/dermakg_data/ (downloads land here).
# Pipeline outputs go to /kaggle/working/dermakg_results/.
# =============================================================================

from __future__ import annotations

# -----------------------------------------------------------------------------
# §0  USER-EDITABLE CONFIG
# -----------------------------------------------------------------------------

# Optional manual overrides — None = let DataLoader auto-discover/download.
#
# NOTE FOR REVIEWERS RUNNING OUTSIDE KAGGLE:
# This path is a Kaggle-environment convenience path and will not exist on
# other systems. The loader will fall back automatically through:
#   1. kagglehub download
#   2. A manually placed Skin_Metadata.csv at:
#      /kaggle/working/dermakg_data/dermacon/Skin_Metadata.csv
#      (download from Harvard Dataverse DOI: 10.7910/DVN/W7OUZM —
#       see https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/W7OUZM)
#   3. Graceful degradation: skin stats will use Fitzpatrick17k only.
_DERMACON_OVERRIDE_PATH = (
    "/kaggle/input/datasets/avishekrauniyar/"
    "dermacon-in-dataset-release-v1-0/METADATA/Skin_Metadata.csv"
)
_FITZPATRICK_OVERRIDE_PATH = None      # None ⇒ download from GitHub
_PRIMEKG_OVERRIDE_PATH = None          # None ⇒ download from Harvard Dataverse

DATA_DIR_STR = "/kaggle/working/dermakg_data"
OUTPUT_DIR = "/kaggle/working/dermakg_results"

TARGET_SUBGROUP = "IV-VI"
N_PER_TRIAL = 30
TOP_K_DISEASES = 200
TOP_K_CANDIDATES = 1000
INCLUDE_TVD = True

# -----------------------------------------------------------------------------
# §1  IMPORTS & PATHS
# -----------------------------------------------------------------------------

import argparse, csv, glob, gzip, hashlib, io, json, logging, math, os
import pickle, re, shutil, subprocess, sys, time, unittest, urllib.request, warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, sparse, special, stats

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dermakg_causal")

DATA_DIR = Path(DATA_DIR_STR)
SEED_DEFAULT = 42


# =============================================================================
# §2  V5.5 DATA LOADER — lifted from your dermakg_v5_5.py DataLoader class
# =============================================================================
# Identical paths/URLs/columns. Feeds into the new pipeline below.
# =============================================================================

class DataLoader:
    """Verbatim port of dermakg_v5_5.DataLoader (load methods only)."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        for sub in ("primekg", "fitzpatrick17k", "dermacon", "ontologies"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)

    # ----- PrimeKG ----------------------------------------------------------

    def load_primekg(self) -> pd.DataFrame:
        if _PRIMEKG_OVERRIDE_PATH and Path(_PRIMEKG_OVERRIDE_PATH).exists():
            logger.info("PrimeKG: using override %s", _PRIMEKG_OVERRIDE_PATH)
            return pd.read_csv(_PRIMEKG_OVERRIDE_PATH, low_memory=False)
        # Auto-discover existing Kaggle inputs first
        for hit in sorted(glob.glob("/kaggle/input/**/kg.csv", recursive=True)):
            try:
                size_mb = os.path.getsize(hit) / (1024 * 1024)
            except OSError:
                continue
            if size_mb >= 50:    # PrimeKG kg.csv is ~750MB
                logger.info("PrimeKG: found existing input %s (%.0f MB)", hit, size_mb)
                return pd.read_csv(hit, low_memory=False)
        path = self.data_dir / "primekg" / "primekg.csv"
        if not path.exists():
            if not _HAS_REQUESTS:
                raise RuntimeError("requests not installed; cannot download PrimeKG")
            logger.info("Downloading PrimeKG (~250 MB) from Harvard Dataverse...")
            r = requests.get(
                "https://dataverse.harvard.edu/api/access/datafile/6180620",
                stream=True, timeout=600)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        df = pd.read_csv(path, low_memory=False)
        logger.info("PrimeKG: %d edges", len(df))
        return df

    # ----- Fitzpatrick17k ---------------------------------------------------

    def load_fitzpatrick(self) -> pd.DataFrame:
        if _FITZPATRICK_OVERRIDE_PATH:
            p = Path(_FITZPATRICK_OVERRIDE_PATH)
            if p.exists():
                logger.info("Fitzpatrick17k: using override %s", p)
                return pd.read_csv(p)
            logger.warning("Fitzpatrick17k override %s missing; falling back.", p)
        # Auto-discover Kaggle inputs
        for hit in sorted(glob.glob(
                "/kaggle/input/**/fitzpatrick17k.csv", recursive=True)):
            logger.info("Fitzpatrick17k: found existing input %s", hit)
            return pd.read_csv(hit)
        path = self.data_dir / "fitzpatrick17k" / "fitzpatrick17k.csv"
        if not path.exists():
            if not _HAS_REQUESTS:
                raise RuntimeError("requests not installed; cannot download Fitzpatrick17k")
            logger.info("Downloading Fitzpatrick17k from GitHub raw...")
            r = requests.get(
                "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/"
                "main/fitzpatrick17k.csv", timeout=120)
            r.raise_for_status()
            with open(path, "w") as f:
                f.write(r.text)
        return pd.read_csv(path)

    # ----- DermaCon-IN ------------------------------------------------------

    def load_dermacon(self) -> pd.DataFrame:
        if _DERMACON_OVERRIDE_PATH:
            p = Path(_DERMACON_OVERRIDE_PATH)
            if p.exists():
                logger.info("DermaCon: using override %s", p)
                return self._parse_dermacon(p)
            logger.warning("DermaCon override %s missing; falling back.", p)
        candidates = [
            Path("/kaggle/input/datasets/avishekrauniyar/"
                 "dermacon-in-dataset-release-v1-0/METADATA/Skin_Metadata.csv"),
            Path("/kaggle/input/dermacon-in-dataset-release-v1-0/"
                 "METADATA/Skin_Metadata.csv"),
            Path("/kaggle/input/dermacon-in/METADATA/Skin_Metadata.csv"),
        ]
        for c in candidates:
            if c.exists():
                logger.info("DermaCon: found at %s", c)
                return self._parse_dermacon(c)
        try:
            import kagglehub
            path = kagglehub.dataset_download(
                "avishekrauniyar/dermacon-in-dataset-release-v1-0",
                force_download=False)
            for c in Path(path).rglob("Skin_Metadata.csv"):
                return self._parse_dermacon(c)
        except Exception as e:
            logger.warning("DermaCon kagglehub fallback failed: %s", e)

        # Fallback: download canonical release from Harvard Dataverse
        try:
            if _HAS_REQUESTS:
                out_dir = self.data_dir / "dermacon"
                out_dir.mkdir(parents=True, exist_ok=True)
                # Harvard Dataverse DOI: 10.7910/DVN/W7OUZM
                # Reviewers should manually download Skin_Metadata.csv from
                # https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/W7OUZM
                # and place it at out_dir/Skin_Metadata.csv
                manual_path = out_dir / "Skin_Metadata.csv"
                if manual_path.exists():
                    logger.info("DermaCon: using manually downloaded %s", manual_path)
                    return self._parse_dermacon(manual_path)
                logger.warning(
                    "DermaCon-IN: Kaggle path missing. Please download "
                    "Skin_Metadata.csv from Harvard Dataverse "
                    "(DOI 10.7910/DVN/W7OUZM) and place at %s", manual_path)
        except Exception as e:
            logger.warning("DermaCon Harvard Dataverse fallback failed: %s", e)

        logger.warning(
            "DermaCon-IN unavailable; skin stats will use Fitzpatrick17k only.")
        return pd.DataFrame()

    @staticmethod
    def _parse_dermacon(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        rename = {
            "Disease_label": "disease_label", "Fitzpatrick": "fitzpatrick",
            "Monk_skin_tone": "monk_skin_tone", "Age": "age",
            "Body_part": "body_part", "Main_class": "main_class",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "fitzpatrick" in df.columns:
            df["fst_numeric"] = df["fitzpatrick"].apply(
                lambda v: int(re.search(r"(\d+)", str(v)).group(1))
                if not pd.isna(v) and re.search(r"(\d+)", str(v)) else None)
        if "monk_skin_tone" in df.columns:
            df["mst_numeric"] = df["monk_skin_tone"].apply(
                lambda v: int(re.search(r"(\d+)", str(v)).group(1))
                if not pd.isna(v) and re.search(r"(\d+)", str(v)) else None)
        return df

    # ----- compute_skin_stats — verbatim from v5.5 -------------------------

    def compute_skin_stats(
        self, fitz_df: pd.DataFrame, dermacon_df: pd.DataFrame,
    ) -> Dict:
        """Same shape as dermakg_v5_5.DataLoader.compute_skin_stats."""
        unified: Dict[str, Dict] = {}
        if "label" in fitz_df.columns and "fitzpatrick_scale" in fitz_df.columns:
            for label, group in fitz_df.groupby("label"):
                if pd.isna(label):
                    continue
                key = str(label).lower().strip()
                fst = group["fitzpatrick_scale"].value_counts().to_dict()
                i_iii = sum(fst.get(v, 0) for v in [1, 2, 3])
                iv_vi = sum(fst.get(v, 0) for v in [4, 5, 6])
                total = i_iii + iv_vi
                unified[key] = dict(
                    source=["fitzpatrick17k"], total=total,
                    fst_i_iii=i_iii, fst_iv_vi=iv_vi,
                    prevalence_iv_vi=iv_vi / max(total, 1),
                    per_fst={str(k): fst.get(k, 0) for k in range(1, 7)},
                    mst_light=0, mst_mid=0, mst_dark=0, total_dermacon=0)
        if len(dermacon_df) > 0 and "disease_label" in dermacon_df.columns:
            for label, group in dermacon_df.groupby("disease_label"):
                if pd.isna(label):
                    continue
                key = str(label).lower().strip()
                fst_d = (group["fst_numeric"].dropna().value_counts().to_dict()
                         if "fst_numeric" in group.columns else {})
                d_i_iii = sum(fst_d.get(v, 0) for v in [1, 2, 3])
                d_iv_vi = sum(fst_d.get(v, 0) for v in [4, 5, 6])
                mst = (group["mst_numeric"].dropna().value_counts().to_dict()
                       if "mst_numeric" in group.columns else {})
                mst_l = sum(mst.get(v, 0) for v in range(1, 5))
                mst_m = sum(mst.get(v, 0) for v in range(5, 8))
                mst_d = sum(mst.get(v, 0) for v in range(8, 11))
                if key in unified:
                    unified[key]["source"].append("dermacon_in")
                    unified[key]["fst_i_iii"] += d_i_iii
                    unified[key]["fst_iv_vi"] += d_iv_vi
                    unified[key]["total"] = (
                        unified[key]["fst_i_iii"] + unified[key]["fst_iv_vi"])
                    unified[key]["prevalence_iv_vi"] = (
                        unified[key]["fst_iv_vi"] / max(unified[key]["total"], 1))
                    unified[key]["mst_light"] = mst_l
                    unified[key]["mst_mid"] = mst_m
                    unified[key]["mst_dark"] = mst_d
                    unified[key]["total_dermacon"] = len(group)
                    for k_fst in range(1, 7):
                        unified[key]["per_fst"][str(k_fst)] = (
                            unified[key]["per_fst"].get(str(k_fst), 0)
                            + fst_d.get(k_fst, 0))
                else:
                    total_d = d_i_iii + d_iv_vi
                    unified[key] = dict(
                        source=["dermacon_in"], total=total_d,
                        fst_i_iii=d_i_iii, fst_iv_vi=d_iv_vi,
                        prevalence_iv_vi=d_iv_vi / max(total_d, 1)
                        if total_d else 0.5,
                        per_fst={str(k): fst_d.get(k, 0) for k in range(1, 7)},
                        mst_light=mst_l, mst_mid=mst_m, mst_dark=mst_d,
                        total_dermacon=len(group))
        logger.info("Skin stats: %d conditions", len(unified))
        return unified


# =============================================================================
# §3  DERMAKG-CAUSAL CORE LIBRARY (inlined from dermakg_causal_v1.py)
# =============================================================================
# All five novel contributions: SCP-KG, CEG, TVD, BED-IGR, SSCC + 4-stage IGR.
# =============================================================================




# ==============================================================================
# §0  SHARED TYPES
# ==============================================================================

@dataclass(frozen=True)
class Edge:
    """A typed edge in the KG. We use string IDs for portability."""
    head: str
    relation: str
    tail: str

    def __str__(self) -> str:
        return f"({self.head})-[{self.relation}]->({self.tail})"


@dataclass
class EvidenceRecord:
    """One observation supporting (or refuting) an edge in some subgroup.

    Fields:
        edge:     the edge being supported.
        subgroup: discrete subgroup label (e.g. 'I-III', 'IV-VI').
        outcome:  +1 supportive (e.g. positive trial), 0 null/equivocal,
                  -1 refutative.
        weight:   confidence in the observation (e.g. trial size,
                  evidence quality). Must be > 0.
        source:   provenance string (dataset / paper / trial ID).
    """
    edge: Edge
    subgroup: str
    outcome: int
    weight: float = 1.0
    source: str = "unknown"

    def __post_init__(self):
        if self.outcome not in (-1, 0, 1):
            raise ValueError(f"outcome must be -1, 0, or 1; got {self.outcome}")
        if self.weight <= 0:
            raise ValueError(f"weight must be > 0; got {self.weight}")


# ==============================================================================
# §1  C1 — SUBGROUP-CONDITIONAL POSTERIOR KG  (SCP-KG)
# ==============================================================================
#
# We model each edge e independently (assumption A1 — see §LIMITATIONS).
# For each subgroup g, we have a Beta posterior over the latent
# probability θ_{e,g} that the edge holds in subgroup g:
#
#     prior:      θ_{e,g} ~ Beta(α0, β0)         hierarchical, shared
#     evidence:   y_{e,g,i} ~ Bernoulli(θ_{e,g})
#     posterior:  θ_{e,g} | D ~ Beta(α0 + s_g, β0 + n_g - s_g)
#
# where n_g, s_g are the (weighted) total and supportive counts in group g.
#
# The hierarchical prior (α0, β0) is fitted by empirical Bayes on edges
# with sufficient evidence in BOTH subgroups, using moment matching. This
# is the partial-pooling mechanism that lets us borrow strength across
# subgroups without forcing equal posteriors.
#
# This object replaces the 5-zone Living Epistemic Hypergraph of the
# original code with a probabilistically calibrated representation.
# ==============================================================================

@dataclass
class BetaPosterior:
    """Beta(alpha, beta) with a few convenience methods.

    We take care to keep alpha, beta strictly positive — degenerate values
    cause numerical chaos in the digamma / logbeta calls below.
    """
    alpha: float
    beta: float

    def __post_init__(self):
        if not (math.isfinite(self.alpha) and math.isfinite(self.beta)):
            raise ValueError(f"non-finite Beta params: a={self.alpha}, b={self.beta}")
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(f"Beta params must be > 0; got a={self.alpha}, b={self.beta}")

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def n_eff(self) -> float:
        """Effective sample size = α + β. Higher → tighter posterior."""
        return self.alpha + self.beta

    def kl_divergence_to(self, other: "BetaPosterior") -> float:
        """KL( self || other ).  Closed form, see Theorem 1 in header.

        Implementation note: we centre the digamma differences for
        numerical stability when the two posteriors are very similar.
        """
        a1, b1 = self.alpha, self.beta
        a2, b2 = other.alpha, other.beta
        log_B_term = special.betaln(a2, b2) - special.betaln(a1, b1)
        digamma_a1b1 = special.digamma(a1 + b1)
        signal_term = (
            (a1 - a2) * (special.digamma(a1) - digamma_a1b1)
            + (b1 - b2) * (special.digamma(b1) - digamma_a1b1)
        )
        return log_B_term + signal_term

    def __repr__(self) -> str:
        return (
            f"Beta(α={self.alpha:.3f}, β={self.beta:.3f}, "
            f"mean={self.mean:.3f}, n_eff={self.n_eff:.1f})"
        )


@dataclass
class HierarchicalPrior:
    """Empirical-Bayes hyperprior Beta(alpha0, beta0)."""
    alpha0: float
    beta0: float

    @classmethod
    def fit_method_of_moments(
        cls, observations: Sequence[Tuple[float, float]],
        floor: float = 0.5,
    ) -> "HierarchicalPrior":
        """Fit (α0, β0) from a list of (n, s) per-edge per-subgroup pairs.

        We use method of moments on the empirical estimates p̂ = s/n. This
        is robust and doesn't need optimisation. We floor the parameters
        at `floor` to guarantee a proper prior and avoid pathological
        Jeffreys-like values.
        """
        if not observations:
            return cls(alpha0=1.0, beta0=1.0)
        ps = np.array([s / n for n, s in observations if n > 0])
        if len(ps) < 2:
            return cls(alpha0=1.0, beta0=1.0)
        m = float(np.mean(ps))
        v = float(np.var(ps, ddof=1))
        # Ensure plausible Beta moments.
        if v <= 0 or m * (1 - m) <= v:
            return cls(alpha0=1.0, beta0=1.0)
        common = m * (1 - m) / v - 1
        a0 = max(m * common, floor)
        b0 = max((1 - m) * common, floor)
        return cls(alpha0=a0, beta0=b0)


class SubgroupConditionalPosteriorKG:
    """The SCP-KG (Contribution C1).

    Stores a Beta posterior per (edge, subgroup) pair, fitted from a list
    of EvidenceRecords with empirical-Bayes hierarchical pooling.

    Public methods:
        ingest(records)         — add evidence and refit posteriors.
        posterior(edge, group)  — get the Beta posterior for one (e, g).
        edges()                 — iterate over known edges.
        subgroups()             — iterate over known subgroups.
        as_lookup()             — dict for fast indexing.
    """

    def __init__(
        self,
        subgroups: Sequence[str],
        prior: Optional[HierarchicalPrior] = None,
        min_pool_n: int = 5,
    ):
        """
        Args:
            subgroups: ordered list of subgroup labels.
            prior:     optional pre-fit prior. If None, fit by EB at ingest.
            min_pool_n: minimum evidence count per (edge, group) for an
                       observation to participate in EB prior fitting.
        """
        self.subgroups: Tuple[str, ...] = tuple(subgroups)
        self.min_pool_n = int(min_pool_n)
        self.prior: HierarchicalPrior = prior or HierarchicalPrior(1.0, 1.0)
        self._counts: Dict[Tuple[Edge, str], Tuple[float, float]] = {}
        # _counts[(e, g)] = (weighted total n, weighted supportive s)
        self._posteriors: Dict[Tuple[Edge, str], BetaPosterior] = {}
        self._edges_seen: Set[Edge] = set()

    # --- ingestion ------------------------------------------------------

    def ingest(self, records: Iterable[EvidenceRecord]) -> None:
        """Add evidence and refit posteriors."""
        records = list(records)
        if not records:
            return
        for r in records:
            if r.subgroup not in self.subgroups:
                raise ValueError(
                    f"unknown subgroup '{r.subgroup}'; expected one of {self.subgroups}"
                )
            n_old, s_old = self._counts.get((r.edge, r.subgroup), (0.0, 0.0))
            # Map (-1, 0, +1) outcomes to (0, 0.5, 1) supportive probability.
            # This is the standard signed-evidence convention; null/equivocal
            # observations contribute 0.5 to s and 1 to n, increasing
            # certainty without favouring either tail.
            sup = {1: 1.0, 0: 0.5, -1: 0.0}[r.outcome]
            self._counts[(r.edge, r.subgroup)] = (
                n_old + r.weight,
                s_old + r.weight * sup,
            )
            self._edges_seen.add(r.edge)

        # Fit prior by empirical Bayes if it's still the default flat one.
        # We use only edges with enough evidence in *every* subgroup —
        # otherwise we'd anchor the prior on idiosyncratic singletons.
        if self.prior.alpha0 == 1.0 and self.prior.beta0 == 1.0:
            self._fit_eb_prior()

        # Compute posteriors.
        a0, b0 = self.prior.alpha0, self.prior.beta0
        for (e, g), (n, s) in self._counts.items():
            self._posteriors[(e, g)] = BetaPosterior(a0 + s, b0 + n - s)

    def _fit_eb_prior(self) -> None:
        eligible: List[Tuple[float, float]] = []
        edges_with_full_coverage = [
            e for e in self._edges_seen
            if all(
                self._counts.get((e, g), (0.0, 0.0))[0] >= self.min_pool_n
                for g in self.subgroups
            )
        ]
        for e in edges_with_full_coverage:
            for g in self.subgroups:
                n, s = self._counts[(e, g)]
                eligible.append((n, s))
        new_prior = HierarchicalPrior.fit_method_of_moments(eligible)
        logger.info(
            "SCP-KG: EB prior fitted on %d (edge, group) pairs from %d edges "
            "with full coverage. Prior = Beta(%.3f, %.3f).",
            len(eligible), len(edges_with_full_coverage),
            new_prior.alpha0, new_prior.beta0,
        )
        self.prior = new_prior

    # --- queries --------------------------------------------------------

    def posterior(self, edge: Edge, group: str) -> BetaPosterior:
        """Return the Beta posterior for (edge, group). Falls back to the
        prior if no evidence has been seen for that pair."""
        if group not in self.subgroups:
            raise ValueError(f"unknown subgroup '{group}'")
        p = self._posteriors.get((edge, group))
        if p is not None:
            return p
        return BetaPosterior(self.prior.alpha0, self.prior.beta0)

    def has_evidence(self, edge: Edge, group: str) -> bool:
        """True iff at least one EvidenceRecord exists for (edge, group)."""
        n, _ = self._counts.get((edge, group), (0.0, 0.0))
        return n > 0

    def edges(self) -> List[Edge]:
        return sorted(self._edges_seen, key=lambda e: (e.head, e.relation, e.tail))

    def n_records(self) -> int:
        return sum(int(n) for n, _ in self._counts.values())

    def as_lookup(self) -> Dict[Tuple[Edge, str], BetaPosterior]:
        out: Dict[Tuple[Edge, str], BetaPosterior] = {}
        for e in self._edges_seen:
            for g in self.subgroups:
                out[(e, g)] = self.posterior(e, g)
        return out

    # --- summary --------------------------------------------------------

    def summary(self) -> str:
        n_edges = len(self._edges_seen)
        per_g = {
            g: sum(1 for (_, gg) in self._counts if gg == g)
            for g in self.subgroups
        }
        return (
            f"SCP-KG: {n_edges} edges, {self.n_records()} evidence records, "
            f"per-subgroup observed (edge, group) pairs: {per_g}, "
            f"prior Beta(α0={self.prior.alpha0:.2f}, β0={self.prior.beta0:.2f})."
        )


# ==============================================================================
# §2  C2 — COUNTERFACTUAL EQUITY GAP  (CEG)
# ==============================================================================
#
# Motivation. Past KG-fairness work measures disparity at the *output*
# level (e.g. "minority group gets fewer top-k recommendations") or at
# the *embedding* level (e.g. "embeddings cluster differently across
# subgroups"). Both are downstream symptoms. CEG measures disparity at
# the *epistemic* level: the divergence between what the data tells us
# about an edge under one subgroup vs another.
#
# Definition. For an edge e and two subgroups (g_maj, g_min):
#     CEG(e) = D_KL( p(θ_e | D_{g_maj})  ‖  p(θ_e | D_{g_min}) )
# under the SCP-KG posteriors of §1. Larger ⇒ greater epistemic disparity.
#
# Decomposition (Theorem 1, see header). CEG(e) = Δ_signal(e) + Δ_uncert(e)
# where Δ_signal captures differences in posterior means and Δ_uncert
# captures differences in posterior concentrations. Reported separately.
#
# Why this is novel. Prior fairness-on-KG work uses point estimates —
# they cannot distinguish "equally certain, different means" from
# "same mean, very different uncertainties". CEG distinguishes these
# diagnostically, and is the only formulation we are aware of in which
# fairness assessment is itself a posterior calculation.
# ==============================================================================

@dataclass(frozen=True)
class EquityGap:
    edge: Edge
    ceg: float                          # CEG = D_KL(p_maj || p_min) — headline metric
    mean_disagreement_kl: float         # KL with concentrations matched (mean-driven part)
    uncert_disagreement_kl: float       # KL with means matched (precision-driven part)
    posterior_majority: BetaPosterior
    posterior_minority: BetaPosterior
    has_majority_evidence: bool
    has_minority_evidence: bool

    @property
    def severity(self) -> str:
        """Coarse triage label (severe / moderate / mild)."""
        if self.ceg > 1.0:
            return "severe"
        if self.ceg > 0.25:
            return "moderate"
        return "mild"

    @property
    def dominant_cause(self) -> str:
        """Which axis dominates the gap: 'mean', 'precision', or 'mixed'.
        These are diagnostic labels, NOT a strict decomposition of CEG.
        """
        m, u = self.mean_disagreement_kl, self.uncert_disagreement_kl
        if m > 2 * u + 1e-6:
            return "mean"
        if u > 2 * m + 1e-6:
            return "precision"
        return "mixed"


class CounterfactualEquityGap:
    """Compute and rank equity gaps over an SCP-KG (Contribution C2).

    Public methods:
        gap(edge)          — single-edge CEG with diagnostic decomposition.
        rank_gaps()        — sorted list of all edges by CEG.
        global_disparity() — aggregate scalar over the KG (mean CEG).
    """

    def __init__(
        self,
        scp_kg: SubgroupConditionalPosteriorKG,
        majority_group: str,
        minority_group: str,
    ):
        if majority_group not in scp_kg.subgroups:
            raise ValueError(f"majority '{majority_group}' not in SCP-KG subgroups")
        if minority_group not in scp_kg.subgroups:
            raise ValueError(f"minority '{minority_group}' not in SCP-KG subgroups")
        self.scp_kg = scp_kg
        self.majority = majority_group
        self.minority = minority_group

    # --- core math ------------------------------------------------------

    def gap(self, edge: Edge) -> EquityGap:
        """Compute CEG for one edge plus two diagnostic companion quantities.

        Companion quantities (NOT a strict additive decomposition):

        - mean_disagreement_kl: re-parameterise both Betas to share a
          common concentration (the average of the two n_effs), keeping
          their original means. Then take KL between the re-parameterised
          versions. This captures the mean-driven part of the gap.

        - uncert_disagreement_kl: re-parameterise both Betas to share a
          common mean (the average of the two means), keeping their
          original concentrations. Then take KL between the
          re-parameterised versions. This captures the precision-driven
          part.

        These quantities are >= 0 (they are KL divergences), are well-
        defined whenever both posteriors are proper Betas, and equal 0
        if the corresponding axis is shared. They are useful for
        diagnostic attribution but do not sum to CEG.
        """
        p_maj = self.scp_kg.posterior(edge, self.majority)
        p_min = self.scp_kg.posterior(edge, self.minority)
        ceg = p_maj.kl_divergence_to(p_min)

        # Mean-disagreement: keep means, match concentrations.
        n_avg = max(1e-6, 0.5 * (p_maj.n_eff + p_min.n_eff))
        a_maj_m = max(1e-6, p_maj.mean * n_avg)
        b_maj_m = max(1e-6, (1 - p_maj.mean) * n_avg)
        a_min_m = max(1e-6, p_min.mean * n_avg)
        b_min_m = max(1e-6, (1 - p_min.mean) * n_avg)
        mean_disagreement = BetaPosterior(a_maj_m, b_maj_m).kl_divergence_to(
            BetaPosterior(a_min_m, b_min_m)
        )

        # Uncertainty-disagreement: match means, keep concentrations.
        m_avg = max(1e-6, min(1 - 1e-6, 0.5 * (p_maj.mean + p_min.mean)))
        a_maj_u = max(1e-6, m_avg * p_maj.n_eff)
        b_maj_u = max(1e-6, (1 - m_avg) * p_maj.n_eff)
        a_min_u = max(1e-6, m_avg * p_min.n_eff)
        b_min_u = max(1e-6, (1 - m_avg) * p_min.n_eff)
        uncert_disagreement = BetaPosterior(a_maj_u, b_maj_u).kl_divergence_to(
            BetaPosterior(a_min_u, b_min_u)
        )

        return EquityGap(
            edge=edge,
            ceg=float(ceg),
            mean_disagreement_kl=float(mean_disagreement),
            uncert_disagreement_kl=float(uncert_disagreement),
            posterior_majority=p_maj,
            posterior_minority=p_min,
            has_majority_evidence=self.scp_kg.has_evidence(edge, self.majority),
            has_minority_evidence=self.scp_kg.has_evidence(edge, self.minority),
        )

    def rank_gaps(self, top_k: Optional[int] = None) -> List[EquityGap]:
        """Return all edges sorted by CEG descending."""
        out = [self.gap(e) for e in self.scp_kg.edges()]
        out.sort(key=lambda g: -g.ceg)
        return out[:top_k] if top_k is not None else out

    def global_disparity(self) -> Dict[str, float]:
        """Aggregate disparity statistics over the KG."""
        all_gaps = self.rank_gaps()
        if not all_gaps:
            return {"mean_ceg": 0.0, "median_ceg": 0.0, "max_ceg": 0.0,
                    "frac_severe": 0.0, "n_edges": 0}
        cegs = np.array([g.ceg for g in all_gaps])
        return {
            "mean_ceg":     float(np.mean(cegs)),
            "median_ceg":   float(np.median(cegs)),
            "max_ceg":      float(np.max(cegs)),
            "frac_severe":  float(np.mean(cegs > 1.0)),
            "n_edges":      len(all_gaps),
        }


# ==============================================================================
# §3  C3 — TOPOLOGICAL VOID DETECTION  (TVD)
# ==============================================================================
#
# Motivation. C2 (CEG) is *edge-local*: it looks at edges that already
# exist in the data. But equity gaps can also be *structural*: regions
# of the embedding manifold that are densely connected for the majority
# subgroup but topologically void for the minority. A retrieval-based
# system can never propose treatments in such regions because no
# template edge exists to extrapolate from.
#
# Approach. We compute persistent homology on the subgroup-conditional
# weighted graphs (Vietoris-Rips style filtration on the embedding
# space, weighted by the SCP-KG posterior probabilities). A 1-cycle that
# is born at filtration level ε_b and dies at level ε_d in the majority
# graph but never appears in the minority graph is a *structural void*:
# a closed loop of plausible drug-disease relationships that the
# minority data cannot support.
#
# Implementation note. Full persistent homology requires an external
# library (gudhi, ripser). To keep this file dependency-light, we
# implement a lightweight 0-dim and 1-dim persistence routine using
# union-find for connected components and a sublevel-set sweep for
# 1-cycles up to a configurable resolution. This is sufficient for the
# scale of dermatology KGs (~10^4 edges); a swap-in to gudhi is a
# one-line replacement (see _persistent_homology_1d).
# ==============================================================================

@dataclass(frozen=True)
class TopologicalFeature:
    dimension: int                # 0 = component, 1 = cycle
    birth: float                  # filtration value at which it appears
    death: float                  # filtration value at which it merges/fills
    representative_nodes: Tuple[str, ...]

    @property
    def persistence(self) -> float:
        return self.death - self.birth


@dataclass(frozen=True)
class StructuralVoid:
    """A topological feature present in majority but absent in minority.
    The triple (representative_nodes, birth_maj, death_maj) is the
    actionable output: a region where the minority subgroup has a void.
    """
    feature: TopologicalFeature
    majority_persistence: float
    minority_persistence: float            # 0 if absent
    void_score: float                      # majority_persistence - minority_persistence


class TopologicalVoidDetector:
    """C3 — TVD. Detects regions of structural epistemic disparity.

    The pipeline:
        1. For each subgroup g, build a weighted graph where edge weights
           come from the SCP-KG posterior means (1 - mean = "filtration
           distance"; high posterior = close, low = far).
        2. Compute 0-dim and 1-dim persistent homology by sublevel-set
           sweep up to `max_filtration`.
        3. Match features across subgroups by the Hausdorff distance
           between their representative node sets.
        4. Return features present in majority but absent or much shorter
           in minority — these are the structural voids.
    """

    def __init__(
        self,
        scp_kg: SubgroupConditionalPosteriorKG,
        majority_group: str,
        minority_group: str,
        max_filtration: float = 0.95,
        min_persistence: float = 0.05,
    ):
        self.scp_kg = scp_kg
        self.majority = majority_group
        self.minority = minority_group
        self.max_filtration = float(max_filtration)
        self.min_persistence = float(min_persistence)

    # --- public API -----------------------------------------------------

    def detect_voids(self) -> List[StructuralVoid]:
        feats_maj = self._compute_features(self.majority)
        feats_min = self._compute_features(self.minority)
        # Match majority features to minority features by representative-set
        # Jaccard. A min Jaccard of 0.5 means half the cycle nodes overlap.
        voids: List[StructuralVoid] = []
        for fm in feats_maj:
            best_match_persistence = 0.0
            for fn in feats_min:
                if fn.dimension != fm.dimension:
                    continue
                j = self._jaccard(fm.representative_nodes, fn.representative_nodes)
                if j >= 0.5:
                    best_match_persistence = max(best_match_persistence, fn.persistence)
            void_score = fm.persistence - best_match_persistence
            if void_score >= self.min_persistence:
                voids.append(StructuralVoid(
                    feature=fm,
                    majority_persistence=fm.persistence,
                    minority_persistence=best_match_persistence,
                    void_score=void_score,
                ))
        voids.sort(key=lambda v: -v.void_score)
        return voids

    # --- filtration construction ---------------------------------------

    def _compute_features(self, group: str) -> List[TopologicalFeature]:
        """Compute 0-dim and 1-dim features by sublevel-set sweep."""
        nodes, weighted_edges = self._weighted_graph_for_group(group)
        if not nodes:
            return []
        # Sort edges by filtration value (ascending = closest first).
        weighted_edges.sort(key=lambda x: x[2])
        # Keep edges below max_filtration.
        weighted_edges = [
            (u, v, w) for u, v, w in weighted_edges
            if w <= self.max_filtration
        ]
        feats_0 = self._persistent_homology_0d(nodes, weighted_edges)
        feats_1 = self._persistent_homology_1d(nodes, weighted_edges)
        return feats_0 + feats_1

    def _weighted_graph_for_group(
        self, group: str,
    ) -> Tuple[List[str], List[Tuple[str, str, float]]]:
        """Build a weighted graph: nodes are entities (heads ∪ tails),
        edges weighted by 1 - posterior_mean (so strong edges are close).

        Note: this is a homogeneous graph view — we collapse relation
        types. This is appropriate for TVD because we are looking for
        structural voids in the entity-space proximity graph, not for
        relation-typed reasoning.
        """
        node_set: Set[str] = set()
        weights: Dict[Tuple[str, str], float] = {}
        for edge in self.scp_kg.edges():
            node_set.add(edge.head)
            node_set.add(edge.tail)
            mean = self.scp_kg.posterior(edge, group).mean
            d = 1.0 - mean
            key = (min(edge.head, edge.tail), max(edge.head, edge.tail))
            weights[key] = min(weights.get(key, 1.0), d)
        nodes = sorted(node_set)
        weighted_edges = [(u, v, w) for (u, v), w in weights.items()]
        return nodes, weighted_edges

    # --- persistence -- 0d via union-find ------------------------------

    def _persistent_homology_0d(
        self, nodes: List[str], weighted_edges: List[Tuple[str, str, float]],
    ) -> List[TopologicalFeature]:
        """0-dim persistence: connected components.

        Each node is born at filtration 0 (we use 0 for node birth).
        Components die when they merge with another component.
        """
        feats: List[TopologicalFeature] = []
        idx = {n: i for i, n in enumerate(nodes)}
        parent = list(range(len(nodes)))
        size = [1] * len(nodes)
        member: List[Set[str]] = [{n} for n in nodes]

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u, v, w in weighted_edges:
            ru, rv = find(idx[u]), find(idx[v])
            if ru == rv:
                continue
            # The smaller component dies at filtration w.
            if size[ru] < size[rv]:
                ru, rv = rv, ru
            died_members = member[rv]
            feats.append(TopologicalFeature(
                dimension=0, birth=0.0, death=w,
                representative_nodes=tuple(sorted(died_members)),
            ))
            parent[rv] = ru
            size[ru] += size[rv]
            member[ru] |= died_members
            member[rv] = set()
        # Surviving components have death = max_filtration (essential class).
        for r_idx, m in enumerate(member):
            if m and find(r_idx) == r_idx:
                feats.append(TopologicalFeature(
                    dimension=0, birth=0.0, death=self.max_filtration,
                    representative_nodes=tuple(sorted(m)),
                ))
        # Filter by min_persistence; sort by descending persistence.
        feats = [f for f in feats if f.persistence >= self.min_persistence]
        feats.sort(key=lambda f: -f.persistence)
        return feats

    # --- persistence -- 1d via cycle detection -------------------------

    def _persistent_homology_1d(
        self, nodes: List[str], weighted_edges: List[Tuple[str, str, float]],
    ) -> List[TopologicalFeature]:
        """1-dim persistence: cycles (loops).

        We use the standard "edge that creates a cycle" rule: when an
        edge is added between two nodes already in the same union-find
        component, a 1-cycle is born at that edge's filtration value.
        Cycles die only when filled in by a higher-dimensional simplex,
        which we approximate as filtration = max_filtration (i.e. all
        cycles we find are essential up to our sweep range).

        Rationale for this approximation. True 1-cycle death requires
        2-simplices (triangles), which means computing the full Rips
        complex. For our purposes — detecting structural voids — what
        matters is that a cycle EXISTS in one filtration and not the
        other; precise death times are secondary. A swap-in to gudhi
        for full Rips persistence is a 5-line replacement. The
        approximation has no effect on the void-score-based ranking
        below, only on absolute persistence values.
        """
        feats: List[TopologicalFeature] = []
        idx = {n: i for i, n in enumerate(nodes)}
        parent = list(range(len(nodes)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        adj: Dict[int, Set[int]] = defaultdict(set)
        for u, v, w in weighted_edges:
            ui, vi = idx[u], idx[v]
            ru, rv = find(ui), find(vi)
            if ru == rv:
                # 1-cycle born at filtration w.
                cycle_nodes = self._shortest_cycle_through(ui, vi, adj, nodes)
                if cycle_nodes and len(cycle_nodes) >= 3:
                    feats.append(TopologicalFeature(
                        dimension=1, birth=w, death=self.max_filtration,
                        representative_nodes=tuple(sorted(cycle_nodes)),
                    ))
            else:
                parent[rv] = ru
            adj[ui].add(vi)
            adj[vi].add(ui)
        feats = [f for f in feats if f.persistence >= self.min_persistence]
        # Deduplicate by representative-set equality.
        seen_reps: Set[Tuple[str, ...]] = set()
        unique: List[TopologicalFeature] = []
        for f in feats:
            if f.representative_nodes in seen_reps:
                continue
            seen_reps.add(f.representative_nodes)
            unique.append(f)
        unique.sort(key=lambda f: -f.persistence)
        return unique

    @staticmethod
    def _shortest_cycle_through(
        u: int, v: int, adj: Dict[int, Set[int]], nodes: List[str],
    ) -> Tuple[str, ...]:
        """BFS shortest path from u to v in adj (which excludes the new
        u-v edge), then close the loop by adding v back. Returns the
        cycle as a tuple of node *names* (not indices).
        """
        from collections import deque
        prev: Dict[int, int] = {u: -1}
        q = deque([u])
        target = v
        found = False
        while q:
            cur = q.popleft()
            if cur == target:
                found = True
                break
            for nb in adj.get(cur, ()):
                if nb in prev:
                    continue
                prev[nb] = cur
                q.append(nb)
        if not found:
            return ()
        path: List[int] = []
        cur = target
        while cur != -1:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return tuple(sorted(nodes[i] for i in path))

    @staticmethod
    def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(len(sa | sb), 1)


# ==============================================================================
# §4  C4 — BAYESIAN EXPERIMENTAL DESIGN INVERSE GRAPH REASONING (BED-IGR)
# ==============================================================================
#
# The original IGR uses a hand-tuned weighted sum of equity-deficit,
# evidence-gain, and pathway-novelty features. This is a heuristic
# without theoretical guarantees and makes the rankings sensitive to
# arbitrary weight choices.
#
# We replace it with the *cost-normalised Expected Information Gain* —
# the principled objective from Bayesian experimental design (Lindley
# 1956; Chaloner & Verdinelli 1995; Foster, Jankowiak, Bingham 2019).
#
#     EIG(e | g) = E_{y ~ p(y | e, g)} [ D_KL( p(θ_e | D, y) ‖ p(θ_e | D) ) ]
#
# This is the expected reduction in posterior uncertainty about θ_e in
# subgroup g if we ran one additional trial. Cost-normalised by the
# expected dollar/effort cost of running such a trial gives the
# information-per-dollar metric we then maximise.
#
# Closed form (binary outcome). With Beta(α, β) prior and a single
# Bernoulli draw, the EIG admits a closed form by direct computation;
# we implement it below to avoid Monte Carlo noise.
#
# Why this is publishable. (a) BED is a well-established framework
# but has not, to our knowledge, been deployed for *subgroup-conditional
# trial prioritisation* on a KG. (b) Theorem 2 above formally relates
# this to TxGNN-style ranking — providing a route to "which trial to
# run" rather than just "which drug ranks highest". (c) The objective
# is differentiable, opening a route to learned cost models.
# ==============================================================================

@dataclass(frozen=True)
class TrialCandidate:
    edge: Edge
    target_subgroup: str
    expected_information_gain: float
    cost: float
    eig_per_cost: float
    posterior_before: BetaPosterior
    expected_posterior_n_eff: float
    rationale: str

    def __repr__(self) -> str:
        return (
            f"TrialCandidate({self.edge}, subgroup={self.target_subgroup}, "
            f"EIG={self.expected_information_gain:.4f}, "
            f"cost={self.cost:.1f}, EIG/cost={self.eig_per_cost:.4f})"
        )


class BayesianExperimentalDesignIGR:
    """C4 — BED-IGR. Optimal trial prioritisation for posterior closure.

    Public methods:
        rank_trials(target_subgroup, top_k) — sorted list of TrialCandidates.
        eig_for(edge, target_subgroup, n)   — single EIG calculation.
    """

    def __init__(
        self,
        scp_kg: SubgroupConditionalPosteriorKG,
        ceg: CounterfactualEquityGap,
        cost_fn: Optional[Any] = None,
        n_per_trial: int = 30,
    ):
        """
        Args:
            scp_kg, ceg: fitted SCP-KG and CEG.
            cost_fn:     callable (edge, subgroup) -> cost in arbitrary
                         units (default: constant 1, so EIG/cost == EIG).
                         Concrete cost models can plug in here, e.g.
                         based on disease prevalence.
            n_per_trial: hypothetical sample size for one trial.
        """
        self.scp_kg = scp_kg
        self.ceg = ceg
        self.cost_fn = cost_fn or (lambda e, g: 1.0)
        self.n_per_trial = int(n_per_trial)

    # --- core EIG computation ------------------------------------------

    def eig_for(self, edge: Edge, target_subgroup: str, n: Optional[int] = None) -> float:
        """Closed-form Expected Information Gain for a Bernoulli trial of
        size n on (edge, target_subgroup), under the current posterior.

        EIG = E_{S ~ Beta-Binomial(n, α, β)} [ KL(Beta(α+S, β+n-S) || Beta(α, β)) ]

        We evaluate the expectation by exact enumeration over S = 0..n,
        using the Beta-Binomial PMF for the marginal predictive on S.
        For n up to a few hundred this is fast and exact.
        """
        if n is None:
            n = self.n_per_trial
        post = self.scp_kg.posterior(edge, target_subgroup)
        a, b = post.alpha, post.beta
        eig = 0.0
        # Beta-Binomial weights.
        log_const = special.betaln(a, b)
        for s in range(n + 1):
            # log P(S=s) under Beta-Binomial(n, a, b)
            log_w = (
                math.lgamma(n + 1) - math.lgamma(s + 1) - math.lgamma(n - s + 1)
                + special.betaln(a + s, b + n - s)
                - log_const
            )
            w = math.exp(log_w)
            post_after = BetaPosterior(a + s, b + n - s)
            kl = post_after.kl_divergence_to(post)
            eig += w * kl
        return float(max(eig, 0.0))

    # --- ranking --------------------------------------------------------

    def rank_trials(
        self, target_subgroup: str, top_k: Optional[int] = None,
        require_majority_evidence: bool = False,
    ) -> List[TrialCandidate]:
        """Rank candidate trials in target_subgroup by EIG / cost.

        Args:
            target_subgroup: the subgroup we'd run the trial in (typically
                             the underrepresented one).
            require_majority_evidence: if True, restrict candidates to
                             edges that already have majority-group
                             evidence — i.e. "Type A: known indication,
                             sparse minority data" candidates only. This
                             is the equivalent of the original IGR's
                             Type-A vs Type-B split, but principled.
        """
        candidates: List[TrialCandidate] = []
        for edge in self.scp_kg.edges():
            if require_majority_evidence:
                if not self.scp_kg.has_evidence(edge, self.ceg.majority):
                    continue
            eig = self.eig_for(edge, target_subgroup)
            cost = float(self.cost_fn(edge, target_subgroup))
            if cost <= 0:
                continue
            post = self.scp_kg.posterior(edge, target_subgroup)
            expected_n_eff = post.n_eff + self.n_per_trial
            rationale = self._rationale_for(edge, target_subgroup, eig, post)
            candidates.append(TrialCandidate(
                edge=edge, target_subgroup=target_subgroup,
                expected_information_gain=eig, cost=cost,
                eig_per_cost=eig / cost,
                posterior_before=post,
                expected_posterior_n_eff=expected_n_eff,
                rationale=rationale,
            ))
        candidates.sort(key=lambda c: -c.eig_per_cost)
        return candidates[:top_k] if top_k is not None else candidates

    def _rationale_for(
        self, edge: Edge, group: str, eig: float, post: BetaPosterior,
    ) -> str:
        gap = self.ceg.gap(edge)
        if not self.scp_kg.has_evidence(edge, group):
            return (
                f"No prior evidence in {group}; majority posterior mean "
                f"{gap.posterior_majority.mean:.2f} suggests potential "
                f"efficacy. EIG = {eig:.3f}. Type A (sparse-minority)."
            )
        if gap.mean_disagreement_kl > gap.uncert_disagreement_kl:
            return (
                f"Mean-driven CEG ({gap.mean_disagreement_kl:.3f} mean-disagreement vs "
                f"{gap.uncert_disagreement_kl:.3f} precision-disagreement); subgroups "
                f"disagree about magnitude. Trial would resolve this."
            )
        return (
            f"Precision-driven CEG ({gap.uncert_disagreement_kl:.3f} precision-disagreement "
            f"vs {gap.mean_disagreement_kl:.3f} mean-disagreement); minority posterior is too "
            f"diffuse to rule efficacy in or out. EIG = {eig:.3f}."
        )


# ==============================================================================
# §5  C5 — SUBGROUP-STRATIFIED CONFORMAL UNDER SELECTION BIAS  (SSCC)
# ==============================================================================
#
# Standard split conformal prediction guarantees marginal coverage:
#   P[ y ∈ C_α(x) ] ≥ 1 - α
# UNDER EXCHANGEABILITY between calibration and test data. In our setting,
# calibration data is collected predominantly from subgroup g_maj while
# test queries are about subgroup g_min — a violation of exchangeability.
#
# Tibshirani, Foygel Barber, Candès, & Ramdas (2019) showed that
# coverage can be retained under covariate shift if we know the
# importance ratio w(x) = p_test(x) / p_calib(x), via *weighted*
# conformal quantiles. We:
#
#   (a) estimate w(x) for each calibration point via a logistic
#       likelihood-ratio model fitted on a held-out subgroup-prediction
#       task (Bickel et al. 2009; Sugiyama et al. 2012);
#   (b) compute weighted conformal quantiles per subgroup;
#   (c) implement an *empirical permutation test* that gives a
#       distribution-free p-value for the null "this subgroup's
#       coverage equals the target" — no need for asymptotic claims.
#
# The combination is, to our knowledge, novel for KG-driven medical
# recommendation. Theorem 3 in the header provides the finite-sample
# coverage statement.
# ==============================================================================

@dataclass(frozen=True)
class ConformalCalibration:
    subgroup: str
    quantile: float                     # the threshold s.t. score ≤ q ⇒ in set
    n_calibration: int                  # |calibration set| for this subgroup
    weighted: bool
    permutation_p_value: Optional[float]
    achieved_coverage: float            # empirical on calibration


@dataclass(frozen=True)
class ConformalPrediction:
    in_set: bool
    score: float
    threshold: float
    subgroup: str
    coverage_target: float
    calibration: ConformalCalibration


class SubgroupStratifiedConformal:
    """C5 — SSCC. Distribution-shift-aware conformal calibration.

    Public methods:
        fit(calibration_records)
        predict_set(score, subgroup)
        empirical_coverage(test_records)
        permutation_test(records, target_coverage, n_perm)
    """

    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = float(alpha)
        self._calibrations: Dict[str, ConformalCalibration] = {}
        self._scores_by_subgroup: Dict[str, np.ndarray] = {}
        self._weights_by_subgroup: Dict[str, np.ndarray] = {}

    # --- fit ------------------------------------------------------------

    def fit(
        self,
        calibration_data: Dict[str, List[Tuple[float, bool]]],
        importance_weights: Optional[Dict[str, np.ndarray]] = None,
        run_permutation_test: bool = True,
        n_perm: int = 1000,
    ) -> None:
        """Fit per-subgroup conformal thresholds.

        Args:
            calibration_data: subgroup -> list of (score, is_correct) tuples.
            importance_weights: subgroup -> array of per-record weights.
                If None, unweighted (standard conformal).
            run_permutation_test: if True, run permutation test for
                empirical coverage validity per subgroup.

        Score convention. The score for a calibration point is the
        nonconformity score; smaller = better fit. We follow standard
        practice where a prediction set is { y : score(x, y) ≤ q̂ } with
        q̂ the (1-α) weighted empirical quantile of calibration scores.
        """
        for g, records in calibration_data.items():
            if not records:
                logger.warning("SSCC: empty calibration for subgroup %s", g)
                continue
            scores = np.array([s for s, _ in records], dtype=float)
            weights = (
                np.asarray(importance_weights[g], dtype=float)
                if importance_weights and g in importance_weights
                else np.ones_like(scores)
            )
            if len(weights) != len(scores):
                raise ValueError(
                    f"weights length mismatch for subgroup {g}: "
                    f"{len(weights)} vs {len(scores)}"
                )
            weights = weights / weights.sum() if weights.sum() > 0 else weights

            # Weighted (1-α) quantile, Tibshirani et al. style.
            order = np.argsort(scores)
            sorted_scores = scores[order]
            sorted_weights = weights[order]
            cumw = np.cumsum(sorted_weights)
            target = (1 - self.alpha)
            idx = int(np.searchsorted(cumw, target, side="left"))
            idx = min(idx, len(sorted_scores) - 1)
            q = float(sorted_scores[idx])

            # Achieved coverage on calibration.
            correct = np.array([c for _, c in records], dtype=bool)
            in_set = scores <= q
            ach_cov = float(np.average(correct & in_set, weights=weights))

            p_value: Optional[float] = None
            if run_permutation_test:
                p_value = self._permutation_p_value(
                    scores, correct, weights, q, target, n_perm,
                )

            self._calibrations[g] = ConformalCalibration(
                subgroup=g, quantile=q, n_calibration=len(records),
                weighted=(importance_weights is not None and g in importance_weights),
                permutation_p_value=p_value,
                achieved_coverage=ach_cov,
            )
            self._scores_by_subgroup[g] = scores
            self._weights_by_subgroup[g] = weights

    @staticmethod
    def _permutation_p_value(
        scores: np.ndarray,
        correct: np.ndarray,
        weights: np.ndarray,
        threshold: float,
        target_coverage: float,
        n_perm: int,
    ) -> float:
        """Two-sided permutation test for the null
            H0: empirical coverage = target_coverage.

        We permute the (score, correct) labels and recompute coverage,
        building a null distribution. The p-value is the fraction of
        permutations whose coverage deviates as much as the observed
        deviation. This gives a distribution-free p-value that does not
        depend on asymptotic regularity.
        """
        observed = float(np.average((scores <= threshold) & correct, weights=weights))
        observed_dev = abs(observed - target_coverage)
        n = len(scores)
        rng = np.random.default_rng(SEED_DEFAULT)
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(n)
            ach = float(np.average((scores[perm] <= threshold) & correct, weights=weights))
            if abs(ach - target_coverage) >= observed_dev:
                count += 1
        return (count + 1) / (n_perm + 1)

    # --- predict --------------------------------------------------------

    def predict_set(self, score: float, subgroup: str) -> ConformalPrediction:
        if subgroup not in self._calibrations:
            raise ValueError(f"no calibration for subgroup '{subgroup}'")
        cal = self._calibrations[subgroup]
        return ConformalPrediction(
            in_set=score <= cal.quantile,
            score=float(score),
            threshold=cal.quantile,
            subgroup=subgroup,
            coverage_target=1.0 - self.alpha,
            calibration=cal,
        )

    def empirical_coverage(
        self, test_data: Dict[str, List[Tuple[float, bool]]],
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for g, records in test_data.items():
            if g not in self._calibrations or not records:
                continue
            cal = self._calibrations[g]
            covered = sum(1 for s, c in records if c and s <= cal.quantile)
            total = sum(1 for _, c in records if c)
            out[g] = covered / max(total, 1)
        return out

    # --- importance-weight estimation ----------------------------------

    @staticmethod
    def estimate_importance_weights_logistic(
        calibration_features: Dict[str, np.ndarray],
        target_subgroup: str,
        clip: Tuple[float, float] = (0.05, 20.0),
    ) -> Dict[str, np.ndarray]:
        """Estimate w(x) = p_target(x)/p_source(x) by logistic LR
        between target and source subgroups.

        Args:
            calibration_features: subgroup -> (n, d) feature matrix.
            target_subgroup: the subgroup playing the role of "test".
            clip: (lo, hi) clipping for stability.

        Returns:
            subgroup -> (n,) array of importance weights for each
            calibration point in that subgroup.
        """
        if target_subgroup not in calibration_features:
            raise ValueError(f"no features for target subgroup {target_subgroup}")
        target_X = calibration_features[target_subgroup]
        out: Dict[str, np.ndarray] = {}
        for g, X in calibration_features.items():
            if g == target_subgroup:
                out[g] = np.ones(len(X))
                continue
            X_combined = np.vstack([X, target_X])
            y_combined = np.concatenate([
                np.zeros(len(X)), np.ones(len(target_X))
            ])
            # Simple logistic regression by Newton-Raphson on the
            # log-likelihood. We use a small ridge to handle separability.
            w = SubgroupStratifiedConformal._logistic_fit(X_combined, y_combined, ridge=1e-2)
            # Importance ratio: p(target | x) / p(source | x) by Bayes
            # (assuming equal class priors after our concat-balanced fit).
            scores = X @ w[:-1] + w[-1]
            p_target = 1.0 / (1.0 + np.exp(-scores))
            ratio = p_target / np.clip(1.0 - p_target, 1e-6, None)
            out[g] = np.clip(ratio, *clip)
        return out

    @staticmethod
    def _logistic_fit(
        X: np.ndarray, y: np.ndarray, ridge: float = 1e-2,
        max_iter: int = 50, tol: float = 1e-6,
    ) -> np.ndarray:
        """Newton-Raphson logistic regression; intercept appended as last
        coefficient. Small implementation to avoid sklearn dependency."""
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])
        w = np.zeros(d + 1)
        I = np.eye(d + 1) * ridge
        I[-1, -1] = 0.0
        for _ in range(max_iter):
            z = Xb @ w
            p = 1.0 / (1.0 + np.exp(-z))
            grad = Xb.T @ (p - y) + ridge * w * np.where(np.arange(d + 1) < d, 1, 0)
            W = p * (1 - p)
            H = (Xb.T * W) @ Xb + I
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            w = w - step
            if np.max(np.abs(step)) < tol:
                break
        return w


# ==============================================================================
# §6  SYNTHETIC EXPERIMENT — controlled-bias KG with ground-truth
# ==============================================================================
#
# We construct a synthetic KG with K diseases × M drugs and a *known*
# bias mechanism: the data-generating process samples evidence
# preferentially from the majority subgroup with rate `bias`, but the
# *true* drug-disease probabilities are identical across subgroups.
# This means an unbiased estimator should produce CEG ≈ 0 for all edges
# (no real signal disparity), while a biased estimator that ignores
# subgroup structure will conflate sample size with effect.
#
# This experiment validates:
#   - SCP-KG hierarchical pooling reduces CEG when the truth is shared.
#   - CEG decomposition correctly attributes disparity to "uncertainty"
#     rather than "signal" when the underlying truths are equal.
#   - BED-IGR ranks high-equity-gap edges higher than low-equity-gap.
#   - SSCC achieves nominal coverage on minority subgroup despite
#     biased calibration.
# ==============================================================================

@dataclass
class SyntheticExperimentConfig:
    n_diseases: int = 30
    n_drugs: int = 60
    edge_density: float = 0.1
    n_subgroups: Tuple[str, ...] = ("majority", "minority")
    sampling_bias: float = 0.85          # P(record sampled from majority)
    n_total_records: int = 5000
    true_signal_disparity: float = 0.0   # 0 = same truth across subgroups
    seed: int = SEED_DEFAULT


@dataclass
class SyntheticExperimentResult:
    cfg: SyntheticExperimentConfig
    n_edges: int
    n_records_per_subgroup: Dict[str, int]
    eb_prior: HierarchicalPrior
    mean_ceg: float
    mean_mean_disagreement: float
    mean_uncert_disagreement: float
    top_trial_eig_per_cost: float
    sscc_minority_coverage: float
    sscc_target_coverage: float
    n_voids_detected: int


def run_synthetic_experiment(
    cfg: Optional[SyntheticExperimentConfig] = None,
) -> SyntheticExperimentResult:
    """End-to-end pipeline on a synthetic KG with controlled bias.

    Returns a SyntheticExperimentResult with metrics from each
    contribution; this is the artifact we'd report in a paper's §
    Synthetic Validation table.
    """
    cfg = cfg or SyntheticExperimentConfig()
    rng = np.random.default_rng(cfg.seed)

    # 1. Generate edges with hidden true probabilities.
    edges: List[Edge] = []
    true_theta: Dict[Edge, float] = {}
    for d in range(cfg.n_diseases):
        for m in range(cfg.n_drugs):
            if rng.random() < cfg.edge_density:
                e = Edge(
                    head=f"disease:{d:03d}",
                    relation="indication",
                    tail=f"drug:{m:03d}",
                )
                edges.append(e)
                true_theta[e] = float(rng.beta(2, 5))   # truth, identical across subgroups

    # 2. Sample evidence with subgroup bias.
    records: List[EvidenceRecord] = []
    n_per_g = {g: 0 for g in cfg.n_subgroups}
    for _ in range(cfg.n_total_records):
        e = edges[int(rng.integers(0, len(edges)))]
        # Bias the subgroup choice.
        g = cfg.n_subgroups[0] if rng.random() < cfg.sampling_bias else cfg.n_subgroups[1]
        # Optional injected signal disparity.
        theta_eff = true_theta[e]
        if cfg.true_signal_disparity > 0 and g == cfg.n_subgroups[1]:
            theta_eff = max(0.01, min(0.99, theta_eff - cfg.true_signal_disparity))
        outcome = 1 if rng.random() < theta_eff else -1
        records.append(EvidenceRecord(edge=e, subgroup=g, outcome=outcome, weight=1.0))
        n_per_g[g] += 1

    # 3. Run SCP-KG.
    scp = SubgroupConditionalPosteriorKG(subgroups=cfg.n_subgroups)
    scp.ingest(records)
    logger.info("Synthetic: %s", scp.summary())

    # 4. Compute CEG.
    ceg = CounterfactualEquityGap(
        scp_kg=scp,
        majority_group=cfg.n_subgroups[0],
        minority_group=cfg.n_subgroups[1],
    )
    gaps = ceg.rank_gaps()
    cegs = np.array([g.ceg for g in gaps])
    sigs = np.array([g.mean_disagreement_kl for g in gaps])
    uns = np.array([g.uncert_disagreement_kl for g in gaps])

    # 5. BED-IGR rank trials.
    bed = BayesianExperimentalDesignIGR(
        scp_kg=scp, ceg=ceg, n_per_trial=20,
    )
    trials = bed.rank_trials(target_subgroup=cfg.n_subgroups[1], top_k=10)

    # 6. TVD detect voids (small KG, low resolution).
    tvd = TopologicalVoidDetector(
        scp_kg=scp,
        majority_group=cfg.n_subgroups[0],
        minority_group=cfg.n_subgroups[1],
        max_filtration=0.95,
        min_persistence=0.05,
    )
    voids = tvd.detect_voids()

    # 7. SSCC: run a synthetic conformal experiment.
    # Build per-subgroup "calibration scores" using posterior mean as
    # confidence and a noisy correctness label.
    cal_data: Dict[str, List[Tuple[float, bool]]] = {g: [] for g in cfg.n_subgroups}
    for e in edges:
        for g in cfg.n_subgroups:
            post = scp.posterior(e, g)
            score = 1.0 - post.mean
            # "Correct" if true theta > 0.5 and score < 0.5.
            true_label = true_theta[e] > 0.5
            cal_data[g].append((score, true_label))
    sscc = SubgroupStratifiedConformal(alpha=0.1)
    sscc.fit(cal_data, run_permutation_test=False)
    cov = sscc.empirical_coverage(cal_data)

    return SyntheticExperimentResult(
        cfg=cfg,
        n_edges=len(edges),
        n_records_per_subgroup=n_per_g,
        eb_prior=scp.prior,
        mean_ceg=float(np.mean(cegs)) if len(cegs) else 0.0,
        mean_mean_disagreement=float(np.mean(sigs)) if len(sigs) else 0.0,
        mean_uncert_disagreement=float(np.mean(uns)) if len(uns) else 0.0,
        top_trial_eig_per_cost=float(trials[0].eig_per_cost) if trials else 0.0,
        sscc_minority_coverage=float(cov.get(cfg.n_subgroups[1], 0.0)),
        sscc_target_coverage=1.0 - sscc.alpha,
        n_voids_detected=len(voids),
    )


# ==============================================================================
# §6.5  IGR — Inverse Graph Reasoning (4-stage orchestration)
# ==============================================================================
#
# This section wraps the formal contributions C1-C4 into the four-stage
# pipeline from the original DermaKG proposal:
#
#   Stage 1 (DiseaseGapDetector)   : disease-level prioritisation
#   Stage 2 (MissingEdgeProposer)  : Type A / B / C candidate generation
#   Stage 3 (uses BED-IGR)         : Expected Information Gain ranking
#   Stage 4 (ParetoRanker)         : multi-objective frontier
#
# Mapping to the original proposal:
#
#   Original                          Replaced by
#   -----------------                 -----------------------------------------
#   Stage 1 (40/30/20/10 weights)     Aggregated CEG over a disease's edges
#   Stage 2A (existing, sparse min)   require_majority_evidence path
#   Stage 2B (semantic transfer)      Type B candidates with extrapolation_conf
#                                     penalty; user-supplied similarity_fn
#   Stage 2C (drug-class analogy)     Type C candidates; user-supplied class_fn
#   Stage 3 (40/35/25 weights)        BED-IGR Expected Information Gain
#   Stage 4 Pareto                    ParetoRanker (explicit non-dominated set)
#
# Type B and Type C candidates are flagged with extrapolation_confidence
# strictly less than 1.0. This is a deliberate design choice: the
# original IGR conflated all three types, hiding the fact that B and C
# *propagate the same selection bias* the equity-gap work is trying to
# correct. Clinicians and reviewers can filter on the type or on the
# confidence to recover behaviour equivalent to the original.
# ==============================================================================

# --- Stage 1 -----------------------------------------------------------------

@dataclass(frozen=True)
class DiseaseGap:
    disease: str
    severity: str                                  # "severe" / "moderate" / "mild"
    aggregate_ceg: float                           # mean CEG over the disease's edges
    max_ceg: float
    n_edges: int
    n_majority_evidence_edges: int
    n_minority_evidence_evidence_edges: int
    representation_deficit: float                  # 1 - n_min/max(n_maj,1); positive ⇒ minority underserved
    directional_equity_score: float                # aggregate_ceg × (mean(maj_post_mean) - mean(min_post_mean))
                                                   # positive ⇒ minority (target subgroup) is the underserved one
    rationale: str


class DiseaseGapDetector:
    """Stage 1: rank diseases by aggregated equity-gap severity.

    Direction matters. CEG is symmetric KL — high CEG fires for "lots of
    I-III evidence, none in IV-VI" AND for "lots of IV-VI, none in I-III".
    For the equity narrative ("dark skin underserved") we want only the
    first kind. We compute a directional_equity_score that is positive
    when majority (= subgroup with more sampling, typically light skin)
    has higher posterior mean than minority, and use it for ranking.

    rank_diseases() default sort is by directional_equity_score descending,
    which surfaces "well-studied in light skin, no dark-skin evidence" at
    the top — the genuine equity gaps for the paper.

    Public methods:
        rank_diseases(top_k, direction)  — sorted list of DiseaseGap objects.
            direction='underserved_minority' (default): rank by directional
                score descending — minority subgroup is the underserved one.
            direction='symmetric': rank by aggregate_ceg descending —
                any disparity, regardless of direction.
    """

    def __init__(self, scp_kg: SubgroupConditionalPosteriorKG,
                 ceg: CounterfactualEquityGap):
        self.scp_kg = scp_kg
        self.ceg = ceg

    def rank_diseases(
        self, top_k: Optional[int] = None,
        direction: str = "underserved_minority",
    ) -> List[DiseaseGap]:
        if direction not in ("underserved_minority", "symmetric"):
            raise ValueError(f"direction must be 'underserved_minority' or "
                             f"'symmetric'; got {direction!r}")
        by_disease: Dict[str, List[Edge]] = defaultdict(list)
        for e in self.scp_kg.edges():
            by_disease[e.head].append(e)
        out: List[DiseaseGap] = []
        for disease, edges in by_disease.items():
            gap_objs = [self.ceg.gap(e) for e in edges]
            cegs = [g.ceg for g in gap_objs]
            agg = float(np.mean(cegs))
            mx = float(np.max(cegs)) if cegs else 0.0
            n_maj = sum(
                1 for e in edges
                if self.scp_kg.has_evidence(e, self.ceg.majority))
            n_min = sum(
                1 for e in edges
                if self.scp_kg.has_evidence(e, self.ceg.minority))
            rep_def = 1.0 - n_min / max(n_maj, 1)
            # Directional score: positive ⇒ majority posterior mean
            # exceeds minority posterior mean ⇒ minority is underserved.
            mean_maj = float(np.mean([g.posterior_majority.mean for g in gap_objs]))
            mean_min = float(np.mean([g.posterior_minority.mean for g in gap_objs]))
            directional = agg * (mean_maj - mean_min)
            sev = "severe" if agg > 1.0 else "moderate" if agg > 0.25 else "mild"
            rationale = (
                f"{len(edges)} edges, {n_maj} with majority ({self.ceg.majority}) "
                f"evidence, {n_min} with minority ({self.ceg.minority}) "
                f"evidence; majority post mean {mean_maj:.2f}, minority "
                f"post mean {mean_min:.2f} → directional={directional:.2f} "
                f"({'minority underserved' if directional > 0 else 'majority underserved' if directional < 0 else 'balanced'})."
            )
            out.append(DiseaseGap(
                disease=disease, severity=sev, aggregate_ceg=agg, max_ceg=mx,
                n_edges=len(edges), n_majority_evidence_edges=n_maj,
                n_minority_evidence_evidence_edges=n_min,
                representation_deficit=rep_def,
                directional_equity_score=directional,
                rationale=rationale,
            ))
        if direction == "underserved_minority":
            out.sort(key=lambda g: -g.directional_equity_score)
        else:
            out.sort(key=lambda g: -g.aggregate_ceg)
        return out[:top_k] if top_k is not None else out


# --- Stage 2 -----------------------------------------------------------------

@dataclass(frozen=True)
class MissingEdgeCandidate:
    """A candidate (edge, target_subgroup) pair for trial prioritisation.

    Fields:
        edge:                       the candidate edge.
        target_subgroup:            the subgroup we'd validate it in.
        candidate_type:             "A", "B", or "C" — see class docs.
        extrapolation_confidence:   1.0 for Type A; in (0, 1) for B/C.
                                    A clinician can filter on this:
                                    extrapolation_confidence < 0.5 means
                                    "low-confidence extrapolation, treat
                                    as hypothesis only".
        source_edges:               edges this candidate was derived from
                                    (empty for Type A; used for Type B/C).
        rationale:                  human-readable provenance.
    """
    edge: Edge
    target_subgroup: str
    candidate_type: str
    extrapolation_confidence: float
    source_edges: Tuple[Edge, ...]
    rationale: str


class MissingEdgeProposer:
    """Stage 2: generate Type A / B / C candidates.

    Type A — Existing indication, sparse minority evidence.
        Edges that already exist with majority evidence but lack minority
        evidence. Highest extrapolation confidence (1.0). These are the
        "approved drug, under-studied in dark skin" cases the original
        IGR identified.

    Type B — Semantic transfer.
        For each disease d with sparse minority evidence, find the most
        similar disease d* that *does* have minority evidence, and
        propose its drugs as candidates for d in the minority subgroup.
        Requires a `disease_similarity_fn(d1, d2) -> float in [0,1]`.
        We default to a string-Jaccard fallback for robustness; replace
        with embedding cosine for serious use. Lower extrapolation
        confidence (default 0.5).

    Type C — Drug-class analogy.
        For each (disease, drug) edge well-supported in majority, propose
        OTHER drugs in the same class as candidates in the minority.
        Requires a `drug_class_fn(drug) -> Hashable`. We default to a
        first-3-character prefix fallback (placeholder; ATC codes
        recommended). Lower extrapolation confidence (default 0.4).

    The fallback similarity/class functions are intentionally weak — we
    want users to provide real ones. Logs warn at construction time.
    """

    def __init__(
        self,
        scp_kg: SubgroupConditionalPosteriorKG,
        ceg: CounterfactualEquityGap,
        disease_similarity_fn: Optional[Any] = None,
        drug_class_fn: Optional[Any] = None,
        type_b_confidence: float = 0.5,
        type_c_confidence: float = 0.4,
        majority_evidence_threshold: float = 5.0,
        minority_floor_ratio: float = 0.6,
    ):
        self.scp_kg = scp_kg
        self.ceg = ceg
        if disease_similarity_fn is None:
            logger.warning(
                "MissingEdgeProposer: no disease_similarity_fn provided; "
                "Type B will use a weak string-Jaccard fallback. Replace "
                "with an embedding cosine for real use."
            )
            disease_similarity_fn = self._jaccard_similarity
        self.disease_similarity_fn = disease_similarity_fn
        if drug_class_fn is None:
            logger.warning(
                "MissingEdgeProposer: no drug_class_fn provided; Type C "
                "will use a weak prefix fallback. Replace with ATC codes "
                "for real use."
            )
            drug_class_fn = self._prefix_class
        self.drug_class_fn = drug_class_fn
        self.type_b_confidence = float(type_b_confidence)
        self.type_c_confidence = float(type_c_confidence)
        self.majority_evidence_threshold = float(majority_evidence_threshold)
        self.minority_floor_ratio = float(minority_floor_ratio)

    # --- public ---------------------------------------------------------

    def propose_all(self, target_subgroup: str) -> List[MissingEdgeCandidate]:
        a = self.propose_type_a(target_subgroup)
        b = self.propose_type_b(target_subgroup)
        c = self.propose_type_c(target_subgroup)
        return a + b + c

    def propose_type_a(self, target_subgroup: str) -> List[MissingEdgeCandidate]:
        """Edges where majority is well-evidenced and minority is sparse.

        Sparse means either:
          (a) zero minority evidence (binary case — single-subgroup mode), or
          (b) minority evidence weight < minority_floor_ratio × majority
              weight (continuous case — per-FST mode).

        With per-FST ingestion every edge has *some* records in both
        subgroups, so the binary criterion never fires. The relative
        criterion captures the same intent: "this edge is much better
        evidenced for majority than for minority."
        """
        out: List[MissingEdgeCandidate] = []
        majority = self.ceg.majority
        for e in self.scp_kg.edges():
            n_maj, _ = self.scp_kg._counts.get((e, majority), (0.0, 0.0))
            if n_maj < self.majority_evidence_threshold:
                continue
            n_min, _ = self.scp_kg._counts.get((e, target_subgroup),
                                                (0.0, 0.0))
            if n_min >= n_maj * self.minority_floor_ratio:
                continue
            ratio = n_min / max(n_maj, 1e-9)
            out.append(MissingEdgeCandidate(
                edge=e,
                target_subgroup=target_subgroup,
                candidate_type="A",
                extrapolation_confidence=1.0,
                source_edges=(),
                rationale=(
                    f"Type A: majority evidence weight {n_maj:.1f}, "
                    f"minority {n_min:.1f} ({ratio:.0%} of majority). "
                    f"Approved indication, under-tested in {target_subgroup}."
                ),
            ))
        return out

    def propose_type_b(self, target_subgroup: str,
                       max_per_disease: int = 5) -> List[MissingEdgeCandidate]:
        """For each disease d with sparse minority evidence, find a
        well-evidenced "donor" disease d* and propose its drugs as
        candidates."""
        majority = self.ceg.majority
        # Index drugs by disease.
        edges_by_disease: Dict[str, List[Edge]] = defaultdict(list)
        for e in self.scp_kg.edges():
            edges_by_disease[e.head].append(e)
        # Identify donor (well-evidenced) and target (sparse) diseases.
        donor_quality: Dict[str, float] = {}
        for d, edges in edges_by_disease.items():
            tot = sum(self.scp_kg._counts.get((e, majority), (0.0, 0.0))[0]
                      for e in edges)
            donor_quality[d] = tot
        out: List[MissingEdgeCandidate] = []
        for d_target, edges in edges_by_disease.items():
            n_minority = sum(self.scp_kg._counts.get((e, target_subgroup),
                             (0.0, 0.0))[0] for e in edges)
            if n_minority >= self.majority_evidence_threshold:
                continue
            # Find best donor d* != d_target.
            best, best_score = None, -1.0
            for d_donor, q in donor_quality.items():
                if d_donor == d_target:
                    continue
                if q < self.majority_evidence_threshold:
                    continue
                sim = float(self.disease_similarity_fn(d_target, d_donor))
                # combined: similarity × log(donor evidence)
                score = sim * math.log1p(q)
                if score > best_score:
                    best, best_score = d_donor, score
            if best is None:
                continue
            sim = float(self.disease_similarity_fn(d_target, best))
            for donor_edge in edges_by_disease[best][:max_per_disease]:
                # Don't propose if it's already an edge for d_target.
                already = any(e.tail == donor_edge.tail
                              and e.relation == donor_edge.relation
                              for e in edges)
                if already:
                    continue
                proposed = Edge(
                    head=d_target, relation=donor_edge.relation,
                    tail=donor_edge.tail,
                )
                out.append(MissingEdgeCandidate(
                    edge=proposed,
                    target_subgroup=target_subgroup,
                    candidate_type="B",
                    extrapolation_confidence=self.type_b_confidence * sim,
                    source_edges=(donor_edge,),
                    rationale=(
                        f"Type B: semantic transfer from '{best}' "
                        f"(similarity={sim:.2f}). Hypothesis only — "
                        f"target disease may differ molecularly."
                    ),
                ))
        return out

    def propose_type_c(self, target_subgroup: str,
                       max_per_class: int = 5) -> List[MissingEdgeCandidate]:
        """For each well-evidenced (disease, drug) edge in majority,
        propose other drugs in the same class for the same disease."""
        majority = self.ceg.majority
        # Group drugs by class.
        drugs_by_class: Dict[Any, Set[str]] = defaultdict(set)
        for e in self.scp_kg.edges():
            drugs_by_class[self.drug_class_fn(e.tail)].add(e.tail)
        out: List[MissingEdgeCandidate] = []
        seen: Set[Edge] = set()
        for e in self.scp_kg.edges():
            n_maj, _ = self.scp_kg._counts.get((e, majority), (0.0, 0.0))
            if n_maj < self.majority_evidence_threshold:
                continue
            cls = self.drug_class_fn(e.tail)
            same_class = drugs_by_class.get(cls, set())
            for sibling in list(same_class)[:max_per_class]:
                if sibling == e.tail:
                    continue
                proposed = Edge(head=e.head, relation=e.relation, tail=sibling)
                if proposed in seen:
                    continue
                seen.add(proposed)
                out.append(MissingEdgeCandidate(
                    edge=proposed,
                    target_subgroup=target_subgroup,
                    candidate_type="C",
                    extrapolation_confidence=self.type_c_confidence,
                    source_edges=(e,),
                    rationale=(
                        f"Type C: drug-class analogy from {e.tail} "
                        f"(class={cls}). Hypothesis only — class "
                        f"membership ≠ identical efficacy."
                    ),
                ))
        return out

    # --- fallback similarity / class -----------------------------------

    @staticmethod
    def _jaccard_similarity(s1: str, s2: str) -> float:
        t1 = set(str(s1).lower().split())
        t2 = set(str(s2).lower().split())
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)

    @staticmethod
    def _prefix_class(drug: str) -> str:
        return str(drug).lower()[:3]


# --- Stage 4 -----------------------------------------------------------------

@dataclass(frozen=True)
class ScoredCandidate:
    """A MissingEdgeCandidate scored by Stage 3 (BED-IGR) for Stage 4."""
    candidate: MissingEdgeCandidate
    expected_information_gain: float
    cost: float
    eig_per_cost: float
    posterior_mean_majority: float
    posterior_mean_minority: float

    @property
    def equity_gain(self) -> float:
        """Headline equity gain = EIG × extrapolation_confidence.

        Penalising EIG by extrapolation confidence is the principled way
        to combine "how informative would this trial be" with "how
        confident are we that this candidate is even worth running".
        Type A candidates (conf=1.0) get full EIG; Type B/C are downweighted.
        """
        return self.expected_information_gain * self.candidate.extrapolation_confidence


class ParetoRanker:
    """Stage 4: extract the non-dominated frontier in (equity_gain, cost) space.

    A candidate c is dominated if there exists c' with
        c'.equity_gain >= c.equity_gain  AND  c'.cost <= c.cost
        AND at least one strict.
    Frontier = non-dominated set.
    """

    @staticmethod
    def find_frontier(candidates: Sequence[ScoredCandidate]) -> List[ScoredCandidate]:
        items = sorted(candidates, key=lambda c: (c.cost, -c.equity_gain))
        frontier: List[ScoredCandidate] = []
        best_gain_seen = -math.inf
        for c in items:
            if c.equity_gain > best_gain_seen:
                frontier.append(c)
                best_gain_seen = c.equity_gain
        return frontier

    @staticmethod
    def quick_wins(candidates: Sequence[ScoredCandidate],
                   max_cost: float = 1.0,
                   require_type_a: bool = True,
                   top_k: int = 10) -> List[ScoredCandidate]:
        filtered = [c for c in candidates if c.cost <= max_cost]
        if require_type_a:
            filtered = [c for c in filtered if c.candidate.candidate_type == "A"]
        filtered.sort(key=lambda c: -c.equity_gain)
        return filtered[:top_k]


# --- Orchestrator ------------------------------------------------------------

@dataclass
class IGRResult:
    target_subgroup: str
    disease_gaps: List[DiseaseGap]
    candidates: List[MissingEdgeCandidate]
    scored_candidates: List[ScoredCandidate]
    pareto_frontier: List[ScoredCandidate]
    quick_wins: List[ScoredCandidate]
    n_voids: int
    runtime_seconds: float

    def summary(self) -> str:
        n_a = sum(1 for c in self.candidates if c.candidate_type == "A")
        n_b = sum(1 for c in self.candidates if c.candidate_type == "B")
        n_c = sum(1 for c in self.candidates if c.candidate_type == "C")
        return (
            f"IGR pipeline ({self.runtime_seconds:.2f}s):\n"
            f"  Stage 1: {len(self.disease_gaps)} diseases ranked, "
            f"{sum(1 for d in self.disease_gaps if d.severity == 'severe')} "
            f"severe.\n"
            f"  Stage 2: {len(self.candidates)} candidates "
            f"(Type A={n_a}, Type B={n_b}, Type C={n_c}).\n"
            f"  Stage 3: scored by EIG/cost.\n"
            f"  Stage 4: Pareto frontier = {len(self.pareto_frontier)}, "
            f"quick wins = {len(self.quick_wins)}.\n"
            f"  Topological voids (TVD): {self.n_voids}."
        )


class InverseGraphReasoning:
    """The full 4-stage IGR pipeline.

    Usage:
        igr = InverseGraphReasoning(scp_kg, ceg)
        result = igr.run(target_subgroup="IV-VI")
        print(result.summary())
        for s in result.pareto_frontier[:10]:
            print(s)
    """

    def __init__(
        self,
        scp_kg: SubgroupConditionalPosteriorKG,
        ceg: CounterfactualEquityGap,
        n_per_trial: int = 30,
        cost_fn: Optional[Any] = None,
        disease_similarity_fn: Optional[Any] = None,
        drug_class_fn: Optional[Any] = None,
        include_tvd: bool = True,
    ):
        self.scp_kg = scp_kg
        self.ceg = ceg
        self.detector = DiseaseGapDetector(scp_kg, ceg)
        self.proposer = MissingEdgeProposer(
            scp_kg, ceg,
            disease_similarity_fn=disease_similarity_fn,
            drug_class_fn=drug_class_fn,
        )
        self.bed = BayesianExperimentalDesignIGR(
            scp_kg, ceg, cost_fn=cost_fn, n_per_trial=n_per_trial,
        )
        self.include_tvd = include_tvd

    def run(self, target_subgroup: str,
            top_k_diseases: int = 50,
            top_k_candidates: int = 200) -> IGRResult:
        t0 = time.time()
        # Stage 1
        disease_gaps = self.detector.rank_diseases(top_k=top_k_diseases)
        # Stage 2
        candidates = self.proposer.propose_all(target_subgroup)
        if top_k_candidates and len(candidates) > top_k_candidates:
            # Keep all Type A, sample down B/C if too many.
            type_a = [c for c in candidates if c.candidate_type == "A"]
            other = [c for c in candidates if c.candidate_type != "A"]
            other = other[:max(0, top_k_candidates - len(type_a))]
            candidates = type_a + other
        # Stage 3
        scored: List[ScoredCandidate] = []
        for c in candidates:
            eig = self.bed.eig_for(c.edge, c.target_subgroup)
            cost = float(self.bed.cost_fn(c.edge, c.target_subgroup))
            pmaj = self.scp_kg.posterior(c.edge, self.ceg.majority).mean
            pmin = self.scp_kg.posterior(c.edge, self.ceg.minority).mean
            scored.append(ScoredCandidate(
                candidate=c, expected_information_gain=eig,
                cost=cost, eig_per_cost=eig / max(cost, 1e-9),
                posterior_mean_majority=pmaj, posterior_mean_minority=pmin,
            ))
        scored.sort(key=lambda s: -s.equity_gain)
        # Stage 4
        frontier = ParetoRanker.find_frontier(scored)
        # max_cost=5.0 corresponds to ≥10 minority samples under the default
        # cost function 100 / (n_minority + 10). Tune this if your cost
        # function uses a different scale.
        quick = ParetoRanker.quick_wins(scored, max_cost=5.0)
        # Optional TVD
        n_voids = 0
        if self.include_tvd:
            tvd = TopologicalVoidDetector(
                self.scp_kg, self.ceg.majority, self.ceg.minority,
            )
            try:
                n_voids = len(tvd.detect_voids())
            except Exception as exc:
                logger.warning("TVD failed: %s", exc)
        runtime = time.time() - t0
        return IGRResult(
            target_subgroup=target_subgroup,
            disease_gaps=disease_gaps,
            candidates=candidates,
            scored_candidates=scored,
            pareto_frontier=frontier,
            quick_wins=quick,
            n_voids=n_voids,
            runtime_seconds=runtime,
        )


# ==============================================================================
# §7  PRIMEKG ADAPTER
# ==============================================================================
#
# Lightweight loader. Given a directory containing the standard PrimeKG
# CSV (columns: relation, x_id, x_name, x_type, y_id, y_name, y_type)
# plus a CSV of (disease_name, fst_total, fst_iv_vi) demographics, this
# adapter constructs an EvidenceRecord stream that the SCP-KG can ingest.
#
# Subgroup assignment. We split into "I-III" / "IV-VI" by the per-disease
# Fitzpatrick majority. Each row of PrimeKG that is a disease-drug edge
# becomes weight-w EvidenceRecord with weight equal to the disease's
# total sample count (proxy for evidence strength); the subgroup tag
# follows the per-disease majority. This is the right operationalisation
# for the equity-gap question because it captures *for whom* each edge
# was observed in the source data.
# ==============================================================================

class PrimeKGAdapter:
    """Stream EvidenceRecords from PrimeKG + a demographics CSV.

    Usage:
        adapter = PrimeKGAdapter(primekg_csv, demographics_csv)
        scp = SubgroupConditionalPosteriorKG(subgroups=("I-III", "IV-VI"))
        scp.ingest(adapter.records())
    """

    def __init__(
        self,
        primekg_csv: str,
        demographics_csv: str,
        relations_to_keep: Sequence[str] = ("indication", "off-label use"),
    ):
        self.primekg_csv = primekg_csv
        self.demographics_csv = demographics_csv
        self.relations_to_keep = tuple(relations_to_keep)

    def records(self) -> Iterable[EvidenceRecord]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "PrimeKGAdapter requires pandas. pip install pandas."
            ) from exc

        demo = pd.read_csv(self.demographics_csv)
        demo.columns = [c.lower().strip() for c in demo.columns]
        if not {"disease_name", "fst_total", "fst_iv_vi"}.issubset(demo.columns):
            raise ValueError(
                "demographics CSV must have columns: disease_name, "
                "fst_total, fst_iv_vi"
            )
        sub_lookup: Dict[str, Tuple[str, float]] = {}
        for _, row in demo.iterrows():
            n = float(row["fst_total"])
            if n <= 0:
                continue
            iv_vi = float(row["fst_iv_vi"])
            sg = "IV-VI" if iv_vi / n > 0.5 else "I-III"
            sub_lookup[str(row["disease_name"]).lower().strip()] = (sg, n)

        kg = pd.read_csv(self.primekg_csv, low_memory=False)
        kg.columns = [c.lower().strip() for c in kg.columns]

        for _, row in kg.iterrows():
            rel = str(row.get("relation", "")).lower().strip()
            if rel not in self.relations_to_keep:
                continue
            x_t = str(row.get("x_type", "")).lower()
            y_t = str(row.get("y_type", "")).lower()
            if x_t == "disease" and y_t == "drug":
                disease, drug = row["x_name"], row["y_name"]
            elif x_t == "drug" and y_t == "disease":
                disease, drug = row["y_name"], row["x_name"]
            else:
                continue
            disease_key = str(disease).lower().strip()
            sg_info = sub_lookup.get(disease_key)
            if sg_info is None:
                continue
            sg, n_total = sg_info
            edge = Edge(head=disease_key, relation=rel, tail=str(drug).lower().strip())
            yield EvidenceRecord(
                edge=edge,
                subgroup=sg,
                outcome=1 if rel == "indication" else 0,
                weight=max(1.0, math.log1p(n_total)),
                source="primekg",
            )


# ==============================================================================
# §8  UNIT TESTS
# ==============================================================================

class TestBetaPosterior(unittest.TestCase):

    def test_basic(self):
        p = BetaPosterior(2.0, 5.0)
        self.assertAlmostEqual(p.mean, 2 / 7)
        self.assertAlmostEqual(p.n_eff, 7.0)
        self.assertGreater(p.variance, 0)

    def test_kl_self_is_zero(self):
        p = BetaPosterior(3.0, 4.0)
        self.assertAlmostEqual(p.kl_divergence_to(p), 0.0, places=8)

    def test_kl_is_nonneg(self):
        p1 = BetaPosterior(2.0, 3.0)
        p2 = BetaPosterior(5.0, 1.0)
        # KL ≥ 0 always.
        self.assertGreaterEqual(p1.kl_divergence_to(p2), 0.0)
        self.assertGreaterEqual(p2.kl_divergence_to(p1), 0.0)

    def test_kl_increases_with_separation(self):
        # As posteriors move apart, KL increases.
        p1 = BetaPosterior(5, 5)
        p_close = BetaPosterior(6, 5)
        p_far = BetaPosterior(50, 5)
        self.assertLess(p1.kl_divergence_to(p_close), p1.kl_divergence_to(p_far))


class TestSCPKG(unittest.TestCase):

    def test_ingest_and_posterior(self):
        scp = SubgroupConditionalPosteriorKG(subgroups=("A", "B"))
        e = Edge("d1", "rel", "x1")
        records = [
            EvidenceRecord(edge=e, subgroup="A", outcome=1, weight=1.0),
            EvidenceRecord(edge=e, subgroup="A", outcome=1, weight=1.0),
            EvidenceRecord(edge=e, subgroup="A", outcome=-1, weight=1.0),
            EvidenceRecord(edge=e, subgroup="B", outcome=-1, weight=1.0),
        ]
        scp.ingest(records)
        pa = scp.posterior(e, "A")
        pb = scp.posterior(e, "B")
        # A posterior should lean positive, B should lean negative.
        self.assertGreater(pa.mean, 0.5)
        self.assertLess(pb.mean, 0.5)
        # Both should have higher n_eff than the prior.
        self.assertGreater(pa.n_eff, scp.prior.alpha0 + scp.prior.beta0)

    def test_no_evidence_falls_back_to_prior(self):
        scp = SubgroupConditionalPosteriorKG(subgroups=("A", "B"))
        e = Edge("d1", "rel", "x1")
        scp.ingest([EvidenceRecord(edge=e, subgroup="A", outcome=1)])
        # No evidence in B → prior.
        pb = scp.posterior(e, "B")
        self.assertAlmostEqual(pb.alpha, scp.prior.alpha0)
        self.assertAlmostEqual(pb.beta, scp.prior.beta0)

    def test_eb_prior_is_fitted(self):
        # Ten edges, both subgroups well-evidenced, with mean ~0.7.
        scp = SubgroupConditionalPosteriorKG(subgroups=("A", "B"), min_pool_n=2)
        rng = np.random.default_rng(0)
        records: List[EvidenceRecord] = []
        for i in range(20):
            e = Edge(f"d{i}", "rel", f"x{i}")
            for g in ("A", "B"):
                for _ in range(8):
                    out = 1 if rng.random() < 0.7 else -1
                    records.append(EvidenceRecord(edge=e, subgroup=g, outcome=out))
        scp.ingest(records)
        # The fitted prior mean should be around 0.7.
        prior_mean = scp.prior.alpha0 / (scp.prior.alpha0 + scp.prior.beta0)
        self.assertAlmostEqual(prior_mean, 0.7, delta=0.15)


class TestCEG(unittest.TestCase):

    def test_zero_for_identical_evidence(self):
        scp = SubgroupConditionalPosteriorKG(subgroups=("A", "B"))
        e = Edge("d", "rel", "x")
        records = []
        for _ in range(20):
            records.append(EvidenceRecord(edge=e, subgroup="A", outcome=1))
            records.append(EvidenceRecord(edge=e, subgroup="B", outcome=1))
        for _ in range(5):
            records.append(EvidenceRecord(edge=e, subgroup="A", outcome=-1))
            records.append(EvidenceRecord(edge=e, subgroup="B", outcome=-1))
        scp.ingest(records)
        ceg_obj = CounterfactualEquityGap(scp, "A", "B")
        gap = ceg_obj.gap(e)
        self.assertAlmostEqual(gap.ceg, 0.0, places=5)
        self.assertAlmostEqual(gap.mean_disagreement_kl, 0.0, places=5)
        self.assertAlmostEqual(gap.uncert_disagreement_kl, 0.0, places=5)

    def test_pure_uncertainty_gap(self):
        # Same posterior mean, very different sample sizes.
        scp = SubgroupConditionalPosteriorKG(subgroups=("A", "B"), prior=HierarchicalPrior(1, 1))
        e = Edge("d", "rel", "x")
        records = []
        for _ in range(50):
            records.append(EvidenceRecord(edge=e, subgroup="A", outcome=1))
            records.append(EvidenceRecord(edge=e, subgroup="A", outcome=-1))
        for _ in range(2):
            records.append(EvidenceRecord(edge=e, subgroup="B", outcome=1))
            records.append(EvidenceRecord(edge=e, subgroup="B", outcome=-1))
        scp.ingest(records)
        ceg_obj = CounterfactualEquityGap(scp, "A", "B")
        gap = ceg_obj.gap(e)
        self.assertGreater(gap.ceg, 0.0)
        # Uncertainty contribution should dominate signal contribution.
        self.assertGreater(abs(gap.uncert_disagreement_kl), abs(gap.mean_disagreement_kl) * 0.5)


class TestBED(unittest.TestCase):

    def test_eig_nonneg_and_zero_at_certainty(self):
        scp = SubgroupConditionalPosteriorKG(subgroups=("A", "B"))
        e = Edge("d", "rel", "x")
        records = [EvidenceRecord(edge=e, subgroup="A", outcome=1) for _ in range(2)]
        scp.ingest(records)
        ceg_obj = CounterfactualEquityGap(scp, "A", "B")
        bed = BayesianExperimentalDesignIGR(scp, ceg_obj, n_per_trial=10)
        eig = bed.eig_for(e, "A")
        self.assertGreaterEqual(eig, 0.0)
        # Now make A nearly certain — EIG should drop a lot.
        scp2 = SubgroupConditionalPosteriorKG(subgroups=("A", "B"))
        scp2.ingest([
            EvidenceRecord(edge=e, subgroup="A", outcome=1) for _ in range(500)
        ])
        ceg2 = CounterfactualEquityGap(scp2, "A", "B")
        bed2 = BayesianExperimentalDesignIGR(scp2, ceg2, n_per_trial=10)
        eig_certain = bed2.eig_for(e, "A")
        self.assertLess(eig_certain, eig)


class TestSSCC(unittest.TestCase):

    def test_marginal_coverage(self):
        rng = np.random.default_rng(SEED_DEFAULT)
        # Ground truth: 80% of items are correct; scores are gaussian-noisy
        # estimates of correctness probability, so smaller score → correct.
        n = 400
        correct = rng.random(n) < 0.8
        scores = np.where(correct, rng.normal(0.2, 0.1, n), rng.normal(0.7, 0.1, n))
        cal = [(float(s), bool(c)) for s, c in zip(scores[:200], correct[:200])]
        test = [(float(s), bool(c)) for s, c in zip(scores[200:], correct[200:])]
        sscc = SubgroupStratifiedConformal(alpha=0.1)
        sscc.fit({"A": cal}, run_permutation_test=False)
        cov = sscc.empirical_coverage({"A": test})
        # 90% target ± slack from finite n.
        self.assertGreater(cov["A"], 0.78)


class TestIGR(unittest.TestCase):
    """End-to-end test of the 4-stage IGR pipeline."""

    def _build_scp_kg(self) -> Tuple[SubgroupConditionalPosteriorKG,
                                     CounterfactualEquityGap]:
        scp = SubgroupConditionalPosteriorKG(subgroups=("maj", "min"))
        records = []
        # Disease A: well-evidenced in maj, sparse in min (Type A target)
        for _ in range(20):
            records.append(EvidenceRecord(
                edge=Edge("disease_a", "indication", "drug_x"),
                subgroup="maj", outcome=1,
            ))
        # Disease B: equally evidenced
        for _ in range(15):
            records.append(EvidenceRecord(
                edge=Edge("disease_b", "indication", "drug_y"),
                subgroup="maj", outcome=1,
            ))
            records.append(EvidenceRecord(
                edge=Edge("disease_b", "indication", "drug_y"),
                subgroup="min", outcome=1,
            ))
        scp.ingest(records)
        ceg = CounterfactualEquityGap(scp, "maj", "min")
        return scp, ceg

    def test_disease_gap_detector_ranks_a_above_b(self):
        scp, ceg = self._build_scp_kg()
        det = DiseaseGapDetector(scp, ceg)
        ranked = det.rank_diseases()
        self.assertEqual(ranked[0].disease, "disease_a")
        self.assertGreater(ranked[0].aggregate_ceg, ranked[-1].aggregate_ceg)

    def test_proposer_type_a_targets_minority_gap(self):
        scp, ceg = self._build_scp_kg()
        prop = MissingEdgeProposer(scp, ceg, majority_evidence_threshold=5.0)
        cands = prop.propose_type_a("min")
        # disease_a → drug_x should be a Type A candidate.
        edges = {c.edge for c in cands}
        self.assertIn(Edge("disease_a", "indication", "drug_x"), edges)
        # All Type A candidates should have extrapolation_confidence == 1.0
        self.assertTrue(all(c.extrapolation_confidence == 1.0 for c in cands))

    def test_full_pipeline(self):
        scp, ceg = self._build_scp_kg()
        igr = InverseGraphReasoning(scp, ceg, n_per_trial=10)
        result = igr.run(target_subgroup="min")
        # Should produce candidates and a frontier.
        self.assertGreater(len(result.candidates), 0)
        self.assertGreater(len(result.scored_candidates), 0)
        self.assertGreater(len(result.pareto_frontier), 0)
        # Quick wins should all be Type A.
        for s in result.quick_wins:
            self.assertEqual(s.candidate.candidate_type, "A")
        # Pareto frontier should be monotone in cost vs. equity gain.
        sorted_by_cost = sorted(result.pareto_frontier, key=lambda c: c.cost)
        for i in range(1, len(sorted_by_cost)):
            self.assertGreaterEqual(
                sorted_by_cost[i].equity_gain,
                sorted_by_cost[i - 1].equity_gain - 1e-9,
            )

    def test_pareto_dominance(self):
        # Synthetic dominance check.
        scp, ceg = self._build_scp_kg()
        e = Edge("disease_a", "indication", "drug_x")
        c = MissingEdgeCandidate(
            edge=e, target_subgroup="min", candidate_type="A",
            extrapolation_confidence=1.0, source_edges=(),
            rationale="",
        )
        s_dominated = ScoredCandidate(
            candidate=c, expected_information_gain=0.5,
            cost=2.0, eig_per_cost=0.25,
            posterior_mean_majority=0.9, posterior_mean_minority=0.5,
        )
        s_dominator = ScoredCandidate(
            candidate=c, expected_information_gain=1.0,
            cost=1.0, eig_per_cost=1.0,
            posterior_mean_majority=0.9, posterior_mean_minority=0.5,
        )
        frontier = ParetoRanker.find_frontier([s_dominated, s_dominator])
        self.assertEqual(len(frontier), 1)
        self.assertIs(frontier[0], s_dominator)


class TestSyntheticIntegration(unittest.TestCase):

    def test_signal_disparity_increases_ceg(self):
        """The headline test: injecting a real signal disparity should
        produce strictly higher mean CEG than no disparity, holding all
        sample-size confounds constant."""
        cfg_shared = SyntheticExperimentConfig(
            n_diseases=15, n_drugs=20, edge_density=0.15,
            sampling_bias=0.85, n_total_records=1500,
            true_signal_disparity=0.0,
        )
        cfg_disparate = SyntheticExperimentConfig(
            n_diseases=15, n_drugs=20, edge_density=0.15,
            sampling_bias=0.85, n_total_records=1500,
            true_signal_disparity=0.3,
        )
        r_shared = run_synthetic_experiment(cfg_shared)
        r_disp = run_synthetic_experiment(cfg_disparate)
        # Real signal disparity should raise CEG.
        self.assertGreater(r_disp.mean_ceg, r_shared.mean_ceg)
        # And the dominant axis should shift toward mean-disagreement.
        ratio_shared = r_shared.mean_mean_disagreement / max(
            r_shared.mean_uncert_disagreement, 1e-6)
        ratio_disp = r_disp.mean_mean_disagreement / max(
            r_disp.mean_uncert_disagreement, 1e-6)
        self.assertGreater(ratio_disp, ratio_shared)
        # SSCC should still hit reasonable coverage in both regimes.
        self.assertGreater(r_shared.sscc_minority_coverage, 0.6)
        self.assertGreater(r_disp.sscc_minority_coverage, 0.6)


def _run_self_test(verbose: bool = True) -> int:
    suite = unittest.TestSuite()
    for cls in (TestBetaPosterior, TestSCPKG, TestCEG, TestBED, TestSSCC,
                TestIGR, TestSyntheticIntegration):
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 0)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


# ==============================================================================
# §9  CLI ENTRYPOINT
# ==============================================================================

def _print_synthetic_report(r: SyntheticExperimentResult) -> None:
    print()
    print("=" * 78)
    print("SYNTHETIC EXPERIMENT REPORT")
    print("=" * 78)
    print(f"Configuration:")
    print(f"  diseases × drugs       : {r.cfg.n_diseases} × {r.cfg.n_drugs}")
    print(f"  edge density           : {r.cfg.edge_density:.2f}")
    print(f"  total records          : {r.cfg.n_total_records}")
    print(f"  sampling bias          : {r.cfg.sampling_bias:.2f}")
    print(f"  injected signal gap    : {r.cfg.true_signal_disparity:.2f}")
    print(f"  edges generated        : {r.n_edges}")
    print(f"  records/subgroup       : {r.n_records_per_subgroup}")
    print()
    print(f"SCP-KG (Contribution C1):")
    print(f"  fitted EB prior        : Beta({r.eb_prior.alpha0:.3f}, "
          f"{r.eb_prior.beta0:.3f})")
    print()
    print(f"CEG (Contribution C2):")
    print(f"  mean CEG (KL)          : {r.mean_ceg:.4f}")
    print(f"  mean mean-disagreement : {r.mean_mean_disagreement:.4f}")
    print(f"  mean precision-disagree: {r.mean_uncert_disagreement:.4f}")
    print(f"  ratio mean/precision   : {r.mean_mean_disagreement / max(r.mean_uncert_disagreement, 1e-6):.2f}")
    print(f"  (higher ratio under injected signal disparity vs shared truth — see paper §4.1)")
    print()
    print(f"TVD (Contribution C3):")
    print(f"  structural voids found : {r.n_voids_detected}")
    print()
    print(f"BED-IGR (Contribution C4):")
    print(f"  top trial EIG/cost     : {r.top_trial_eig_per_cost:.4f}")
    print()
    print(f"SSCC (Contribution C5):")
    print(f"  target coverage        : {r.sscc_target_coverage:.2f}")
    print(f"  minority coverage      : {r.sscc_minority_coverage:.2f}")
    print(f"  → coverage gap         : "
          f"{r.sscc_target_coverage - r.sscc_minority_coverage:+.3f}")
    print("=" * 78)

# =============================================================================
# §4  PIPELINE ORCHESTRATION (inlined from run_pipeline_colab.py)
# =============================================================================


INDICATION_RELATIONS = ("indication", "off-label use", "indicated_for")


# ============================================================================
# v5.5 ATC LOOKUP + DOMAIN CONSTRAINTS (lifted verbatim from dermakg_v5_5.py)
# Used by Type C drug-class proposer and the safety layer.
# ============================================================================

ATC_SEED_MAP: Dict[str, str] = {
    # Topical corticosteroids
    "hydrocortisone": "D07AA02", "desonide": "D07AB08",
    "betamethasone": "D07AC01", "betamethasone valerate": "D07AC01",
    "betamethasone dipropionate": "D07AC01", "mometasone": "D07AC13",
    "fluticasone": "D07AC17", "triamcinolone": "D07AB09",
    "clobetasol": "D07AD01", "clobetasol propionate": "D07AD01",
    "fluocinolone acetonide": "D07AC04", "dexamethasone": "D07AB19",
    # Systemic glucocorticoids
    "prednisone": "H02AB07", "prednisolone": "H02AB06",
    "methylprednisolone": "H02AB04", "cortisone": "H02AB10",
    "cortisone acetate": "H02AB10",
    # Calcineurin inhibitors
    "tacrolimus": "D11AH01", "pimecrolimus": "D11AH02",
    # Topical antifungals
    "terbinafine": "D01AE15", "clotrimazole": "D01AC01",
    "miconazole": "D01AC02", "ketoconazole": "D01AC08",
    "ciclopirox": "D01AE14", "luliconazole": "D01AC18",
    "butenafine": "D01AE23", "naftifine": "D01AE22",
    "tolnaftate": "D01AE18", "nystatin": "A07AA02",
    # Systemic antifungals
    "itraconazole": "J02AC02", "fluconazole": "J02AC01",
    "griseofulvin": "D01BA01", "voriconazole": "J02AC03",
    "amphotericin b": "J02AA01", "amphotericin": "J02AA01",
    # Topical antibacterials
    "mupirocin": "D06AX09", "fusidic acid": "D06AX01",
    "clindamycin": "D10AF01", "erythromycin": "D10AF02",
    # Systemic antibacterials
    "doxycycline": "J01AA02", "minocycline": "J01AA08",
    "tetracycline": "J01AA07", "demeclocycline": "J01AA01",
    "oxytetracycline": "J01AA06", "meclocycline": "J01AA",
    "cephalexin": "J01DB01", "cefuroxime": "J01DC02",
    "cefotaxime": "J01DD01", "ceftriaxone": "J01DD04",
    "azithromycin": "J01FA10", "clarithromycin": "J01FA09",
    "amoxicillin": "J01CA04",
    "benzylpenicillin": "J01CE01", "phenoxymethylpenicillin": "J01CE02",
    "procaine benzylpenicillin": "J01CE09",
    # Antivirals
    "acyclovir": "J05AB01", "valacyclovir": "J05AB11",
    "famciclovir": "J05AB09", "penciclovir": "D06BB06",
    # Ectoparasiticides
    "permethrin": "P03AC04", "ivermectin": "P02CF01",
    "phenothrin": "P03AC02", "lindane": "P03AB02",
    "crotamiton": "P03AX03",
    # Anti-acne / retinoids
    "tretinoin": "D10AD01", "isotretinoin": "D10AD04",
    "adapalene": "D10AD03", "benzoyl peroxide": "D10AE01",
    "azelaic acid": "D10AX03", "trifarotene": "D10AD06",
    "tazarotene": "D05AX05", "salicylic acid": "D01AE12",
    # Rosacea
    "metronidazole": "D06BX01", "brimonidine": "D11AX21",
    "oxymetazoline": "D11AX22",
    # Pigmentary
    "hydroquinone": "D11AX11", "kojic acid": "D11AX",
    "tranexamic acid": "B02BA01",
    # Psoriasis systemic
    "methotrexate": "L04AX03", "cyclosporine": "L04AD01",
    "acitretin": "D05BB02", "calcipotriol": "D05AX02",
    "tofacitinib": "L04AA29", "deucravacitinib": "L04AA56",
    "apremilast": "L04AA32",
    # Biologics
    "adalimumab": "L04AB04", "infliximab": "L04AB02",
    "etanercept": "L04AB01", "ustekinumab": "L04AC05",
    "secukinumab": "L04AC10", "ixekizumab": "L04AC13",
    "guselkumab": "L04AC16", "risankizumab": "L04AC18",
    "tildrakizumab": "L04AC17", "bimekizumab": "L04AC21",
    "dupilumab": "D11AH05", "tralokinumab": "D11AH07",
    # Oncology
    "pembrolizumab": "L01FF02", "nivolumab": "L01FF01",
    "ipilimumab": "L01FX04", "cemiplimab": "L01FF06",
    "dabrafenib": "L01EC02", "vemurafenib": "L01EC01",
    "trametinib": "L01EE01", "vismodegib": "L01XJ01",
    "sonidegib": "L01XJ02", "cladribine": "L01BB04",
    "fluorouracil": "L01BC02", "bleomycin": "L01DC01",
    # Antihistamines
    "cetirizine": "R06AE07", "loratadine": "R06AX13",
    "fexofenadine": "R06AX26", "astemizole": "R06AX11",
    "cimetidine": "A02BA01", "hydroxyzine": "N05BB01",
    # Hormonal
    "spironolactone": "C03DA01", "ethinyl estradiol": "G03CA01",
    "finasteride": "D11AX10",
    # Other derm
    "omalizumab": "R03DX05", "imiquimod": "D06BB10",
    "hydroxychloroquine": "P01BA02", "dapsone": "J04BA02",
    "rituximab": "L01FA01", "ruxolitinib": "L04AA35",
    "baricitinib": "L04AA37", "upadacitinib": "L04AA44",
    "abrocitinib": "L04AA45",
    # Anesthetics — N01 prefix is BLOCKED for most derm domains
    "benzocaine": "N01BA05", "lidocaine": "N01BB02",
    "tetracaine": "N01BA03", "pramocaine": "N01BB03",
    "pramoxine": "N01BB03",
    # Ophthalmic — should NEVER recommend for skin diseases
    "aflibercept": "S01LA05", "ranibizumab": "S01LA04",
    "brolucizumab": "S01LA06", "verteporfin": "S01LA01",
    # Zinc / vitamins
    "zinc chloride": "A12CB04", "zinc gluconate": "A12CB02",
    "zinc sulfate": "A12CB01",
    # Methoxsalen / psoralens
    "methoxsalen": "D05BA02", "trioxsalen": "D05BA01",
    # Anthralin / dithranol
    "anthralin": "D05AC01", "dithranol": "D05AC01",
    # Misc
    "clioquinol": "D08AH30",
    "aminobenzoic acid": "D02BA02", "benzoic acid": "D08AH",
    "ammonia": None,  # explicitly not a drug — exposure only
}


# Domains where particular ATC prefixes are inappropriate.
# Keys: domain name. Values: dict with allow_prefixes / block_prefixes.
ATC_DOMAIN_CONSTRAINTS: Dict[str, Dict[str, Set[str]]] = {
    "infectious_skin": {
        "allow_prefixes": {"D01", "D06", "J01", "J02", "J04", "J05",
                           "P02", "P03"},
        "block_prefixes": {"N01", "H02AB", "D07", "D10", "S01"},
    },
    "neoplastic_skin": {
        "allow_prefixes": {"L01", "L04", "L03", "D06BB", "P01BA"},
        "block_prefixes": {"S01", "A11", "D07", "D10", "D11AX",
                           "B02BA", "N01"},
    },
    "inflammatory_skin": {
        "allow_prefixes": {"D07", "D11", "L04", "H02", "D05", "R06"},
        "block_prefixes": {"S01"},
    },
    "autoimmune_skin": {
        "allow_prefixes": {"D07", "D11", "L04", "H02", "M01", "D05"},
        "block_prefixes": {"S01"},
    },
    "acneiform": {
        "allow_prefixes": {"D10", "J01", "G03", "C03DA", "D06BX",
                           "D06AX", "D11AX21", "D11AX22", "D11AH",
                           "P02CF", "D10AX"},
        "block_prefixes": {"D07", "H02", "S01", "J02AA"},
    },
    "pigmentary": {
        "allow_prefixes": {"D11", "D10AD", "D07", "L04", "B02BA"},
        "block_prefixes": {"L01", "N01", "S01"},
    },
    "unknown": {"allow_prefixes": set(), "block_prefixes": {"S01"}},
}

# Diseases → domain. Used by safety layer to look up the domain.
DISEASE_DOMAIN_SEEDS: Dict[str, str] = {
    # Acneiform
    "acne": "acneiform", "acne vulgaris": "acneiform",
    "rosacea": "acneiform", "perioral dermatitis": "acneiform",
    "hidradenitis suppurativa": "acneiform",
    # Inflammatory
    "eczema": "inflammatory_skin",
    "atopic dermatitis": "inflammatory_skin",
    "atopic eczema": "inflammatory_skin",
    "contact dermatitis": "inflammatory_skin",
    "urticaria": "inflammatory_skin", "angioedema": "inflammatory_skin",
    "seborrheic dermatitis": "inflammatory_skin",
    "lichen planus": "inflammatory_skin",
    "neurodermatitis": "inflammatory_skin",
    "pyoderma gangrenosum": "inflammatory_skin",
    "stevens-johnson syndrome": "inflammatory_skin",
    "sarcoidosis": "inflammatory_skin", "vasculitis": "inflammatory_skin",
    # Autoimmune
    "psoriasis": "autoimmune_skin",
    "vitiligo": "autoimmune_skin",
    "alopecia areata": "autoimmune_skin",
    "lupus": "autoimmune_skin", "dermatomyositis": "autoimmune_skin",
    "scleroderma": "autoimmune_skin",
    "pemphigus": "autoimmune_skin", "pemphigoid": "autoimmune_skin",
    "lichen sclerosus": "autoimmune_skin",
    "pustular psoriasis": "autoimmune_skin",
    # Infectious
    "tinea": "infectious_skin", "tinea corporis": "infectious_skin",
    "tinea pedis": "infectious_skin", "tinea capitis": "infectious_skin",
    "tinea manuum": "infectious_skin", "tinea cruris": "infectious_skin",
    "candidiasis": "infectious_skin",
    "oral candidiasis": "infectious_skin",
    "scabies": "infectious_skin", "impetigo": "infectious_skin",
    "herpes labialis": "infectious_skin",
    "herpes zoster": "infectious_skin",
    "herpes simplex": "infectious_skin",
    "molluscum contagiosum": "infectious_skin",
    "warts": "infectious_skin", "cellulitis": "infectious_skin",
    "lyme disease": "infectious_skin", "leprosy": "infectious_skin",
    "syphilis": "infectious_skin", "cutaneous anthrax": "infectious_skin",
    # Neoplastic
    "melanoma": "neoplastic_skin",
    "cutaneous melanoma": "neoplastic_skin",
    "basal cell carcinoma": "neoplastic_skin",
    "squamous cell carcinoma": "neoplastic_skin",
    "kaposi sarcoma": "neoplastic_skin",
    "mycosis fungoides": "neoplastic_skin",
    "actinic keratosis": "neoplastic_skin",
    "seborrheic keratosis": "neoplastic_skin",
    "langerhans cell histiocytosis": "neoplastic_skin",
    "lymphangioma": "neoplastic_skin",
    "hemangioma": "neoplastic_skin",
    # Pigmentary
    "melasma": "pigmentary",
    "post-inflammatory hyperpigmentation": "pigmentary",
    "hyperpigmentation": "pigmentary",
}

# Drugs that should NEVER appear regardless of domain.
GLOBAL_NEVER_RECOMMEND: Set[str] = {
    "ammonia",            # exposure causes contact dermatitis, never treats
    "anecortave acetate", "aflibercept", "pegaptanib", "brolucizumab",
    "ranibizumab", "verteporfin",
}


def get_atc(drug_name: str) -> Optional[str]:
    """ATC code lookup from the seed map (case-insensitive, with suffix-strip)."""
    if not drug_name:
        return None
    key = str(drug_name).lower().strip()
    key = re.sub(r"\s*\(.*?\)\s*$", "", key).strip()
    if key in ATC_SEED_MAP:
        return ATC_SEED_MAP[key]
    for variant in (
        key.replace(" acetate", ""), key.replace(" propionate", ""),
        key.replace(" valerate", ""), key.replace(" sulfate", ""),
        key.split()[0] if " " in key else key,
    ):
        if variant in ATC_SEED_MAP:
            return ATC_SEED_MAP[variant]
    return None


def disease_domain(disease_name: str) -> str:
    """Look up clinical domain for a disease."""
    if not disease_name:
        return "unknown"
    key = str(disease_name).lower().strip()
    if key in DISEASE_DOMAIN_SEEDS:
        return DISEASE_DOMAIN_SEEDS[key]
    # Substring match
    for seed_name, dom in DISEASE_DOMAIN_SEEDS.items():
        if seed_name in key or key in seed_name:
            return dom
    return "unknown"


def is_safe_recommendation(drug_name: str, disease_name: str) -> Tuple[bool, str]:
    """Validate a (drug, disease) pair against safety constraints.

    Returns (allowed, reason). allowed=False means the safety layer
    rejects this candidate. Used to filter Pareto/quick-wins output.
    """
    d = (drug_name or "").lower().strip()
    if d in GLOBAL_NEVER_RECOMMEND:
        return False, f"global_never_list:{d}"
    domain = disease_domain(disease_name)
    constraints = ATC_DOMAIN_CONSTRAINTS.get(
        domain, ATC_DOMAIN_CONSTRAINTS["unknown"])
    atc = get_atc(d)
    # Block list always fires
    if atc:
        for b in constraints["block_prefixes"]:
            if atc.startswith(b):
                return False, f"atc_blocked_{b}_for_{domain}"
    # Allow list: if known and non-empty, must match
    if atc and constraints["allow_prefixes"]:
        if not any(atc.startswith(a) for a in constraints["allow_prefixes"]):
            return False, f"atc_not_allowlisted_for_{domain}"
    # ATC unknown: permissive only in 'unknown' domain
    if atc is None and domain != "unknown" and constraints["allow_prefixes"]:
        # Check if drug is on the GLOBAL_NEVER list under a different name
        return True, "atc_unknown_permitted_via_kg_evidence"
    return True, "ok"


def atc_class_prefix(drug_name: str, depth: int = 4) -> str:
    """Drug-class function suitable for MissingEdgeProposer.drug_class_fn.

    Returns the first `depth` characters of the drug's ATC code (default 4
    captures the chemical/pharmacological subgroup). Drugs without an ATC
    map to an empty string, which the proposer treats as unique.
    """
    atc = get_atc(drug_name)
    if not atc:
        return ""
    return atc[:depth]

DRUG_TYPES = ("drug", "drug_or_chemical_compound", "compound")
DISEASE_TYPES = ("disease", "phenotype", "condition")


# ============================================================================
# Data normalisation
# ============================================================================

def _normalise_skin_stats(skin_stats_input: Any) -> Dict[str, Tuple[str, float]]:
    """Normalise heterogeneous skin-stats inputs to {disease: (subgroup, n_total)}.

    Accepts:
        dict-of-dicts: {'eczema': {'total': 100, 'fst_4_5_6': 30}, ...}
        dict-of-dicts with alias key 'fst_iv_vi'
        list of records: [{'disease_name': 'eczema', 'fst_total': 100, 'fst_iv_vi': 30}, ...]
        pandas.DataFrame with columns disease_name, fst_total, fst_iv_vi
    """
    out: Dict[str, Tuple[str, float]] = {}

    # If it's a pandas DataFrame, convert to records.
    try:
        import pandas as pd
        if isinstance(skin_stats_input, pd.DataFrame):
            df = skin_stats_input.rename(
                columns={c: c.lower().strip() for c in skin_stats_input.columns}
            )
            records = df.to_dict("records")
            return _normalise_skin_stats(records)
    except ImportError:
        pass

    # If it's a list, convert each record.
    if isinstance(skin_stats_input, list):
        for r in skin_stats_input:
            r = {str(k).lower().strip(): v for k, v in dict(r).items()}
            name = str(r.get("disease_name") or r.get("disease") or "").lower().strip()
            if not name:
                continue
            total = float(r.get("fst_total") or r.get("total") or 0)
            iv_vi = float(r.get("fst_iv_vi") or r.get("fst_4_5_6") or r.get("dark") or 0)
            if total <= 0:
                continue
            sg = "IV-VI" if iv_vi / total > 0.5 else "I-III"
            out[name] = (sg, total)
        return out

    # Otherwise expect a dict.
    if not isinstance(skin_stats_input, dict):
        raise TypeError(
            f"skin_stats must be dict, list, or DataFrame; got {type(skin_stats_input)}"
        )

    for name, val in skin_stats_input.items():
        name = str(name).lower().strip()
        if isinstance(val, dict):
            v = {str(k).lower().strip(): vv for k, vv in val.items()}
            total = float(v.get("total") or v.get("fst_total") or 0)
            iv_vi = float(v.get("fst_4_5_6") or v.get("fst_iv_vi") or v.get("dark") or 0)
        elif isinstance(val, (list, tuple)) and len(val) >= 2:
            total = float(val[0])
            iv_vi = float(val[1])
        else:
            continue
        if total <= 0:
            continue
        sg = "IV-VI" if iv_vi / total > 0.5 else "I-III"
        out[name] = (sg, total)
    return out


def _records_from_primekg_df(
    primekg_df: Any,
    skin_lookup: Dict[str, Tuple[str, float]],
    relations_to_keep: Tuple[str, ...] = INDICATION_RELATIONS,
) -> Iterable[EvidenceRecord]:
    """Stream EvidenceRecords from a PrimeKG-shaped DataFrame.

    Subgroup assignment uses the per-disease FST majority. Edge weight
    is log1p(disease's total FST sample count) — log scaling prevents
    very-large-cohort diseases from dominating the EB prior.
    """
    df = primekg_df.rename(
        columns={c: str(c).lower().strip() for c in primekg_df.columns}
    )
    required = {"relation", "x_name", "x_type", "y_name", "y_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"PrimeKG DataFrame missing required columns: {missing}"
        )
    n_kept = 0
    n_skipped_relation = 0
    n_skipped_no_demo = 0
    n_skipped_type = 0
    for row in df.itertuples(index=False):
        rel = str(getattr(row, "relation", "")).lower().strip()
        if rel not in relations_to_keep:
            n_skipped_relation += 1
            continue
        x_t = str(getattr(row, "x_type", "")).lower().strip()
        y_t = str(getattr(row, "y_type", "")).lower().strip()
        if x_t in DISEASE_TYPES and y_t in DRUG_TYPES:
            disease, drug = row.x_name, row.y_name
        elif x_t in DRUG_TYPES and y_t in DISEASE_TYPES:
            disease, drug = row.y_name, row.x_name
        else:
            n_skipped_type += 1
            continue
        d_key = str(disease).lower().strip()
        sg_info = skin_lookup.get(d_key)
        if sg_info is None:
            n_skipped_no_demo += 1
            continue
        sg, n_total = sg_info
        edge = Edge(head=d_key, relation=rel,
                    tail=str(drug).lower().strip())
        yield EvidenceRecord(
            edge=edge, subgroup=sg,
            outcome=1 if rel == "indication" else 0,
            weight=max(1.0, math.log1p(n_total)),
            source="primekg",
        )
        n_kept += 1
    logger.info(
        "PrimeKG ingestion (single-subgroup mode): kept %d records; "
        "skipped %d (relation), %d (no demographic), %d (entity type).",
        n_kept, n_skipped_relation, n_skipped_no_demo, n_skipped_type,
    )


def _records_from_primekg_df_per_fst(
    primekg_df: Any,
    skin_stats_full: Dict[str, Dict],
    relations_to_keep: Tuple[str, ...] = INDICATION_RELATIONS,
) -> Iterable[EvidenceRecord]:
    """Per-FST-weighted ingestion (recommended over single-subgroup mode).

    For each PrimeKG drug-disease edge, emit TWO EvidenceRecords — one
    for I-III with weight log1p(fst_i_iii_count), one for IV-VI with
    weight log1p(fst_iv_vi_count). This lets every edge carry evidence
    in BOTH subgroups proportional to actual FST sampling, so the EB
    prior fits, CEG reflects evidence-strength disparity instead of
    presence/absence noise, and downstream EIG varies meaningfully
    across candidates.

    Skips diseases with zero samples in BOTH subgroups (no demographic
    information).

    Args:
        primekg_df:        PrimeKG DataFrame with relation/x_name/x_type/y_name/y_type.
        skin_stats_full:   v5.5-shape dict {disease: {fst_i_iii, fst_iv_vi, ...}}.
        relations_to_keep: PrimeKG relation labels to include.
    """
    df = primekg_df.rename(
        columns={c: str(c).lower().strip() for c in primekg_df.columns}
    )
    required = {"relation", "x_name", "x_type", "y_name", "y_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"PrimeKG DataFrame missing required columns: {missing}"
        )
    # Normalise stats keys for case-insensitive lookup
    stats_lookup: Dict[str, Tuple[float, float]] = {}
    for name, val in skin_stats_full.items():
        if not isinstance(val, dict):
            continue
        v = {str(k).lower().strip(): vv for k, vv in val.items()}
        i_iii = float(v.get("fst_i_iii", 0) or 0)
        iv_vi = float(v.get("fst_iv_vi", v.get("fst_4_5_6", 0)) or 0)
        if i_iii + iv_vi <= 0:
            continue
        stats_lookup[str(name).lower().strip()] = (i_iii, iv_vi)

    n_emitted = 0
    n_edges_seen = 0
    n_skipped_relation = 0
    n_skipped_no_demo = 0
    n_skipped_type = 0
    for row in df.itertuples(index=False):
        rel = str(getattr(row, "relation", "")).lower().strip()
        if rel not in relations_to_keep:
            n_skipped_relation += 1
            continue
        x_t = str(getattr(row, "x_type", "")).lower().strip()
        y_t = str(getattr(row, "y_type", "")).lower().strip()
        if x_t in DISEASE_TYPES and y_t in DRUG_TYPES:
            disease, drug = row.x_name, row.y_name
        elif x_t in DRUG_TYPES and y_t in DISEASE_TYPES:
            disease, drug = row.y_name, row.x_name
        else:
            n_skipped_type += 1
            continue
        d_key = str(disease).lower().strip()
        info = stats_lookup.get(d_key)
        if info is None:
            n_skipped_no_demo += 1
            continue
        i_iii, iv_vi = info
        edge = Edge(head=d_key, relation=rel,
                    tail=str(drug).lower().strip())
        outcome = 1 if rel == "indication" else 0
        n_edges_seen += 1
        if i_iii > 0:
            yield EvidenceRecord(
                edge=edge, subgroup="I-III", outcome=outcome,
                weight=max(1.0, math.log1p(i_iii)),
                source="primekg_per_fst",
            )
            n_emitted += 1
        if iv_vi > 0:
            yield EvidenceRecord(
                edge=edge, subgroup="IV-VI", outcome=outcome,
                weight=max(1.0, math.log1p(iv_vi)),
                source="primekg_per_fst",
            )
            n_emitted += 1
    logger.info(
        "PrimeKG ingestion (per-FST mode): %d edges → %d records "
        "(both-subgroup coverage = %.1f%%); skipped %d (relation), "
        "%d (no demographic), %d (entity type).",
        n_edges_seen, n_emitted,
        100.0 * n_emitted / max(2 * n_edges_seen, 1),
        n_skipped_relation, n_skipped_no_demo, n_skipped_type,
    )


# ============================================================================
# Main runner
# ============================================================================

def run_from_dataframes(
    primekg_df: Any,
    skin_stats: Any,
    output_dir: str,
    target_subgroup: str = "IV-VI",
    n_per_trial: int = 30,
    top_k_diseases: int = 100,
    top_k_candidates: int = 500,
    include_tvd: bool = True,
    relations_to_keep: Tuple[str, ...] = INDICATION_RELATIONS,
    use_per_fst_records: bool = True,
) -> Dict[str, Any]:
    """Run the full pipeline and write all outputs to disk.

    Args:
        use_per_fst_records: if True (default) and skin_stats is a v5.5-shape
            dict with fst_i_iii / fst_iv_vi keys, emit two EvidenceRecords
            per edge weighted by per-disease FST counts. This lets every
            edge carry evidence in BOTH subgroups so the EB prior fits
            and CEG reflects evidence-strength disparity rather than
            presence/absence noise. Strongly recommended.
            If False, falls back to single-subgroup-per-disease mode.

    Returns a dict of metrics for programmatic use; also writes CSV files
    to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # --- Stage 0: ingest ----------------------------------------------------
    # Decide which records function to use. Per-FST mode requires the full
    # v5.5-shape dict with fst_i_iii / fst_iv_vi keys.
    has_per_fst_keys = (
        isinstance(skin_stats, dict)
        and any(
            isinstance(v, dict) and (
                "fst_i_iii" in {str(k).lower() for k in v}
                or "fst_iv_vi" in {str(k).lower() for k in v}
                or "fst_4_5_6" in {str(k).lower() for k in v}
            )
            for v in skin_stats.values()
        )
    )

    if use_per_fst_records and has_per_fst_keys:
        logger.info(
            "Skin demographics: using per-FST evidence records "
            "(both subgroups per edge).")
        records = list(_records_from_primekg_df_per_fst(
            primekg_df, skin_stats, relations_to_keep=relations_to_keep,
        ))
    else:
        skin_lookup = _normalise_skin_stats(skin_stats)
        logger.info(
            "Skin demographics: using single-subgroup-per-disease records "
            "(%d diseases mapped). For better EB pooling pass v5.5-shape "
            "stats with fst_i_iii / fst_iv_vi keys.", len(skin_lookup))
        if not skin_lookup:
            raise ValueError(
                "Skin stats lookup is empty after normalisation. Check that "
                "your skin_stats data has fst_total/total > 0 entries."
            )
        records = list(_records_from_primekg_df(
            primekg_df, skin_lookup, relations_to_keep=relations_to_keep,
        ))

    # min_pool_n=2 in per-FST mode because weights are log1p-scaled:
    # log1p(7) ≈ 2.08, so EB pools edges where the disease has ≥ 7 samples
    # per subgroup. With the default min_pool_n=5 we'd need ~150 samples
    # per subgroup which most derm conditions don't reach.
    _min_pool = 2 if (use_per_fst_records and has_per_fst_keys) else 5
    scp = SubgroupConditionalPosteriorKG(
        subgroups=("I-III", "IV-VI"), min_pool_n=_min_pool,
    )
    if not records:
        raise ValueError(
            "No EvidenceRecords generated from PrimeKG. Check that "
            "your PrimeKG DataFrame has 'indication' relations and that "
            "disease names match between PrimeKG and skin_stats (case- "
            "and whitespace-insensitive)."
        )
    scp.ingest(records)

    summary = scp.summary()
    logger.info(summary)
    with open(os.path.join(output_dir, "scp_kg_summary.txt"), "w") as f:
        f.write(summary + "\n")

    # --- CEG ----------------------------------------------------------------
    ceg = CounterfactualEquityGap(scp, "I-III", "IV-VI")
    top_gaps = ceg.rank_gaps(top_k=100)
    with open(os.path.join(output_dir, "ceg_top100.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "disease", "drug", "relation",
            "ceg", "mean_disagreement_kl", "uncert_disagreement_kl",
            "dominant_cause", "severity",
            "majority_post_mean", "majority_n_eff",
            "minority_post_mean", "minority_n_eff",
            "has_majority_evidence", "has_minority_evidence",
        ])
        for i, g in enumerate(top_gaps, 1):
            w.writerow([
                i, g.edge.head, g.edge.tail, g.edge.relation,
                f"{g.ceg:.4f}",
                f"{g.mean_disagreement_kl:.4f}",
                f"{g.uncert_disagreement_kl:.4f}",
                g.dominant_cause,
                g.severity,
                f"{g.posterior_majority.mean:.3f}",
                f"{g.posterior_majority.n_eff:.1f}",
                f"{g.posterior_minority.mean:.3f}",
                f"{g.posterior_minority.n_eff:.1f}",
                int(g.has_majority_evidence),
                int(g.has_minority_evidence),
            ])

    # --- Full posteriors export (used by comparison cell as a stand-alone -----
    # ranker over ALL (disease, drug) pairs, not just IGR-flagged ones). -----
    # Also export the EB prior so the comparison cell can score unseen pairs.
    with open(os.path.join(output_dir, "scp_all_posteriors.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "disease", "drug", "relation",
            "alpha_iii", "beta_iii", "post_mean_iii", "n_eff_iii",
            "alpha_ivvi", "beta_ivvi", "post_mean_ivvi", "n_eff_ivvi",
            "ceg",
        ])
        for edge in scp.edges():
            try:
                p_iii = scp.posterior(edge, "I-III")
                p_ivvi = scp.posterior(edge, "IV-VI")
                kl = p_iii.kl(p_ivvi) if hasattr(p_iii, "kl") else float("nan")
            except Exception:
                continue
            w.writerow([
                edge.head, edge.tail, edge.relation,
                f"{p_iii.alpha:.4f}", f"{p_iii.beta:.4f}",
                f"{p_iii.mean:.4f}", f"{p_iii.n_eff:.2f}",
                f"{p_ivvi.alpha:.4f}", f"{p_ivvi.beta:.4f}",
                f"{p_ivvi.mean:.4f}", f"{p_ivvi.n_eff:.2f}",
                f"{kl:.4f}",
            ])
    # Export EB prior for unseen-pair scoring
    with open(os.path.join(output_dir, "scp_eb_prior.json"), "w") as f:
        json.dump({
            "alpha0": float(scp.prior.alpha0),
            "beta0":  float(scp.prior.beta0),
        }, f)

    # --- IGR 4-stage --------------------------------------------------------
    # Build a real cost function. Cost reflects minority-cohort recruitment
    # difficulty: rare diseases in IV-VI populations cost more to study.
    # Common diseases with hundreds of FST IV-VI samples are cheap; rare
    # diseases with few samples are expensive. Range ≈ [1.0, 50.0].
    skin_lookup_for_cost: Dict[str, Dict] = {}
    if isinstance(skin_stats, dict):
        for name, val in skin_stats.items():
            if isinstance(val, dict):
                v = {str(k).lower().strip(): vv for k, vv in val.items()}
                skin_lookup_for_cost[str(name).lower().strip()] = dict(
                    fst_i_iii=float(v.get("fst_i_iii", 0) or 0),
                    fst_iv_vi=float(v.get("fst_iv_vi",
                                          v.get("fst_4_5_6", 0)) or 0),
                    total=float(v.get("total", v.get("fst_total", 0)) or 0),
                )

    def _cost_fn(edge, target_subgroup):
        info = skin_lookup_for_cost.get(edge.head, {})
        if target_subgroup == "IV-VI":
            n = info.get("fst_iv_vi", 0)
        else:
            n = info.get("fst_i_iii", 0)
        # Cost ∝ inverse minority-cohort density. Common diseases
        # (n ≥ ~90) hit the floor of 1.0 → eligible for quick_wins.
        # Rare diseases scale up to cost ≈ 10 (n=0). Always ≥ 1.
        return max(1.0, 100.0 / (max(n, 0) + 10))

    igr = InverseGraphReasoning(
        scp, ceg, n_per_trial=n_per_trial, include_tvd=include_tvd,
        cost_fn=_cost_fn,
        drug_class_fn=atc_class_prefix,
    )
    result = igr.run(
        target_subgroup=target_subgroup,
        top_k_diseases=top_k_diseases,
        top_k_candidates=top_k_candidates,
    )
    print(result.summary())

    # Apply the v5.5 safety layer: filter out candidates whose drug-disease
    # pairing fails the ATC domain constraint (e.g. amphotericin → rosacea,
    # ammonia → contact dermatitis, ophthalmic anti-VEGFs anywhere).
    n_pre_safety = len(result.scored_candidates)
    safe_candidates = []
    rejected_safety = []
    for s in result.scored_candidates:
        ok, reason = is_safe_recommendation(s.candidate.edge.tail,
                                             s.candidate.edge.head)
        if ok:
            safe_candidates.append(s)
        else:
            rejected_safety.append((s, reason))
    result.scored_candidates = safe_candidates
    # Recompute Pareto + quick wins on the filtered list
    result.pareto_frontier = ParetoRanker.find_frontier(safe_candidates)
    result.quick_wins = ParetoRanker.quick_wins(safe_candidates,
                                                 max_cost=5.0)
    n_post_safety = len(safe_candidates)
    if n_pre_safety != n_post_safety:
        rej_reasons = Counter(r for _, r in rejected_safety)
        logger.info(
            "Safety filter: %d → %d candidates (rejected %d). Top reasons: %s",
            n_pre_safety, n_post_safety, len(rejected_safety),
            dict(rej_reasons.most_common(5)))
        # Persist rejection log so users can inspect what was filtered
        with open(os.path.join(output_dir, "safety_rejected.csv"),
                  "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["disease", "drug", "type", "reason",
                        "expected_information_gain", "cost"])
            for s, reason in rejected_safety[:1000]:
                c = s.candidate
                w.writerow([
                    c.edge.head, c.edge.tail, c.candidate_type, reason,
                    f"{s.expected_information_gain:.4f}",
                    f"{s.cost:.2f}",
                ])

    # Write Stage 1 — disease-level gaps
    with open(os.path.join(output_dir, "igr_disease_gaps.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "disease", "severity",
            "directional_equity_score", "aggregate_ceg", "max_ceg",
            "n_edges", "n_majority_evidenced", "n_minority_evidenced",
            "representation_deficit", "rationale",
        ])
        for i, d in enumerate(result.disease_gaps, 1):
            w.writerow([
                i, d.disease, d.severity,
                f"{d.directional_equity_score:.4f}",
                f"{d.aggregate_ceg:.4f}", f"{d.max_ceg:.4f}",
                d.n_edges, d.n_majority_evidence_edges,
                d.n_minority_evidence_evidence_edges,
                f"{d.representation_deficit:.3f}", d.rationale,
            ])

    # Write Stage 2/3 — all candidates with scores
    with open(os.path.join(output_dir, "igr_all_candidates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "type", "extrapolation_confidence",
            "disease", "drug", "relation", "target_subgroup",
            "expected_information_gain", "cost", "eig_per_cost",
            "equity_gain", "post_mean_majority", "post_mean_minority",
            "rationale",
        ])
        for i, s in enumerate(result.scored_candidates, 1):
            c = s.candidate
            w.writerow([
                i, c.candidate_type, f"{c.extrapolation_confidence:.3f}",
                c.edge.head, c.edge.tail, c.edge.relation, c.target_subgroup,
                f"{s.expected_information_gain:.4f}",
                f"{s.cost:.2f}", f"{s.eig_per_cost:.4f}",
                f"{s.equity_gain:.4f}",
                f"{s.posterior_mean_majority:.3f}",
                f"{s.posterior_mean_minority:.3f}",
                c.rationale,
            ])

    # Write Stage 4 — quick wins
    with open(os.path.join(output_dir, "igr_quick_wins.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "disease", "drug", "type",
            "expected_information_gain", "cost", "equity_gain",
            "post_mean_majority", "post_mean_minority",
        ])
        for i, s in enumerate(result.quick_wins, 1):
            c = s.candidate
            w.writerow([
                i, c.edge.head, c.edge.tail, c.candidate_type,
                f"{s.expected_information_gain:.4f}", f"{s.cost:.2f}",
                f"{s.equity_gain:.4f}",
                f"{s.posterior_mean_majority:.3f}",
                f"{s.posterior_mean_minority:.3f}",
            ])

    # Write Stage 4 — Pareto frontier
    with open(os.path.join(output_dir, "igr_pareto_frontier.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "disease", "drug", "type",
            "expected_information_gain", "cost", "equity_gain",
            "extrapolation_confidence",
        ])
        sorted_frontier = sorted(result.pareto_frontier, key=lambda s: s.cost)
        for i, s in enumerate(sorted_frontier, 1):
            c = s.candidate
            w.writerow([
                i, c.edge.head, c.edge.tail, c.candidate_type,
                f"{s.expected_information_gain:.4f}", f"{s.cost:.2f}",
                f"{s.equity_gain:.4f}",
                f"{c.extrapolation_confidence:.3f}",
            ])

    # --- TVD output ---------------------------------------------------------
    n_voids_written = 0
    if include_tvd:
        try:
            tvd = TopologicalVoidDetector(scp, "I-III", "IV-VI")
            voids = tvd.detect_voids()
            with open(os.path.join(output_dir, "structural_voids.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "rank", "dimension", "majority_persistence",
                    "minority_persistence", "void_score", "n_nodes",
                    "representative_nodes",
                ])
                for i, v in enumerate(voids, 1):
                    w.writerow([
                        i, v.feature.dimension,
                        f"{v.majority_persistence:.4f}",
                        f"{v.minority_persistence:.4f}",
                        f"{v.void_score:.4f}",
                        len(v.feature.representative_nodes),
                        ";".join(v.feature.representative_nodes[:30]),
                    ])
                    n_voids_written += 1
        except Exception as exc:
            logger.warning("TVD failed: %s", exc)

    # --- Pipeline metrics for the paper ------------------------------------
    metrics = {
        "n_evidence_records":      scp.n_records(),
        "n_edges":                 len(scp.edges()),
        "eb_prior_alpha0":         scp.prior.alpha0,
        "eb_prior_beta0":          scp.prior.beta0,
        "global_disparity":        ceg.global_disparity(),
        "n_disease_gaps":          len(result.disease_gaps),
        "n_severe_disease_gaps":   sum(1 for d in result.disease_gaps if d.severity == "severe"),
        "n_candidates":            len(result.candidates),
        "n_candidates_type_a":     sum(1 for c in result.candidates if c.candidate_type == "A"),
        "n_candidates_type_b":     sum(1 for c in result.candidates if c.candidate_type == "B"),
        "n_candidates_type_c":     sum(1 for c in result.candidates if c.candidate_type == "C"),
        "n_pareto_frontier":       len(result.pareto_frontier),
        "n_quick_wins":            len(result.quick_wins),
        "n_structural_voids":      n_voids_written,
        "wall_time_seconds":       time.time() - t0,
        "target_subgroup":         target_subgroup,
    }
    with open(os.path.join(output_dir, "pipeline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)
    print(f"Output directory: {output_dir}")
    print(f"Wall time: {metrics['wall_time_seconds']:.1f}s")
    print()
    print("Key findings:")
    print(f"  evidence records ingested  : {metrics['n_evidence_records']:,}")
    print(f"  unique edges in SCP-KG     : {metrics['n_edges']:,}")
    print(f"  EB prior                   : Beta({metrics['eb_prior_alpha0']:.2f}, "
          f"{metrics['eb_prior_beta0']:.2f})")
    print(f"  mean CEG over all edges    : {metrics['global_disparity']['mean_ceg']:.3f}")
    print(f"  fraction severe (CEG>1)    : {metrics['global_disparity']['frac_severe']:.1%}")
    print(f"  Type A candidates          : {metrics['n_candidates_type_a']:,}")
    print(f"  Pareto frontier size       : {metrics['n_pareto_frontier']}")
    print(f"  quick wins                 : {metrics['n_quick_wins']}")
    print(f"  structural voids (TVD)     : {metrics['n_structural_voids']}")
    print()
    print("Files written:")
    for fn in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, fn)
        size = os.path.getsize(path)
        print(f"  {fn}  ({size:,} bytes)")
    print("=" * 78)
    return {
        "metrics": metrics,
        "result": result,
        "scp": scp,
        "ceg": ceg,
        "cost_fn": _cost_fn,
    }


def run_from_csv(
    primekg_csv: str,
    skin_stats_csv: str,
    output_dir: str,
    **kwargs,
) -> Dict[str, Any]:
    """CSV path: load both inputs from disk and call run_from_dataframes."""
    import pandas as pd
    primekg_df = pd.read_csv(primekg_csv, low_memory=False)
    skin_df = pd.read_csv(skin_stats_csv)
    return run_from_dataframes(primekg_df, skin_df, output_dir, **kwargs)


# ============================================================================
# CLI
# ============================================================================


# =============================================================================
# §5  RUN: load via v5.5 DataLoader → drive pipeline
# =============================================================================

print("=" * 78)
print("DERMAKG-CAUSAL — KAGGLE PIPELINE")
print("=" * 78)

loader = DataLoader(data_dir=DATA_DIR)

print("\n[1/4] Loading PrimeKG ...")
primekg_df = loader.load_primekg()
print(f"  → {len(primekg_df):,} rows, columns: {list(primekg_df.columns)}")

print("\n[2/4] Loading Fitzpatrick17k ...")
fitz_df = loader.load_fitzpatrick()
print(f"  → {len(fitz_df):,} rows")

print("\n[3/4] Loading DermaCon-IN ...")
try:
    dermacon_df = loader.load_dermacon()
    print(f"  → {len(dermacon_df):,} rows")
except Exception as exc:
    logger.warning("DermaCon load failed (%s); continuing with empty frame.", exc)
    dermacon_df = pd.DataFrame()

print("\n[4/4] Computing skin stats (v5.5 compute_skin_stats) ...")
skin_stats = loader.compute_skin_stats(fitz_df, dermacon_df)
n_with_iv_vi = sum(1 for v in skin_stats.values() if v.get("fst_iv_vi", 0) > 0)
print(f"  → {len(skin_stats):,} disease entries, "
      f"{n_with_iv_vi:,} with FST IV-VI samples")

# Save skin_stats so you can reuse it without re-loading
os.makedirs(OUTPUT_DIR, exist_ok=True)
skin_stats_df = pd.DataFrame([
    dict(disease_name=k, fst_total=v["total"],
         fst_iv_vi=v["fst_iv_vi"], fst_i_iii=v["fst_i_iii"],
         prevalence_iv_vi=v["prevalence_iv_vi"],
         total_dermacon=v.get("total_dermacon", 0))
    for k, v in skin_stats.items()
])
skin_stats_df.to_csv(os.path.join(OUTPUT_DIR, "skin_stats_v5_5.csv"), index=False)
print(f"  → wrote {OUTPUT_DIR}/skin_stats_v5_5.csv")

# -----------------------------------------------------------------------------
# §6  RUN PIPELINE  (SCP-KG → CEG → IGR 4-stage → TVD)
# -----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("§6  RUNNING PIPELINE")
print("=" * 78)

# run_from_dataframes accepts skin_stats as the v5.5 dict-of-dicts directly —
# its _normalise_skin_stats reads either 'fst_total/fst_iv_vi' or
# 'total/fst_iv_vi' or 'total/fst_4_5_6'. Returns a dict with metrics +
# the live SCP-KG, CEG, IGR result, and cost function so the comparison cell
# can use them.
_artifacts = run_from_dataframes(
    primekg_df=primekg_df,
    skin_stats=skin_stats,
    output_dir=OUTPUT_DIR,
    target_subgroup=TARGET_SUBGROUP,
    n_per_trial=N_PER_TRIAL,
    top_k_diseases=TOP_K_DISEASES,
    top_k_candidates=TOP_K_CANDIDATES,
    include_tvd=INCLUDE_TVD,
)
metrics = _artifacts["metrics"]
result = _artifacts["result"]
scp = _artifacts["scp"]
ceg = _artifacts["ceg"]
_cost_fn = _artifacts["cost_fn"]

# -----------------------------------------------------------------------------
# §7  HEADLINE RESULTS
# -----------------------------------------------------------------------------

def _show_csv(label, path, n=15):
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    print(f"\n--- {label} (showing {min(n, len(df))} of {len(df):,}) ---")
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(df.head(n).to_string(index=False))

print("\n" + "=" * 78)
print("§7  HEADLINE RESULTS")
print("=" * 78)
_show_csv("TOP CEG (largest equity gaps)",
          os.path.join(OUTPUT_DIR, "ceg_top100.csv"), n=15)
_show_csv("TOP DISEASE GAPS (Stage 1)",
          os.path.join(OUTPUT_DIR, "igr_disease_gaps.csv"), n=15)
_show_csv("QUICK WINS — Type A, low cost (headline result for the paper)",
          os.path.join(OUTPUT_DIR, "igr_quick_wins.csv"), n=15)
_show_csv("PARETO FRONTIER (Stage 4)",
          os.path.join(OUTPUT_DIR, "igr_pareto_frontier.csv"), n=15)

print("\n" + "=" * 78)
print("DONE — outputs in:", OUTPUT_DIR)
print("=" * 78)
with open(os.path.join(OUTPUT_DIR, "pipeline_metrics.json")) as f:
    print(json.dumps(json.load(f), indent=2))