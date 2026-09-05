"""Packet-DB: jointly learn the frozen geometry, CBA and cross-bank state."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import RESET, _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_armenian_central_bank_features import build_cba_features, load_cba
from research.round6_belarus_nbrb_features import load_nbrb
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_error_regime import row_scores
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_crossbank_consensus import build_crossbank_features
from research.round6_georgia_nbg_features import load_nbg
from research.round6_kazakh_central_bank_features import load_kazakh_nbk
from research.round6_kyrgyz_central_bank_features import load_kyrgyz_nbkr
from research.round6_local_central_bank_features import load_nbt
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_features import load_uzbek_cbu
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/joint_external_stack")
PRIMARY_PATH = Path("results/research/round6/cny_consensus/outputs.pkl")
PRIMARY = "logit50_extra50"
GEOMETRY_PATH = Path("results/research/round6/cny_expert_geometry/outputs.pkl")
GEOMETRY = "primary75_geometry_min75_max2525"
INCUMBENT_PATH = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
INCUMBENT = "geometry75_cba_consensus_basis25"
SEED = 20260905
RANK_WINDOW = 250
RANK_MINIMUM = 20
EXTERNAL_LAG = 20
SOURCES = {
    "armenia_cba": load_cba,
    "tajikistan_nbt": load_nbt,
    "uzbekistan_cbu": load_uzbek_cbu,
    "kazakhstan_nbk": load_kazakh_nbk,
    "kyrgyzstan_nbkr": load_kyrgyz_nbkr,
    "georgia_nbg": load_nbg,
    "belarus_nbrb": load_nbrb,
}
TARGET_FEATURES = (
    "pct_range_30", "pct_range_90", "pct_range_180",
    "ret_1", "ret_5", "ret_20",
)
CBA_FEATURES = (
    "cba_direct_basis_bps", "cba_usd_basis_bps", "cba_cny_basis_bps",
    "cba_consensus_basis_bps", "cba_direct_minus_usd_bps",
    "cba_direct_minus_cny_bps", "cba_usd_minus_cny_bps",
    "cba_rub_quote_ret_1", "cba_rub_quote_ret_2", "cba_rub_quote_ret_5",
    "cba_rub_age_days", "cba_usd_age_days", "cba_cny_age_days",
)
CROSSBANK_FEATURES = (
    "median_consensus", "trimmed_consensus", "fresh_weighted_consensus",
    "negative_dispersion", "positive_breadth", "signed_consensus_to_dispersion",
)
ORDER = (
    "incumbent",
    "joint_logit",
    "joint_logit_stale20",
    "joint_hist",
    "joint_hist_stale20",
    "joint_extra",
    "incumbent75_joint_logit25",
    "incumbent75_joint_hist25",
    "incumbent75_joint_extra25",
)


def _load(path: Path, name: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[name]


def _model(kind):
    if kind == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.025, max_iter=3000, random_state=SEED),
        )
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.03, max_leaf_nodes=5,
            min_samples_leaf=100, l2_regularization=30.0,
            random_state=SEED,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=45,
            max_features=.65, n_jobs=1, random_state=SEED,
        )
    raise KeyError(kind)


def prequential_scores(kind, matrix, y, dates, reach, verbose=True):
    score = np.full(len(y), np.nan, dtype=float)
    logs = []
    finite = np.all(np.isfinite(matrix), axis=1)
    for start in _quarter_starts():
        if start.year < 2024:
            continue
        end = _next_quarter(start)
        train = (
            (dates >= RESET)
            & np.asarray([value < start for value in reach])
            & np.isfinite(y) & finite
        )
        test = (
            (dates >= start) & (dates < end)
            & np.isfinite(y) & finite
        )
        rows = np.flatnonzero(train)
        target = np.flatnonzero(test)
        if len(rows) < 1000 or not len(target):
            continue
        if not all(reach[row] < start for row in rows):
            raise AssertionError("unresolved target admitted to joint stack")
        model = _model(kind)
        model.fit(matrix[rows], y[rows])
        score[target] = model.predict_proba(matrix[target])[:, 1]
        logs.append({
            "candidate": kind, "quarter": str(start),
            "n_train": len(rows), "n_test": len(target),
            "last_resolved": str(max(reach[rows])),
            "n_features": matrix.shape[1],
        })
        if verbose:
            print(
                f"  joint {kind:<6} quarter={start} "
                f"train={len(rows):5d} test={len(target):4d}", flush=True,
            )
    return score, logs


def outcome_causality_check(matrix, y, dates, reach):
    cutoff = dt.date(2025, 6, 30)
    first, _ = prequential_scores("logit", matrix, y, dates, reach, verbose=False)
    changed = y.copy()
    unresolved = np.asarray([
        np.isfinite(y[row]) and reach[row] > cutoff for row in range(len(y))
    ])
    changed[unresolved] = 1.0 - changed[unresolved]
    second, _ = prequential_scores("logit", matrix, changed, dates, reach, verbose=False)
    past = (dates <= cutoff) & np.isfinite(first)
    np.testing.assert_array_equal(first[past], second[past])
    return True


def build_matrices(X, names, index, dates, currencies, series, references):
    primary = _load(PRIMARY_PATH, PRIMARY)
    geometry = _load(GEOMETRY_PATH, GEOMETRY)
    ranks = np.column_stack([
        causal_percentiles(
            row_scores(output, len(index)), dates, currencies,
            RANK_WINDOW, RANK_MINIMUM,
        )
        for output in (primary, geometry)
    ])
    cba, _digest = load_cba()
    cba_matrix, cba_names = build_cba_features(index, series, references, cba)
    cba_columns = np.asarray([cba_names.index(name) for name in CBA_FEATURES])
    cba_block = cba_matrix[:, cba_columns].astype(float)
    sources = {name: loader()[0] for name, loader in SOURCES.items()}
    crossbank, crossbank_names, _availability = build_crossbank_features(
        index, references, sources,
    )
    cross_columns = np.asarray([
        crossbank_names.index(name) for name in CROSSBANK_FEATURES
    ])
    external = np.column_stack([cba_block, crossbank[:, cross_columns]])
    target_columns = np.asarray([names.index(name) for name in TARGET_FEATURES])
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ])
    stable = np.column_stack([ranks, X[:, target_columns], X[:, currency_columns]])
    aligned = np.column_stack([stable, external])
    stale = np.column_stack([
        stable, delayed_by_currency(external, index, rows=EXTERNAL_LAG),
    ])
    feature_names = (
        ["primary_rank", "geometry_rank"]
        + list(TARGET_FEATURES)
        + [names[i] for i in currency_columns]
        + list(CBA_FEATURES)
        + list(CROSSBANK_FEATURES)
    )
    return {"aligned": aligned, "stale": stale}, feature_names


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    y5 = targets["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    matrices, feature_names = build_matrices(
        X, names, index, dates, currencies, series, references,
    )
    outcome_causality_check(matrices["aligned"], y5, dates, reach)

    raw, logs = {}, []
    for candidate, kind, matrix_name in (
        ("joint_logit", "logit", "aligned"),
        ("joint_logit_stale20", "logit", "stale"),
        ("joint_hist", "hist", "aligned"),
        ("joint_hist_stale20", "hist", "stale"),
        ("joint_extra", "extra", "aligned"),
    ):
        score, part = prequential_scores(kind, matrices[matrix_name], y5, dates, reach)
        raw[candidate] = _outputs(score, y5, dates)
        for row in part:
            logs.append({**row, "candidate": candidate})
    incumbent = _load(INCUMBENT_PATH, INCUMBENT)
    outputs = {"incumbent": incumbent, **raw}
    for kind in ("logit", "hist", "extra"):
        outputs[f"incumbent75_joint_{kind}25"] = combine_causal(
            [incumbent, raw[f"joint_{kind}"]], (.75, .25), dates, currencies,
        )
    pd.DataFrame(logs).to_csv(OUT / "training_log.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    screen = horizon_rows(outputs, (2024,), targets, forwards, dates, currencies)
    summary = summarize(screen)
    selected = _choose(summary)
    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    summary.to_csv(OUT / "screen_2024_summary.csv", index=False)

    comparison = {"incumbent": incumbent, "selected": outputs[selected]}
    later_parts = []
    for period, years in (
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        part = horizon_rows(comparison, years, targets, forwards, dates, currencies)
        part["period"] = period
        later_parts.append(part)
    later = pd.concat(later_parts, ignore_index=True)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    summarize(later[later.period == "combined_2025_2026"]).to_csv(
        OUT / "later_summary.csv", index=False,
    )

    h5_rows = []
    for candidate in ORDER:
        output = outputs[candidate]
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(
                output, years, POLICY, y5, forwards[5], dates, currencies,
            )
            item.update({"candidate": candidate, "period": period, **POLICY})
            h5_rows.append(item)
    h5 = pd.DataFrame(h5_rows)
    h5.to_csv(OUT / "standard_h5_results.csv", index=False)
    bootstrap, masks, valid = _bootstrap(
        h5[h5.period == "screen_2024"], outputs, (2025, 2026),
        y5, forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        y5, dates, currencies, valid, masks, "joint_external_stack_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate in ORDER:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    training = pd.DataFrame(logs)
    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    aligned = summary.set_index("candidate")
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DB", "fixed_policy": POLICY,
        "models": ["logit", "hist", "extra"],
        "feature_names": feature_names,
        "training_start": str(RESET), "quarterly_refit": True,
        "h5_training_labels_resolved": chronology_ok,
        "external_stale_control_rows_per_currency": EXTERNAL_LAG,
        "screen_aligned_vs_stale": {
            "logit": {
                "aligned": float(aligned.loc["joint_logit", "horizon_lift_min"]),
                "stale": float(aligned.loc["joint_logit_stale20", "horizon_lift_min"]),
            },
            "hist": {
                "aligned": float(aligned.loc["joint_hist", "horizon_lift_min"]),
                "stale": float(aligned.loc["joint_hist_stale20", "horizon_lift_min"]),
            },
        },
        "selection_period": 2024, "selected": selected,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "future_outcome_corruption_check": True,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after 2024 selection",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if not chronology_ok:
        raise AssertionError("joint stack training chronology failed")
    print("Selected on 2024:", selected)
    print("\nSCREEN\n" + summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).to_string(index=False))
    print("\nLATER\n" + summarize(
        later[later.period == "combined_2025_2026"]
    ).to_string(index=False))
    print("\nLATER BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
