"""Packet-EL: global residual boosting over the causal 15:30 fixing anchor."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_fixing_proxies import proxy_causality_check
from research.round6_moex_spot_1530_features import (
    build_spot_1530_features,
    load_spot_1530_history,
)
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/fixing_residual_boost")
FIXING_PATH = Path("results/research/round6/fixing_proxies/outputs.pkl")
ROUTER_PATH = Path(
    "results/research/round6/fixing_availability_router/outputs.pkl"
)
TRAIN_START = dt.date(2022, 5, 1)
SEED = 20260905
RANK_WINDOW = 250
RANK_MINIMUM = 20
RESIDUAL_WEIGHT = .25
ROUTER_WEIGHT = .75
STALE_ROWS = 20
TARGET_FEATURES = (
    "pct_range_30", "pct_range_90", "pct_range_180",
    "ret_1", "ret_5", "ret_20",
)
RAW_RESIDUALS = ("residual_hist", "residual_extra", "residual_mean")


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def _base_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=.05, max_iter=3000, random_state=SEED),
    )


def _residual_model(kind):
    if kind == "hist":
        return HistGradientBoostingRegressor(
            max_iter=150, learning_rate=.03, max_leaf_nodes=5,
            min_samples_leaf=100, l2_regularization=30.0,
            random_state=SEED,
        )
    if kind == "extra":
        return ExtraTreesRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=40,
            max_features=.65, n_jobs=1, random_state=SEED,
        )
    raise KeyError(kind)


def prequential_scores(base_matrix, aligned, stale, y, dates, reach, verbose=True):
    names = (
        "calibrated_anchor", "residual_hist", "residual_hist_stale20",
        "residual_extra", "residual_extra_stale20",
    )
    scores = {name: np.full(len(y), np.nan, dtype=float) for name in names}
    finite = (
        np.all(np.isfinite(base_matrix), axis=1)
        & np.all(np.isfinite(aligned), axis=1)
        & np.all(np.isfinite(stale), axis=1)
    )
    logs = []
    for start in _quarter_starts():
        if start.year < 2024:
            continue
        end = _next_quarter(start)
        train = (
            (dates >= TRAIN_START)
            & np.asarray([value < start for value in reach])
            & np.isfinite(y)
            & finite
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
            raise AssertionError("unresolved h5 label admitted to fixing residual")
        base = _base_model()
        base.fit(base_matrix[rows], y[rows])
        train_base = base.predict_proba(base_matrix[rows])[:, 1]
        test_base = base.predict_proba(base_matrix[target])[:, 1]
        scores["calibrated_anchor"][target] = test_base
        residual = y[rows] - train_base
        for kind in ("hist", "extra"):
            for state_name, matrix in (("aligned", aligned), ("stale20", stale)):
                candidate = f"residual_{kind}"
                if state_name == "stale20":
                    candidate += "_stale20"
                model = _residual_model(kind)
                model.fit(matrix[rows], residual)
                correction = model.predict(matrix[target])
                scores[candidate][target] = (
                    test_base + RESIDUAL_WEIGHT * correction
                )
                logs.append({
                    "candidate": candidate,
                    "quarter": str(start),
                    "n_train": int(len(rows)),
                    "n_test": int(len(target)),
                    "last_resolved": str(max(reach[rows])),
                    "n_features": int(matrix.shape[1]),
                    "residual_weight": RESIDUAL_WEIGHT,
                })
        if verbose:
            print(
                f"  fixing residual quarter={start} train={len(rows):5d} "
                f"test={len(target):4d}", flush=True,
            )
    return scores, logs


def outcome_causality_check(base_matrix, aligned, stale, y, dates, reach):
    cutoff = dt.date(2025, 6, 30)
    original, _ = prequential_scores(
        base_matrix, aligned, stale, y, dates, reach, verbose=False,
    )
    changed_y = y.copy()
    unresolved = np.asarray([
        np.isfinite(y[row]) and reach[row] > cutoff for row in range(len(y))
    ])
    changed_y[unresolved] = 1.0 - changed_y[unresolved]
    changed, _ = prequential_scores(
        base_matrix, aligned, stale, changed_y, dates, reach, verbose=False,
    )
    past = dates <= cutoff
    for candidate in original:
        available = past & np.isfinite(original[candidate])
        np.testing.assert_array_equal(
            original[candidate][available], changed[candidate][available],
        )
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    history, digest = load_spot_1530_history()
    spot, spot_names = build_spot_1530_features(index, history, references)
    proxy_causality_check(index, history, references)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ], dtype=int)
    target_columns = np.asarray([names.index(name) for name in TARGET_FEATURES])
    currency = X[:, currency_columns]
    static = np.column_stack((currency, X[:, target_columns]))
    anchor = spot[:, spot_names.index(
        "moex_1530_cnyrub_tom_mean_cbr_basis"
    )]
    anchor_rank = causal_percentiles(
        anchor, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    base_matrix = np.column_stack((anchor_rank, currency))
    aligned = np.column_stack((static, anchor_rank, spot))
    stale_spot = delayed_by_currency(spot, index, rows=STALE_ROWS)
    stale = np.column_stack((static, anchor_rank, stale_spot))
    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    outcome_causality_check(base_matrix, aligned, stale, y5, dates, reach)
    raw_scores, logs = prequential_scores(
        base_matrix, aligned, stale, y5, dates, reach,
    )

    fixing = _load(FIXING_PATH, "selected")
    router = _load(ROUTER_PATH, "availability_route")
    learned = {
        name: _outputs(score, y5, dates)
        for name, score in raw_scores.items()
    }
    learned["residual_mean"] = combine_causal(
        (learned["residual_hist"], learned["residual_extra"]),
        (.5, .5), dates, currencies,
    )
    learned["residual_mean_stale20"] = combine_causal(
        (
            learned["residual_hist_stale20"],
            learned["residual_extra_stale20"],
        ),
        (.5, .5), dates, currencies,
    )
    candidates = {
        "fixing_basis": fixing,
        "availability_router": router,
        "calibrated_anchor": learned["calibrated_anchor"],
        **{name: learned[name] for name in RAW_RESIDUALS},
    }
    matched_stale = {
        "residual_hist": learned["residual_hist_stale20"],
        "residual_extra": learned["residual_extra_stale20"],
        "residual_mean": learned["residual_mean_stale20"],
    }
    for name in RAW_RESIDUALS:
        candidate = f"router75_{name}25"
        candidates[candidate] = combine_causal(
            (router, learned[name]),
            (ROUTER_WEIGHT, 1.0 - ROUTER_WEIGHT), dates, currencies,
        )
        matched_stale[candidate] = combine_causal(
            (router, matched_stale[name]),
            (ROUTER_WEIGHT, 1.0 - ROUTER_WEIGHT), dates, currencies,
        )

    screen = horizon_rows(
        candidates, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    comparison = {
        "fixing_basis": fixing,
        "availability_router": router,
        "selected": candidates[selected],
    }
    if selected in matched_stale:
        comparison["matched_stale20"] = matched_stale[selected]
    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    screen_summary.to_csv(OUT / "screen_2024_summary.csv", index=False)
    pd.DataFrame(logs).to_csv(OUT / "training_log.csv", index=False)

    later_parts = []
    for period, years in (
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        part = horizon_rows(
            comparison, years, targets, forwards, dates, currencies,
        )
        part["period"] = period
        later_parts.append(part)
    later = pd.concat(later_parts, ignore_index=True)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    later_summary = summarize(later[later.period == "combined_2025_2026"])
    later_summary.to_csv(OUT / "later_summary.csv", index=False)

    h5_rows = []
    for candidate, output in comparison.items():
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
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(comparison, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bootstrap, masks, valid = _bootstrap(
        h5[h5.period == "screen_2024"], comparison, (2025, 2026),
        y5, forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        y5, dates, currencies, valid, masks, "fixing_residual_boost_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    chronology_ok = bool(np.all(
        pd.to_datetime(pd.DataFrame(logs).last_resolved)
        < pd.to_datetime(pd.DataFrame(logs).quarter)
    ))
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "EL",
        "anchor": "positive 15:30 CNY session-mean basis",
        "fixed_policy": POLICY,
        "training_start": str(TRAIN_START),
        "quarterly_refit": True,
        "rank_window": RANK_WINDOW,
        "rank_minimum": RANK_MINIMUM,
        "base_logit_c": .05,
        "target_features": TARGET_FEATURES,
        "spot_feature_count": len(spot_names),
        "residual_weight": RESIDUAL_WEIGHT,
        "router_weight": ROUTER_WEIGHT,
        "hist_residual": {
            "max_iter": 150, "learning_rate": .03,
            "max_leaf_nodes": 5, "min_samples_leaf": 100,
            "l2_regularization": 30.0,
        },
        "extra_residual": {
            "n_estimators": 400, "max_depth": 6,
            "min_samples_leaf": 40, "max_features": .65,
        },
        "stale_control_rows": STALE_ROWS,
        "payload_sha256": digest,
        "all_training_labels_resolved": chronology_ok,
        "future_outcome_corruption_check": True,
        "selection_period": 2024,
        "selected": selected,
        "next_cbr_rate_used": False,
        "later_period_status": (
            "protocol-controlled retrospective opened after 2024 selection"
        ),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if not chronology_ok:
        raise AssertionError("fixing residual training chronology failed")
    print("Selected on 2024:", selected)
    print("\nSCREEN\n" + screen_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).to_string(index=False))
    print("\nLATER\n" + later_summary.to_string(index=False))
    print("\nLATER BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
