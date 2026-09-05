"""Packet-CO: causal router between the incumbent and resolved-error regime expert."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_armenian_central_bank_features import build_cba_features, load_cba
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_error_regime import row_scores
from research.round6_cny_survival_hazard import TARGET_FIELDS
from research.round6_cny_waveform_features import build_waveform_features
from research.round6_moex_features import load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose, _forward, horizon_rows, summarize,
)


OUT = Path("results/research/round6/multihorizon_error_router")
INCUMBENT_PATH = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
REGIME_PATH = Path("results/research/round6/cny_error_regime/outputs.pkl")
INCUMBENT = "geometry75_cba_consensus_basis25"
REGIME = "primary75_regime_logit25"
SEED = 20260905
TAIL_WEIGHTS = np.asarray((.10, .15, .20, .25, .30), dtype=float)


def _load(path: Path, name: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[name]


def router_labels(incumbent_score, regime_score, labels, weights):
    inc_loss = np.sum((incumbent_score[:, None] - labels) ** 2 * weights, axis=1)
    regime_loss = np.sum((regime_score[:, None] - labels) ** 2 * weights, axis=1)
    result = (regime_loss < inc_loss).astype(float)
    result[~np.all(np.isfinite(labels), axis=1)] = np.nan
    result[~np.isfinite(incumbent_score) | ~np.isfinite(regime_score)] = np.nan
    return result


def router_matrix(incumbent_score, regime_score, X, names, cba, cba_names, wave, wave_names):
    pair = np.column_stack((incumbent_score, regime_score))
    currency_columns = [i for i, name in enumerate(names) if name.startswith("currency_")]
    target_columns = [names.index(name) for name in TARGET_FIELDS]
    cba_fields = (
        "cba_consensus_basis_bps", "cba_direct_minus_usd_bps",
        "cba_direct_minus_cny_bps", "cba_usd_minus_cny_bps",
        "cba_rub_quote_ret_1", "cba_rub_quote_ret_5", "cba_rub_quote_ret_20",
        "cba_rub_age_days",
    )
    wave_fields = (
        "cny_wave_vol_5", "cny_wave_vol_20", "cny_wave_last_z_20",
        "cny_wave_acceleration_5_5",
    )
    return np.column_stack((
        pair,
        regime_score - incumbent_score,
        np.abs(regime_score - incumbent_score),
        np.min(pair, axis=1), np.max(pair, axis=1), np.mean(pair, axis=1),
        X[:, currency_columns], X[:, target_columns],
        cba[:, [cba_names.index(name) for name in cba_fields]],
        wave[:, [wave_names.index(name) for name in wave_fields]],
    ))


def _model(kind):
    if kind == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.02, max_iter=3000, random_state=SEED),
        )
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.03, max_leaf_nodes=5,
            min_samples_leaf=80, l2_regularization=25.0,
            random_state=SEED,
        )
    if kind == "tree":
        return DecisionTreeClassifier(
            max_depth=2, min_samples_leaf=150, random_state=SEED,
        )
    raise KeyError(kind)


def prequential_router(kind, gate, route_y, incumbent_score, regime_score, dates, reach, verbose=True):
    soft = np.full(len(dates), np.nan)
    hard = np.full(len(dates), np.nan)
    weights = np.full(len(dates), np.nan)
    logs = []
    finite = np.all(np.isfinite(gate), axis=1)
    for start in _quarter_starts():
        if start.year < 2024:
            continue
        end = _next_quarter(start)
        train = np.flatnonzero(
            (dates >= dt.date(2023, 1, 1)) & finite & np.isfinite(route_y)
            & np.asarray([value < start for value in reach])
        )
        target = np.flatnonzero((dates >= start) & (dates < end) & finite)
        if len(train) < 700 or not len(target):
            continue
        if not all(reach[row] < start for row in train):
            raise AssertionError("unresolved h=20 router label admitted")
        model = _model(kind).fit(gate[train], route_y[train])
        weight = model.predict_proba(gate[target])[:, 1]
        weights[target] = weight
        soft[target] = (
            weight * regime_score[target]
            + (1.0 - weight) * incumbent_score[target]
        )
        hard[target] = np.where(
            weight >= .5, regime_score[target], incumbent_score[target],
        )
        logs.append({
            "kind": kind, "quarter": str(start), "n_train": len(train),
            "last_resolved": str(max(reach[train])),
            "regime_weight_mean": float(np.mean(weight)),
            "n_features": gate.shape[1],
        })
        if verbose:
            print(
                f"  router {kind:<5} quarter={start} train={len(train):5d} "
                f"regime_weight={np.mean(weight):.3f}", flush=True,
            )
    return soft, hard, weights, logs


def outcome_causality_check(gate, incumbent_score, regime_score, labels, dates, reach):
    cutoff = dt.date(2025, 6, 30)
    original_y = router_labels(
        incumbent_score, regime_score, labels, np.full(len(HORIZONS), .2),
    )
    original, *_ = prequential_router(
        "logit", gate, original_y, incumbent_score, regime_score,
        dates, reach, verbose=False,
    )
    changed = labels.copy()
    future = np.asarray([value > cutoff for value in reach])
    changed[future] = 1.0 - changed[future]
    changed_y = router_labels(
        incumbent_score, regime_score, changed, np.full(len(HORIZONS), .2),
    )
    altered, *_ = prequential_router(
        "logit", gate, changed_y, incumbent_score, regime_score,
        dates, reach, verbose=False,
    )
    past = (dates <= cutoff) & np.isfinite(original)
    np.testing.assert_array_equal(original[past], altered[past])
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    labels = np.column_stack([targets[f"fav_h{h}"] for h in HORIZONS])
    reach = target_reach_dates(index, series, 20)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    incumbent = _load(INCUMBENT_PATH, INCUMBENT)
    regime = _load(REGIME_PATH, REGIME)
    incumbent_score = row_scores(incumbent, len(index))
    regime_score = row_scores(regime, len(index))
    _broad, _broad_names, references = load_broad_features(index, series)
    cba_source, cba_digest = load_cba()
    cba, cba_names = build_cba_features(index, series, references, cba_source)
    moex, moex_digest = load_moex_history()
    wave, wave_names = build_waveform_features(index, moex)
    gate = router_matrix(
        incumbent_score, regime_score, X, names, cba, cba_names, wave, wave_names,
    )
    stale_gate = delayed_by_currency(gate, index, rows=20)
    outcome_causality_check(
        gate, incumbent_score, regime_score, labels, dates, reach,
    )
    route_targets = {
        "equal": router_labels(
            incumbent_score, regime_score, labels, np.full(len(HORIZONS), .2),
        ),
        "tail": router_labels(
            incumbent_score, regime_score, labels, TAIL_WEIGHTS,
        ),
    }
    outputs = {
        "incumbent": incumbent,
        "regime": regime,
        "incumbent50_regime50": combine_causal(
            [incumbent, regime], (.5, .5), dates, currencies,
        ),
    }
    logs = []
    weight_columns = {}
    for objective, route_y in route_targets.items():
        for kind in ("logit", "hist", "tree"):
            soft, hard, weights, part = prequential_router(
                kind, gate, route_y, incumbent_score, regime_score, dates, reach,
            )
            base = f"router_{objective}_{kind}"
            outputs[base + "_soft"] = _outputs(soft, targets["fav_h5"], dates)
            outputs[base + "_hard"] = _outputs(hard, targets["fav_h5"], dates)
            weight_columns[base] = weights
            for row in part:
                row["objective"] = objective
                logs.append(row)
    stale_soft, _stale_hard, stale_weights, stale_logs = prequential_router(
        "logit", stale_gate, route_targets["equal"], incumbent_score,
        regime_score, dates, reach,
    )
    outputs["router_equal_logit_soft_stale20"] = _outputs(
        stale_soft, targets["fav_h5"], dates,
    )
    weight_columns["router_equal_logit_stale20"] = stale_weights
    for row in stale_logs:
        row["objective"] = "equal_stale20"
        logs.append(row)
    pd.DataFrame(logs).to_csv(OUT / "training_log.csv", index=False)
    pd.DataFrame({
        "date": dates, "currency": currencies, **weight_columns,
    }).to_csv(OUT / "router_weights.csv", index=False)

    screen = horizon_rows(
        outputs, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    screen_summary.to_csv(OUT / "screen_2024_summary.csv", index=False)
    later_parts = []
    for period, years in (
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        part = horizon_rows(outputs, years, targets, forwards, dates, currencies)
        part["period"] = period
        later_parts.append(part)
    later = pd.concat(later_parts, ignore_index=True)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    later_summary = summarize(later[later.period == "combined_2025_2026"])
    later_summary.to_csv(OUT / "later_summary.csv", index=False)
    h5_rows = []
    for candidate, output in outputs.items():
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(
                output, years, POLICY, targets["fav_h5"], forwards[5],
                dates, currencies,
            )
            item.update({"candidate": candidate, "period": period, **POLICY})
            h5_rows.append(item)
    h5 = pd.DataFrame(h5_rows)
    h5.to_csv(OUT / "standard_h5_results.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bootstrap, masks, valid = _bootstrap(
        h5[h5.period == "screen_2024"], outputs, (2025, 2026),
        targets["fav_h5"], forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        targets["fav_h5"], dates, currencies, valid, masks,
        "multihorizon_error_router_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in outputs.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            targets["fav_h5"], forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    chronology = pd.DataFrame(logs)
    chronology_ok = bool(np.all(
        pd.to_datetime(chronology.last_resolved) < pd.to_datetime(chronology.quarter)
    ))
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CO", "fixed_policy": POLICY,
        "experts": [INCUMBENT, REGIME],
        "selection_period": 2024, "selected": selected,
        "router_label": "expert with lower weighted mean Brier loss over resolved h=1/3/5/10/20 outcomes",
        "objectives": {"equal": [.2] * 5, "tail": TAIL_WEIGHTS.tolist()},
        "training_start": "2023-01-01", "purge_horizon": 20,
        "all_training_labels_resolved": chronology_ok,
        "future_outcome_corruption_check": True,
        "stale_control_rows_per_currency": 20,
        "payload_sha256": {"CBA": cba_digest, "MOEX": moex_digest},
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not chronology_ok:
        raise AssertionError("router chronology failed")
    print(f"Selected on 2024: {selected}\n")
    print("SCREEN\n" + screen_summary.sort_values(
        "horizon_lift_min", ascending=False,
    ).to_string(index=False))
    print("\nLATER\n" + later_summary.sort_values(
        "horizon_lift_min", ascending=False,
    ).to_string(index=False))


if __name__ == "__main__":
    main()
