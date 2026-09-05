"""Packet-CK: a purged committee trained jointly for h=1/3/5/10/20."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import HORIZONS, benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import RESET, _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_armenian_central_bank_features import build_cba_features, load_cba
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_basis_features import build_cny_basis_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_survival_hazard import TARGET_FIELDS
from research.round6_cny_waveform_features import build_waveform_features
from research.round6_local_central_bank_features import build_nbt_features, load_nbt
from research.round6_moex_features import load_moex_history
from research.round6_multihorizon_case_audit import corridor_period_adjusted_lift
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate, _fire


OUT = Path("results/research/round6/long_horizon_committee")
INCUMBENT_PATH = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
INCUMBENT = "geometry75_cba_consensus_basis25"
SEED = 20260905
AGGREGATES = (
    "long_logit_direct_h5", "long_logit_geometric", "long_logit_harmonic",
    "long_logit_minimum", "long_logit_weighted",
    "long_extra_direct_h5", "long_extra_geometric", "long_extra_weighted",
    "long_logit_geometric_stale20",
)


def _model(kind):
    if kind == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.025, max_iter=3000, random_state=SEED),
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=350, max_depth=7, min_samples_leaf=28,
            max_features=.65, n_jobs=-1, random_state=SEED,
        )
    raise KeyError(kind)


def prequential_probabilities(matrix, labels, dates, reach, kind, verbose=True):
    result = np.full((len(dates), len(HORIZONS)), np.nan)
    logs = []
    complete = np.all(np.isfinite(labels), axis=1)
    for start in _quarter_starts():
        end = _next_quarter(start)
        train = np.flatnonzero(
            (dates >= RESET) & complete
            & np.asarray([value < start for value in reach])
        )
        # Scoring rows do not need an h=20 outcome yet.  Requiring complete
        # labels here silently removed the last 20 observations per corridor
        # and made the committee incomparable with the incumbent at h=1/3/5.
        # Only training rows need fully resolved labels.
        target = np.flatnonzero((dates >= start) & (dates < end))
        if len(train) < 700 or not len(target):
            continue
        if not all(reach[row] < start for row in train):
            raise AssertionError("unresolved h=20 label admitted")
        for column, h in enumerate(HORIZONS):
            model = _model(kind).fit(matrix[train], labels[train, column])
            result[target, column] = model.predict_proba(matrix[target])[:, 1]
        logs.append({
            "model": kind, "quarter": str(start), "n_train": len(train),
            "last_resolved": str(max(reach[train])), "n_features": matrix.shape[1],
        })
        if verbose:
            print(
                f"  long {kind:<5} quarter={start} train={len(train):5d} "
                f"features={matrix.shape[1]:3d}", flush=True,
            )
    return result, logs


def aggregate(probabilities, prefix):
    clipped = np.clip(probabilities, 1e-9, 1.0)
    weights = np.asarray((.10, .15, .25, .25, .25), dtype=float)
    all_missing = np.all(~np.isfinite(probabilities), axis=1)
    with np.errstate(all="ignore"):
        result = {
        f"{prefix}_direct_h5": probabilities[:, HORIZONS.index(5)],
        f"{prefix}_geometric": np.exp(np.nanmean(np.log(clipped), axis=1)),
        f"{prefix}_harmonic": 1.0 / np.nanmean(1.0 / clipped, axis=1),
        f"{prefix}_minimum": np.nanmin(probabilities, axis=1),
        f"{prefix}_weighted": np.exp(np.nansum(np.log(clipped) * weights, axis=1)),
        }
    for values in result.values():
        values[all_missing] = np.nan
    return result


def outcome_causality_check(matrix, labels, dates, reach):
    cutoff = np.datetime64("2025-06-30").astype(object)
    first, _ = prequential_probabilities(matrix, labels, dates, reach, "logit", verbose=False)
    changed = labels.copy()
    future = np.asarray([value > cutoff for value in reach])
    changed[future] = 1.0 - changed[future]
    second, _ = prequential_probabilities(matrix, changed, dates, reach, "logit", verbose=False)
    past = (dates <= cutoff) & np.all(np.isfinite(first), axis=1)
    np.testing.assert_array_equal(first[past], second[past])
    return True


def _forward(series, index, h):
    result = np.full(len(index), np.nan)
    for row, (currency, position, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, position, h)
        if value is not None:
            result[row] = value
    return result


def _screen_rows(outputs, targets, series, index, dates, currencies):
    rows = []
    for candidate, output in outputs.items():
        for h in HORIZONS:
            y = targets[f"fav_h{h}"]
            valid, fired = _fire(output, (2024,), POLICY, y, dates, currencies)
            active = valid & fired
            case_lift, _base, _macro = corridor_period_adjusted_lift(
                y, valid, fired, currencies, dates, (2024,),
            )
            rows.append({
                "candidate": candidate, "horizon": h, "case_lift": case_lift,
                "symmetric_benefit_bps": float(np.nanmean(
                    targets[f"benefit_h{h}"][active]
                )),
                "future_only_benefit_bps": float(np.nanmean(
                    _forward(series, index, h)[active]
                )),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    labels = np.column_stack([targets[f"fav_h{h}"] for h in HORIZONS])
    reach = target_reach_dates(index, series, 20)
    history, moex_digest = load_moex_history()
    _broad, _broad_names, references = load_broad_features(index, series)
    basis, basis_names = build_cny_basis_features(index, history, references["CNY"])
    wave, wave_names = build_waveform_features(index, history)
    cba, cba_digest = load_cba()
    cba_matrix, cba_names = build_cba_features(index, series, references, cba)
    nbt, nbt_digest = load_nbt()
    nbt_matrix, nbt_names = build_nbt_features(index, series, references, nbt)
    target_columns = np.asarray([names.index(name) for name in TARGET_FIELDS])
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ])
    local = np.column_stack([X[:, target_columns], X[:, currency_columns]])
    external = np.column_stack([basis, wave, cba_matrix, nbt_matrix])
    aligned = np.column_stack([local, external])
    stale = np.column_stack([local, delayed_by_currency(external, index, rows=20)])
    outcome_causality_check(aligned, labels, dates, reach)
    logit, logit_logs = prequential_probabilities(aligned, labels, dates, reach, "logit")
    extra, extra_logs = prequential_probabilities(aligned, labels, dates, reach, "extra")
    stale_logit, stale_logs = prequential_probabilities(stale, labels, dates, reach, "logit")
    scores = aggregate(logit, "long_logit")
    scores.update({
        key: value for key, value in aggregate(extra, "long_extra").items()
        if key in AGGREGATES
    })
    scores["long_logit_geometric_stale20"] = aggregate(
        stale_logit, "stale",
    )["stale_geometric"]
    outputs = {name: _outputs(scores[name], targets["fav_h5"], dates) for name in AGGREGATES}
    screen_detail = _screen_rows(outputs, targets, series, index, dates, currencies)
    screen_summary = screen_detail.groupby("candidate", as_index=False).agg(
        horizon_lift_min=("case_lift", "min"),
        horizon_lift_mean=("case_lift", "mean"),
        symmetric_benefit_min=("symmetric_benefit_bps", "min"),
        future_benefit_min=("future_only_benefit_bps", "min"),
    )
    fresh = screen_summary.candidate != "long_logit_geometric_stale20"
    feasible = screen_summary[
        fresh & screen_summary.symmetric_benefit_min.gt(0)
        & screen_summary.future_benefit_min.gt(0)
    ]
    pool = feasible if len(feasible) else screen_summary[fresh]
    selected = str(pool.sort_values(
        ["horizon_lift_min", "horizon_lift_mean", "symmetric_benefit_min"],
        ascending=False,
    ).iloc[0].candidate)
    incumbent = pickle.load(INCUMBENT_PATH.open("rb"))[INCUMBENT]
    comparison = {"long_selected": outputs[selected], "incumbent": incumbent}
    for weight in (.10, .25):
        comparison[f"incumbent{int((1-weight)*100)}_long{int(weight*100)}"] = combine_causal(
            [incumbent, outputs[selected]], (1.0 - weight, weight), dates, currencies,
        )
    screen_detail.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    screen_summary.to_csv(OUT / "screen_2024_summary.csv", index=False)
    later_detail = []
    for candidate, output in comparison.items():
        for period, years in (
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            for h in HORIZONS:
                y = targets[f"fav_h{h}"]
                valid, fired = _fire(output, years, POLICY, y, dates, currencies)
                active = valid & fired
                case_lift, _base, _macro = corridor_period_adjusted_lift(
                    y, valid, fired, currencies, dates, years,
                )
                later_detail.append({
                    "candidate": candidate, "period": period, "horizon": h,
                    "case_lift": case_lift,
                    "symmetric_benefit_bps": float(np.nanmean(targets[f"benefit_h{h}"][active])),
                    "future_only_benefit_bps": float(np.nanmean(_forward(series, index, h)[active])),
                })
    later = pd.DataFrame(later_detail)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    later_summary = later[later.period == "combined_2025_2026"].groupby(
        "candidate", as_index=False,
    ).agg(
        horizon_lift_min=("case_lift", "min"),
        horizon_lift_mean=("case_lift", "mean"),
        symmetric_benefit_min=("symmetric_benefit_bps", "min"),
        future_benefit_min=("future_only_benefit_bps", "min"),
    )
    later_summary.to_csv(OUT / "later_summary.csv", index=False)
    y5 = targets["fav_h5"]
    benefit5 = _forward(series, index, 5)
    standard = []
    for candidate, output in comparison.items():
        for period, years in (
            ("screen_2024", (2024,)), ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)), ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(output, years, POLICY, y5, benefit5, dates, currencies)
            item.update({"candidate": candidate, "period": period, **POLICY})
            standard.append(item)
    standard = pd.DataFrame(standard)
    standard.to_csv(OUT / "standard_h5_results.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(comparison, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bootstrap, masks, valid = _bootstrap(
        standard[standard.period == "screen_2024"], comparison,
        (2025, 2026), y5, benefit5, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        y5, dates, currencies, valid, masks, "long_horizon_committee_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, benefit5, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    pd.DataFrame(logit_logs + extra_logs + [
        {"model": "stale20_" + row["model"], **{k: v for k, v in row.items() if k != "model"}}
        for row in stale_logs
    ]).to_csv(OUT / "training_log.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CK", "fixed_policy": POLICY, "selection_period": 2024,
        "selection_objective": "maximum worst case-lift over h=1/3/5/10/20",
        "selection_constraints": "positive symmetric and future-only benefit at every h",
        "selected": selected, "target_horizons": HORIZONS,
        "purge_horizon": 20, "all_training_labels_resolved": True,
        "future_outcome_corruption_check": True,
        "feature_count": aligned.shape[1],
        "payload_sha256": {"moex": moex_digest, "cba": cba_digest, "nbt": nbt_digest},
        "asof_rule": "market/local-CB observations strictly before signal date",
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected on 2024: {selected}\n")
    print(screen_summary.sort_values("horizon_lift_min", ascending=False).to_string(index=False))
    print("\nLATER\n" + later_summary.to_string(index=False))
    print("\nH5\n" + standard.to_string(index=False))


if __name__ == "__main__":
    main()
