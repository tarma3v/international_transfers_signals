"""T46: per-currency 15:30 probability head with exact CNY fallback."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS
from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.round5_features import load_round5_features
from research.temperature_t45_direct_pair_benefit import aligned_direct_panel


OUT = Path("results/research/temperature/t46_local_pair_fallback")
BASE = Path("results/research/temperature/t5_market_grid/outputs.npz")
DIRECT = Path("results/research/round7/direct_pairs/direct_market_panel.csv")
REGISTERED = Path("research/temperature_t46_local_pair_fallback_registered.md")
MIN_TRAIN_DATE = dt.date(2022, 2, 24)
EMBARGO_DAYS = 2
MIN_TRAIN_ROWS = 150
SEED = 20260907
BLOCKS = (20, 50)
BOOTSTRAP_DRAWS = 2000
FEATURE_NAMES = (
    "cny_basis_1530", "cny_past_rank_1530", "direct_mean_basis",
    "direct_last_basis", "direct_previous_basis", "direct_intraday_return",
    "direct_range_bps", "direct_log_count", "direct_age_hours",
    "direct_quality", "direct_log_tom_count", "direct_log_tod_count",
)


def loadz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as source:
        return {key: source[key] for key in source.files}


def clipped_logit(probability):
    probability = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def direct_probability_features(panel: pd.DataFrame, cny_raw, cny_rank):
    basis = panel[["mean_basis", "last_basis", "previous_basis"]].to_numpy(float)
    basis = np.clip(basis, -5000.0, 5000.0)
    values = np.column_stack([
        np.asarray(cny_raw, float),
        np.asarray(cny_rank, float),
        basis,
        panel.intraday_return.to_numpy(float),
        panel.range_bps.to_numpy(float),
        np.log1p(panel["count"].fillna(0.0).to_numpy(float)),
        np.clip(panel.age_minutes.fillna(720.0).to_numpy(float), 0, 720) / 60,
        panel.quality.fillna(0.0).to_numpy(float),
        np.log1p(panel.tom_count.fillna(0.0).to_numpy(float)),
        np.log1p(panel.tod_count.fillna(0.0).to_numpy(float)),
    ])
    values[~np.isfinite(values)] = 0.0
    assert values.shape[1] == len(FEATURE_NAMES)
    return values


def quarter_origins(dates):
    return [
        dt.date(year, month, 1)
        for year in range(2023, max(day.year for day in dates) + 1)
        for month in (1, 4, 7, 10)
    ]


def next_quarter(origin):
    return (pd.Timestamp(origin).to_period("Q") + 1).start_time.date()


def fit_local_heads(matrix, target, baseline, maturity, dates, currencies,
                    hard, quality):
    target = np.asarray(target, float)
    baseline = np.asarray(baseline, float)
    dates = np.asarray(dates, object)
    currencies = np.asarray(currencies)
    maturity = np.asarray(maturity, object)
    hard = np.asarray(hard, bool)
    quality = np.asarray(quality, float)
    candidate = baseline.copy()
    local_probability = np.full(len(dates), np.nan)
    active = np.zeros(len(dates), bool)
    n_train = np.zeros(len(dates), np.int32)
    logs = []
    for origin in quarter_origins(dates):
        end = next_quarter(origin)
        cutoff = origin - dt.timedelta(days=EMBARGO_DAYS)
        mature = np.asarray([value < cutoff for value in maturity], bool)
        for currency in CORRIDORS:
            train = (
                (currencies == currency) & (dates >= MIN_TRAIN_DATE)
                & (dates < origin) & mature & hard & np.isfinite(target)
                & np.all(np.isfinite(matrix), axis=1)
            )
            query = (
                (currencies == currency) & (dates >= origin) & (dates < end)
                & hard & np.isfinite(baseline)
                & np.all(np.isfinite(matrix), axis=1)
            )
            fitted = train.sum() >= MIN_TRAIN_ROWS and np.unique(target[train]).size == 2
            n_train[query] = int(train.sum())
            if fitted:
                ids = np.flatnonzero(train)
                age_days = np.asarray([(origin - dates[i]).days for i in ids], float)
                weights = np.exp2(-age_days / 730.0) * quality[train]
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=0.05, penalty="l2", solver="lbfgs", max_iter=2000,
                        random_state=SEED,
                    ),
                )
                model.fit(matrix[train], target[train].astype(int),
                          logisticregression__sample_weight=weights)
                if query.any():
                    local = model.predict_proba(matrix[query])[:, 1]
                    q = np.clip(quality[query], 0.0, 1.0)
                    mixed = (1 - q) * clipped_logit(baseline[query]) + q * clipped_logit(local)
                    candidate[query] = 1 / (1 + np.exp(-mixed))
                    local_probability[query] = local
                    active[query] = True
                coefficients = model.named_steps["logisticregression"].coef_[0].tolist()
                intercept = float(model.named_steps["logisticregression"].intercept_[0])
                last_maturity = str(max(maturity[train]))
            else:
                coefficients = []
                intercept = np.nan
                last_maturity = ""
            logs.append({
                "origin": str(origin), "currency": currency,
                "n_train": int(train.sum()), "n_query": int(query.sum()),
                "model_fit": bool(fitted), "last_target_maturity": last_maturity,
                "intercept": intercept, "coefficients": coefficients,
            })
    return candidate, local_probability, active, n_train, logs


def ece(y, probability):
    y = np.asarray(y, float)
    probability = np.asarray(probability, float)
    edges = np.linspace(0, 1, 11)
    bins = np.minimum(np.searchsorted(edges, probability, side="right") - 1, 9)
    return float(sum(
        np.mean(bins == index) * abs(probability[bins == index].mean() - y[bins == index].mean())
        for index in range(10) if np.any(bins == index)
    ))


def metrics(target, candidate, baseline, mask):
    mask = (
        np.asarray(mask, bool) & np.isfinite(target) & np.isfinite(candidate)
        & np.isfinite(baseline)
    )
    y = np.asarray(target, float)[mask].astype(int)
    p = np.clip(np.asarray(candidate, float)[mask], 1e-6, 1 - 1e-6)
    b = np.clip(np.asarray(baseline, float)[mask], 1e-6, 1 - 1e-6)
    if not len(y):
        return {"n": 0, **{key: np.nan for key in (
            "brier_candidate", "brier_baseline", "brier_delta",
            "logloss_candidate", "logloss_baseline", "logloss_delta",
            "auc_candidate", "auc_baseline", "auc_delta", "ap_candidate",
            "ap_baseline", "ece_candidate", "ece_baseline", "ece_delta",
        )}}
    bc, bb = np.mean((p - y) ** 2), np.mean((b - y) ** 2)
    lc = log_loss(y, p, labels=[0, 1])
    lb = log_loss(y, b, labels=[0, 1])
    if np.unique(y).size == 2:
        ac, ab = roc_auc_score(y, p), roc_auc_score(y, b)
        apc, apb = average_precision_score(y, p), average_precision_score(y, b)
    else:
        ac = ab = apc = apb = np.nan
    ec, eb = ece(y, p), ece(y, b)
    return {
        "n": int(len(y)), "brier_candidate": float(bc),
        "brier_baseline": float(bb), "brier_delta": float(bc - bb),
        "logloss_candidate": float(lc), "logloss_baseline": float(lb),
        "logloss_delta": float(lc - lb), "auc_candidate": float(ac),
        "auc_baseline": float(ab), "auc_delta": float(ac - ab),
        "ap_candidate": float(apc), "ap_baseline": float(apb),
        "ece_candidate": ec, "ece_baseline": eb, "ece_delta": ec - eb,
    }


def metric_grid(target, candidate, baseline, active, dates, currencies,
                h, period, years):
    scope = np.asarray([day.year in years for day in dates], bool)
    rows = [
        {"h": h, "period": period, "slice": "all", "group": "ALL",
         **metrics(target, candidate, baseline, scope)},
        {"h": h, "period": period, "slice": "active_local", "group": "ALL",
         **metrics(target, candidate, baseline, scope & active)},
    ]
    for currency in CORRIDORS:
        mask = scope & active & (currencies == currency)
        rows.append({
            "h": h, "period": period, "slice": "active_currency",
            "group": currency, **metrics(target, candidate, baseline, mask),
        })
    return rows


def bootstrap_rows(target, candidate, baseline, dates, h, period, years):
    frame = pd.DataFrame({
        "date": dates, "target": target, "candidate": candidate,
        "baseline": baseline,
    })
    valid = (
        frame.date.map(lambda day: day.year in years)
        & np.isfinite(frame.target) & np.isfinite(frame.candidate)
        & np.isfinite(frame.baseline)
    )
    frame = frame[valid].copy()
    frame["delta"] = ((frame.candidate - frame.target) ** 2
                      - (frame.baseline - frame.target) ** 2)
    daily = frame.groupby("date", sort=True).delta.mean().to_numpy(float)
    output = []
    for block in BLOCKS:
        rng = np.random.default_rng(SEED + h * 1000 + block + min(years))
        n = len(daily)
        n_blocks = int(np.ceil(n / block))
        starts = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n_blocks))
        ids = ((starts[:, :, None] + np.arange(block)) % n)
        draws = daily[ids.reshape(BOOTSTRAP_DRAWS, -1)[:, :n]].mean(axis=1)
        output.append({
            "h": h, "period": period, "block_dates": block,
            "n_dates": n, "point_brier_delta": float(daily.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "p_improvement": float(np.mean(draws < 0)),
        })
    return output


def gates(metrics_frame, bootstrap):
    rows = []
    for period in ("screen_2023", "validation_2024"):
        for h in HORIZONS:
            part = metrics_frame[(metrics_frame.period == period) & (metrics_frame.h == h)]
            all_row = part[(part.slice == "all") & (part.group == "ALL")].iloc[0]
            active_row = part[(part.slice == "active_local") & (part.group == "ALL")].iloc[0]
            dense = part[(part.slice == "active_currency") & part.group.isin(("AMD", "KZT"))]
            boot = bootstrap[(bootstrap.period == period) & (bootstrap.h == h)]
            checks = {
                "pooled_brier_better": all_row.brier_delta < 0,
                "pooled_logloss_noninferior": all_row.logloss_delta <= 0,
                "pooled_auc_noninferior": all_row.auc_delta >= -0.005,
                "active_brier_better": active_row.brier_delta < 0,
                "amd_kzt_supported": (
                    len(dense) == 2 and dense.n.ge(30).all()
                    and dense.brier_delta.le(0).all()
                ),
                "bootstrap_upper_nonpositive": (
                    len(boot) == len(BLOCKS) and boot.ci_high.le(0).all()
                ),
            }
            rows.append({"period": period, "h": h, **checks,
                         "stage_pass": bool(all(checks.values()))})
    return pd.DataFrame(rows)


def run_experiment():
    _matrix, _names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], object)
    currencies = np.asarray([row[0] for row in index])
    baseline = loadz(BASE)
    np.testing.assert_array_equal(baseline["currencies"], currencies)
    np.testing.assert_array_equal(
        np.asarray([str(day) for day in dates]), baseline["dates"].astype(str)
    )
    panel = aligned_direct_panel(index)
    hard = panel.hard_quality.to_numpy(bool)
    quality = panel.quality.to_numpy(float)
    matrix = direct_probability_features(
        panel, baseline["raw__cutoff_1530"], baseline["rank__cutoff_1530"]
    )
    targets = build_targets(series, index)
    raw = {}
    metric_rows, bootstrap_rows_all, logs_all = [], [], []
    for h in HORIZONS:
        target = targets[f"fav_h{h}"]
        base = baseline[f"prob__cutoff_1530__h{h}"].astype(float)
        maturity = np.asarray(target_reach_dates(index, series, h), object)
        candidate, local, active, n_train, logs = fit_local_heads(
            matrix, target, base, maturity, dates, currencies, hard, quality
        )
        raw[h] = {
            "target": target, "baseline": base, "candidate": candidate,
            "local": local, "active": active, "n_train": n_train,
            "maturity": maturity,
        }
        logs_all.extend({"h": h, **row} for row in logs)
        for period, years in (("screen_2023", (2023,)),
                              ("validation_2024", (2024,))):
            metric_rows.extend(metric_grid(
                target, candidate, base, active, dates, currencies,
                h, period, years,
            ))
            bootstrap_rows_all.extend(bootstrap_rows(
                target, candidate, base, dates, h, period, years,
            ))
    metrics_frame = pd.DataFrame(metric_rows)
    bootstrap = pd.DataFrame(bootstrap_rows_all)
    gate_frame = gates(metrics_frame, bootstrap)
    open_horizons = [h for h in HORIZONS if gate_frame[gate_frame.h == h].stage_pass.all()]
    for h in open_horizons:
        for period, years in (("open_2025", (2025,)), ("open_2026", (2026,)),
                              ("open_2025_2026", (2025, 2026))):
            item = raw[h]
            metric_rows.extend(metric_grid(
                item["target"], item["candidate"], item["baseline"],
                item["active"], dates, currencies, h, period, years,
            ))
            bootstrap_rows_all.extend(bootstrap_rows(
                item["target"], item["candidate"], item["baseline"],
                dates, h, period, years,
            ))
    return {
        "dates": dates, "currencies": currencies, "panel": panel,
        "matrix": matrix, "hard": hard, "quality": quality, "raw": raw,
        "metrics": pd.DataFrame(metric_rows),
        "bootstrap": pd.DataFrame(bootstrap_rows_all), "gates": gate_frame,
        "open_horizons": open_horizons, "logs": pd.DataFrame(logs_all),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run_experiment()
    result["metrics"].to_csv(OUT / "metrics.csv", index=False)
    result["bootstrap"].to_csv(OUT / "paired_bootstrap.csv", index=False)
    result["gates"].to_csv(OUT / "historical_gates.csv", index=False)
    result["logs"].to_csv(OUT / "training_log.csv", index=False)
    arrays = {
        "dates": np.asarray([str(day) for day in result["dates"]]),
        "currencies": result["currencies"], "features": result["matrix"],
        "hard_quality": result["hard"], "quality": result["quality"],
    }
    for h, item in result["raw"].items():
        for name in ("target", "baseline", "candidate", "local", "active", "n_train"):
            arrays[f"{name}__h{h}"] = item[name]
        arrays[f"maturity__h{h}"] = np.asarray([str(day) for day in item["maturity"]])
    np.savez_compressed(OUT / "outputs.npz", **arrays)
    sources = (BASE, DIRECT, REGISTERED)
    (OUT / "metadata.json").write_text(json.dumps({
        "packet": "temperature-T46", "feature_names": list(FEATURE_NAMES),
        "minimum_train_rows": MIN_TRAIN_ROWS, "minimum_train_date": str(MIN_TRAIN_DATE),
        "embargo_days": EMBARGO_DAYS, "open_horizons": result["open_horizons"],
        "production_promoted": False, "tomorrow_cbr_used": False,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
    }, indent=2))
    print(json.dumps({
        "open_horizons": result["open_horizons"],
        "production_promoted": False,
    }, indent=2))
    print(result["gates"].to_string(index=False))


if __name__ == "__main__":
    main()
