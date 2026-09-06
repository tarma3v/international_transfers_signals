"""T45: quality-gated direct-pair residual for 15:30 future CBR benefit."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import HORIZONS
from ml.validation import target_reach_dates
from research.round5_features import load_round5_features
from research.round6_uzbek_central_bank_models import _forward


OUT = Path("results/research/temperature/t45_direct_pair_benefit")
BASE = Path("results/research/temperature/t6_pre_receipt_benefit/outputs.npz")
PANEL = Path("results/research/round7/direct_pairs/direct_market_panel.csv")
REGISTERED = Path("research/temperature_t45_direct_pair_benefit_registered.md")
MIN_TRAIN_DATE = dt.date(2022, 2, 24)
EMBARGO_DAYS = 2
MIN_TRAIN_ROWS = 200
SEED = 20260907
BOOTSTRAP_DRAWS = 2000
BLOCKS = (20, 50)
DIRECT_COLUMNS = (
    "mean_basis", "last_basis", "previous_basis", "intraday_return",
    "range_bps", "count", "age_minutes", "quality", "tom_count",
    "tod_count",
)


def loadz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as source:
        return {key: source[key] for key in source.files}


def aligned_direct_panel(index) -> pd.DataFrame:
    panel = pd.read_csv(PANEL)
    panel["date"] = pd.to_datetime(panel.date).dt.date
    panel = panel.set_index(["currency", "date"]).reindex(
        [(currency, day) for currency, _position, day in index]
    ).reset_index()
    count = panel["count"].fillna(0.0).to_numpy(float)
    age = panel["age_minutes"].fillna(720.0).to_numpy(float)
    panel["hard_quality"] = (count >= 6.0) & (age <= 60.0)
    panel["quality"] = np.where(
        count > 0.0, np.minimum(count / 24.0, 1.0) * np.exp(-age / 120.0), 0.0
    )
    return panel


def direct_features(panel: pd.DataFrame, baseline, currencies):
    values = panel[list(DIRECT_COLUMNS)].to_numpy(float)
    values[:, :3] = np.clip(values[:, :3], -5000.0, 5000.0)
    values[:, 6] = np.clip(values[:, 6], 0.0, 720.0)
    values[~np.isfinite(values)] = 0.0
    one_hot = np.column_stack([
        np.asarray(currencies) == currency
        for currency in ("AMD", "KGS", "KZT", "TJS", "UZS")
    ]).astype(float)
    matrix = np.column_stack([np.asarray(baseline, float), values, one_hot])
    return matrix


def quarter_origins(dates):
    last_year = max(day.year for day in dates)
    return [
        dt.date(year, month, 1)
        for year in range(2023, last_year + 1)
        for month in (1, 4, 7, 10)
    ]


def next_quarter(origin: dt.date) -> dt.date:
    return (pd.Timestamp(origin).to_period("Q") + 1).start_time.date()


def fit_quarterly_residual(matrix, target, baseline, maturity, dates, hard, quality):
    matrix = np.asarray(matrix, float)
    target = np.asarray(target, float)
    baseline = np.asarray(baseline, float)
    maturity = np.asarray(maturity, object)
    dates = np.asarray(dates, object)
    hard = np.asarray(hard, bool)
    quality = np.asarray(quality, float)
    candidate = baseline.copy()
    residual_prediction = np.zeros(len(dates), dtype=float)
    n_train = np.zeros(len(dates), dtype=np.int32)
    logs = []
    for origin in quarter_origins(dates):
        end = next_quarter(origin)
        cutoff = origin - dt.timedelta(days=EMBARGO_DAYS)
        mature = np.asarray([value < cutoff for value in maturity], bool)
        train = (
            (dates >= MIN_TRAIN_DATE) & (dates < origin) & mature & hard
            & np.isfinite(target) & np.isfinite(baseline)
            & np.all(np.isfinite(matrix), axis=1)
        )
        query = (
            (dates >= origin) & (dates < end) & hard
            & np.isfinite(baseline) & np.all(np.isfinite(matrix), axis=1)
        )
        n_train[query] = int(train.sum())
        ids = np.flatnonzero(train)
        fitted = len(ids) >= MIN_TRAIN_ROWS
        if fitted:
            response = target[train] - baseline[train]
            lo, hi = np.quantile(response, [0.025, 0.975])
            clipped = np.clip(response, lo, hi)
            age_days = np.asarray([(origin - dates[i]).days for i in ids], float)
            weight = np.exp2(-age_days / 730.0) * quality[train]
            model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
            model.fit(matrix[train], clipped, ridge__sample_weight=weight)
            if np.any(query):
                correction = model.predict(matrix[query])
                residual_prediction[query] = correction
                candidate[query] = baseline[query] + quality[query] * correction
            coefficients = model.named_steps["ridge"].coef_.tolist()
            intercept = float(model.named_steps["ridge"].intercept_)
            last_maturity = str(max(maturity[train]))
        else:
            lo = hi = intercept = np.nan
            coefficients = []
            last_maturity = ""
        logs.append({
            "origin": str(origin), "n_train": int(len(ids)),
            "n_query": int(query.sum()), "model_fit": bool(fitted),
            "target_clip_lo": float(lo), "target_clip_hi": float(hi),
            "intercept": intercept, "coefficients": coefficients,
            "last_target_maturity": last_maturity,
        })
    return candidate, residual_prediction, n_train, logs


def metrics(target, candidate, baseline, mask):
    target = np.asarray(target, float)
    candidate = np.asarray(candidate, float)
    baseline = np.asarray(baseline, float)
    mask = (np.asarray(mask, bool) & np.isfinite(target)
            & np.isfinite(candidate) & np.isfinite(baseline))
    y, p, b = target[mask], candidate[mask], baseline[mask]
    if not len(y):
        return {key: np.nan for key in (
            "mae_candidate", "mae_baseline", "mae_delta", "rmse_candidate",
            "rmse_baseline", "rmse_delta", "bias_candidate", "bias_baseline",
            "spearman_candidate", "spearman_baseline", "spearman_delta",
        )} | {"n": 0}
    spear_p = spearmanr(y, p).statistic if len(y) > 2 else np.nan
    spear_b = spearmanr(y, b).statistic if len(y) > 2 else np.nan
    mae_p, mae_b = np.mean(np.abs(p - y)), np.mean(np.abs(b - y))
    rmse_p = np.sqrt(np.mean((p - y) ** 2))
    rmse_b = np.sqrt(np.mean((b - y) ** 2))
    return {
        "n": int(len(y)), "mae_candidate": float(mae_p),
        "mae_baseline": float(mae_b), "mae_delta": float(mae_p - mae_b),
        "rmse_candidate": float(rmse_p), "rmse_baseline": float(rmse_b),
        "rmse_delta": float(rmse_p - rmse_b),
        "bias_candidate": float(np.mean(p - y)),
        "bias_baseline": float(np.mean(b - y)),
        "spearman_candidate": float(spear_p), "spearman_baseline": float(spear_b),
        "spearman_delta": float(spear_p - spear_b),
    }


def metric_grid(target, candidate, baseline, dates, currencies, hard, h, period, years):
    dates = np.asarray(dates, object)
    currencies = np.asarray(currencies)
    scope = np.asarray([day.year in years for day in dates], bool)
    rows = [
        {"h": h, "period": period, "slice": "all", "group": "ALL",
         **metrics(target, candidate, baseline, scope)},
        {"h": h, "period": period, "slice": "hard_direct", "group": "ALL",
         **metrics(target, candidate, baseline, scope & hard)},
    ]
    for currency in ("AMD", "KGS", "KZT", "TJS", "UZS"):
        local = scope & (currencies == currency)
        rows.extend([
            {"h": h, "period": period, "slice": "currency_all",
             "group": currency,
             **metrics(target, candidate, baseline, local)},
            {"h": h, "period": period, "slice": "currency_hard",
             "group": currency,
             **metrics(target, candidate, baseline, local & hard)},
        ])
    return rows


def bootstrap_rows(target, candidate, baseline, dates, h, period, years):
    frame = pd.DataFrame({
        "date": dates, "target": target, "candidate": candidate,
        "baseline": baseline,
    })
    frame = frame[
        frame.date.map(lambda day: day.year in years)
        & np.isfinite(frame.target) & np.isfinite(frame.candidate)
        & np.isfinite(frame.baseline)
    ].copy()
    frame["delta"] = ((frame.candidate - frame.target).abs()
                      - (frame.baseline - frame.target).abs())
    daily = frame.groupby("date", sort=True).delta.mean().to_numpy(float)
    output = []
    for block in BLOCKS:
        rng = np.random.default_rng(SEED + h * 1000 + block + min(years))
        n = len(daily)
        n_blocks = int(np.ceil(n / block))
        starts = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n_blocks))
        offsets = np.arange(block)
        ids = (starts[:, :, None] + offsets[None, None, :]) % n
        draws = daily[ids.reshape(BOOTSTRAP_DRAWS, -1)[:, :n]].mean(axis=1)
        output.append({
            "h": h, "period": period, "block_dates": block,
            "n_dates": n, "point_mae_delta": float(daily.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "p_improvement": float(np.mean(draws < 0.0)),
        })
    return output


def gate_rows(metric_frame: pd.DataFrame, bootstrap: pd.DataFrame):
    output = []
    for period in ("screen_2023", "validation_2024"):
        for h in HORIZONS:
            part = metric_frame[(metric_frame.period == period) & (metric_frame.h == h)]
            pooled = part[(part.slice == "all") & (part.group == "ALL")].iloc[0]
            hard = part[(part.slice == "hard_direct") & (part.group == "ALL")].iloc[0]
            dense = part[(part.slice == "currency_hard") & part.group.isin(["KZT", "AMD"])]
            boot = bootstrap[(bootstrap.period == period) & (bootstrap.h == h)]
            checks = {
                "all_mae_better": pooled.mae_delta < 0.0,
                "all_rmse_noninferior": pooled.rmse_delta <= 0.0,
                "all_spearman_noninferior": pooled.spearman_delta >= -0.01,
                "hard_mae_better": hard.mae_delta < 0.0,
                "dense_currency_supported": (
                    len(dense) == 2 and dense.n.gt(0).all()
                    and dense.mae_delta.le(2.0).all()
                    and dense.mae_delta.lt(0.0).any()),
                "bootstrap_upper_nonpositive": (
                    len(boot) == len(BLOCKS) and boot.ci_high.le(0.0).all()),
            }
            output.append({
                "period": period, "h": h, **checks,
                "stage_pass": bool(all(checks.values())),
            })
    return pd.DataFrame(output)


def run_experiment():
    _matrix, _names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], object)
    currencies = np.asarray([row[0] for row in index])
    panel = aligned_direct_panel(index)
    hard = panel.hard_quality.to_numpy(bool)
    quality = panel.quality.to_numpy(float)
    base = loadz(BASE)
    raw, logs_by_h, metric_rows, boot_rows = {}, {}, [], []
    targets = {h: _forward(series, index, h) for h in HORIZONS}
    for h in HORIZONS:
        baseline = base[f"expected_bps__cutoff_1530__h{h}"].astype(float)
        maturity = np.asarray(target_reach_dates(index, series, h), object)
        matrix = direct_features(panel, baseline, currencies)
        candidate, correction, n_train, logs = fit_quarterly_residual(
            matrix, targets[h], baseline, maturity, dates, hard, quality)
        raw[h] = {
            "baseline": baseline, "candidate": candidate,
            "correction": correction, "n_train": n_train,
            "matrix": matrix, "maturity": maturity,
        }
        logs_by_h[h] = logs
        for period, years in (("screen_2023", (2023,)),
                              ("validation_2024", (2024,))):
            metric_rows.extend(metric_grid(
                targets[h], candidate, baseline, dates, currencies, hard,
                h, period, years))
            boot_rows.extend(bootstrap_rows(
                targets[h], candidate, baseline, dates, h, period, years))
    metric_frame = pd.DataFrame(metric_rows)
    bootstrap = pd.DataFrame(boot_rows)
    gates = gate_rows(metric_frame, bootstrap)
    open_horizons = [
        h for h in HORIZONS
        if gates[gates.h.eq(h)].stage_pass.all()
    ]
    for h in open_horizons:
        for period, years in (
            ("open_2025", (2025,)),
            ("open_2026", (2026,)),
            ("open_2025_2026", (2025, 2026)),
        ):
            metric_rows.extend(metric_grid(
                targets[h], raw[h]["candidate"], raw[h]["baseline"], dates,
                currencies, hard, h, period, years))
            boot_rows.extend(bootstrap_rows(
                targets[h], raw[h]["candidate"], raw[h]["baseline"], dates,
                h, period, years))
    bootstrap = pd.DataFrame(boot_rows)
    predictions = []
    for h in HORIZONS:
        keep = np.asarray([
            (day.year <= 2024) or (h in open_horizons) for day in dates
        ], bool)
        predictions.append(pd.DataFrame({
            "date": dates, "currency": currencies, "h": h,
            "baseline_expected_bps": raw[h]["baseline"],
            "candidate_expected_bps": np.where(
                keep, raw[h]["candidate"], np.nan),
            "hard_quality": hard, "quality": quality,
            "eligible": hard & np.isfinite(raw[h]["baseline"]),
            "horizon_open": h in open_horizons,
        }))
    logs = pd.concat([
        pd.DataFrame(items).assign(h=h) for h, items in logs_by_h.items()
    ], ignore_index=True)
    return {
        "dates": dates, "currencies": currencies, "panel": panel,
        "hard": hard, "quality": quality, "targets": targets, "raw": raw,
        "metrics": pd.DataFrame(metric_rows), "bootstrap": bootstrap,
        "gates": gates, "open_horizons": open_horizons,
        "predictions": pd.concat(predictions, ignore_index=True), "logs": logs,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run_experiment()
    result["metrics"].to_csv(OUT / "metrics.csv", index=False)
    result["bootstrap"].to_csv(OUT / "paired_bootstrap.csv", index=False)
    result["gates"].to_csv(OUT / "historical_gates.csv", index=False)
    result["logs"].to_csv(OUT / "training_log.csv", index=False)
    result["predictions"].to_csv(
        OUT / "publication_predictions.csv.gz", index=False,
        compression="gzip", float_format="%.15g")
    sources = [
        BASE, PANEL, REGISTERED,
        Path("research/temperature_t45_direct_pair_benefit.py"),
    ]
    metadata = {
        "packet": "temperature-T45",
        "baseline": "T6 cutoff_1530 future-only expected CBR bps",
        "horizons": list(HORIZONS),
        "open_horizons": result["open_horizons"],
        "screen_period": "2023", "validation_period": "2024",
        "open_period": "2025-2026 conditional only",
        "quality_gate": "count>=6 and age_minutes<=60",
        "soft_quality_formula": "min(count/24,1)*exp(-age_minutes/120)",
        "production_promoted": False, "push_changed": False,
        "probability_changed": False, "runtime_changed": False,
        "bank_execution_validated": False,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(result["gates"].to_string(index=False), flush=True)
    print("OPEN_HORIZONS", result["open_horizons"], flush=True)


if __name__ == "__main__":
    main()
