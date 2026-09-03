"""Honest comparison of forecasting models and a GRU for target fav_h5.

All models only receive observations available at the forecast origin. Classical
model parameters are re-estimated at the start of each test year and their
state is then updated observation by observation. The GRU is retrained for each
year using only labels whose five-step horizon ends before that year.

Run:  .venv/bin/python run_time_series_models.py
"""
from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.data import CORRIDORS, REFERENCE, load
from ml.features import build_matrix
from ml.targets import build_targets

FIRST_TEST_YEAR = 2021
HORIZON = 5
LOOKBACK = 60
SEED = 42

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.filterwarnings("ignore", message="No frequency information was provided")


class GRUClassifier(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, 24, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(24), nn.Linear(24, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _state = self.gru(x)
        return self.head(out[:, -1]).squeeze(1)


def _score_forecast(forecast: np.ndarray, current: float) -> float:
    """Higher means all predicted future rates stay above today's rate."""
    return float(np.min(np.asarray(forecast, dtype=float)) - current)


def classical_scores(series, index) -> dict[str, np.ndarray]:
    row_of = {(c, i): r for r, (c, i, _d) in enumerate(index)}
    scores = {
        name: np.full(len(index), np.nan)
        for name in ("drift-20", "seasonal-naive-5", "ETS-5", "SARIMA-5", "SARIMA-20")
    }

    for currency in CORRIDORS:
        s = series[currency]
        z = np.log(s.values)
        print(f"  classical {currency}", flush=True)
        for year in range(FIRST_TEST_YEAR, max(d.year for d in s.dates) + 1):
            test = [i for i, d in enumerate(s.dates) if d.year == year and i + HORIZON < len(s)]
            if not test:
                continue
            start = test[0]
            train = z[:start]
            try:
                ets = ExponentialSmoothing(
                    train, trend=True, damped_trend=True, seasonal=5,
                    initialization_method="estimated",
                ).fit(disp=False, maxiter=100)
            except Exception:
                ets = None
            sarimas = {}
            for period in (5, 20):
                try:
                    sarimas[period] = SARIMAX(
                        train,
                        order=(1, 1, 1),
                        seasonal_order=(1, 0, 0, period),
                        trend="t",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=80)
                except Exception:
                    sarimas[period] = None

            for i in test:
                row = row_of.get((currency, i))
                if row is None:
                    continue
                current = z[i]
                recent = np.diff(z[max(0, i - 20): i + 1])
                drift = float(np.mean(recent)) if len(recent) else 0.0
                scores["drift-20"][row] = _score_forecast(
                    current + drift * np.arange(1, HORIZON + 1), current
                )
                seasonal = z[i - 4: i + 1] if i >= 4 else np.repeat(current, HORIZON)
                scores["seasonal-naive-5"][row] = _score_forecast(seasonal, current)

                if ets is not None:
                    try:
                        ets = ets.extend([current])
                        scores["ETS-5"][row] = _score_forecast(ets.forecast(HORIZON), current)
                    except Exception:
                        ets = None
                for period in (5, 20):
                    result = sarimas[period]
                    if result is None:
                        continue
                    try:
                        result = result.extend([current])
                        sarimas[period] = result
                        scores[f"SARIMA-{period}"][row] = _score_forecast(
                            result.forecast(HORIZON), current
                        )
                    except Exception:
                        sarimas[period] = None
    return scores


def sequence_features(series, currency: str, origin: int, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    dates = series[currency].dates
    own = np.log(series[currency].values)
    usd = np.log(series["USD"].values)
    cny = np.log(series["CNY"].values)
    paths = []
    for values in (own, usd, cny):
        returns = np.diff(values, prepend=values[0]) * 10000.0
        paths.append(returns[origin - LOOKBACK + 1: origin + 1])
    dynamic = (np.column_stack(paths) - mu) / sd
    date_features = np.array([
        [
            np.sin(2 * np.pi * (d.timetuple().tm_yday - 1) / (366 if d.year % 4 == 0 else 365)),
            np.cos(2 * np.pi * (d.timetuple().tm_yday - 1) / (366 if d.year % 4 == 0 else 365)),
            np.sin(2 * np.pi * d.weekday() / 7),
            np.cos(2 * np.pi * d.weekday() / 7),
        ]
        for d in dates[origin - LOOKBACK + 1: origin + 1]
    ], dtype=np.float32)
    one_hot = np.zeros((LOOKBACK, len(CORRIDORS)), dtype=np.float32)
    one_hot[:, CORRIDORS.index(currency)] = 1.0
    # Relative path lets the GRU see where today sits inside its recent range.
    relative_level = ((own[origin - LOOKBACK + 1: origin + 1] - own[origin]) * 100.0)[:, None]
    return np.column_stack([dynamic, relative_level, date_features, one_hot]).astype(np.float32)


def gru_scores(series, index, y: np.ndarray) -> np.ndarray:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    row_of = {(c, i): r for r, (c, i, _d) in enumerate(index)}
    out = np.full(len(index), np.nan)
    common_dates = series[CORRIDORS[0]].dates
    all_returns = {
        c: np.diff(np.log(series[c].values), prepend=np.log(series[c].values[0])) * 10000.0
        for c in CORRIDORS + REFERENCE
    }

    for year in range(FIRST_TEST_YEAR, max(d.year for d in common_dates) + 1):
        wall = dt.date(year, 1, 1)
        first_test = next((i for i, d in enumerate(common_dates) if d >= wall), len(common_dates))
        train_origins = range(LOOKBACK - 1, max(LOOKBACK - 1, first_test - HORIZON))
        raw_train = np.concatenate([
            np.column_stack([all_returns[c][:first_test], all_returns["USD"][:first_test], all_returns["CNY"][:first_test]])
            for c in CORRIDORS
        ])
        mu = raw_train.mean(axis=0)
        sd = raw_train.std(axis=0)
        sd[sd == 0] = 1.0

        X_train, y_train = [], []
        for currency in CORRIDORS:
            for i in train_origins:
                row = row_of.get((currency, i))
                if row is None or np.isnan(y[row]):
                    continue
                X_train.append(sequence_features(series, currency, i, mu, sd))
                y_train.append(y[row])
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        if len(y_train) < 500:
            continue

        network = GRUClassifier(X_train.shape[2])
        optimizer = torch.optim.AdamW(network.parameters(), lr=2e-3, weight_decay=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(SEED + year)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=256, shuffle=True, generator=generator,
        )
        network.train()
        for _epoch in range(30):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(network(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()

        test_rows, X_test = [], []
        for currency in CORRIDORS:
            for i, day in enumerate(series[currency].dates):
                if day.year != year or i + HORIZON >= len(series[currency]):
                    continue
                row = row_of.get((currency, i))
                if row is None or np.isnan(y[row]):
                    continue
                test_rows.append(row)
                X_test.append(sequence_features(series, currency, i, mu, sd))
        network.eval()
        with torch.no_grad():
            pred = torch.sigmoid(network(torch.from_numpy(np.asarray(X_test, dtype=np.float32)))).numpy()
        out[np.asarray(test_rows, dtype=int)] = pred
        print(f"  GRU {year}: train={len(y_train)}, test={len(test_rows)}", flush=True)
    return out


def report(scores: dict[str, np.ndarray], y: np.ndarray, index) -> None:
    dates = np.array([d for _c, _i, d in index], dtype=object)
    currencies = np.array([c for c, _i, _d in index], dtype=object)
    print("\nAUC: цель — текущий курс не будет побит в следующие 5 публикаций")
    print(f"{'модель':<22}{'ВСЕ':>8}" + "".join(f"{c:>8}" for c in CORRIDORS) + "  лет >0.5")
    for name, score in scores.items():
        valid = ~np.isnan(score) & ~np.isnan(y) & np.array([d.year >= FIRST_TEST_YEAR for d in dates])
        pooled = roc_auc_score(y[valid], score[valid])
        per_currency = []
        for currency in CORRIDORS:
            m = valid & (currencies == currency)
            per_currency.append(roc_auc_score(y[m], score[m]))
        positive_years = total_years = 0
        for year in sorted({d.year for d in dates[valid]}):
            m = valid & np.array([d.year == year for d in dates])
            if len(np.unique(y[m])) < 2:
                continue
            total_years += 1
            positive_years += roc_auc_score(y[m], score[m]) > 0.5
        print(f"{name:<22}{pooled:>8.3f}" + "".join(f"{v:>8.3f}" for v in per_currency)
              + f"  {positive_years}/{total_years}")


def main() -> None:
    series = load()
    X, names, index = build_matrix(series, CORRIDORS, REFERENCE)
    y = build_targets(series, index)[f"fav_h{HORIZON}"]
    scores = classical_scores(series, index)
    scores["GRU-classifier"] = gru_scores(series, index, y)
    scores["pct_range_90"] = X[:, names.index("pct_range_90")]
    report(scores, y, index)


if __name__ == "__main__":
    main()
