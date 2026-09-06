"""Calibration and delayed feedback use strictly mature, issued OOS forecasts."""
import datetime as dt

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from ml.data import CORRIDORS

MODES = ('expanding', 'rolling365', 'local365', 'isotonic', 'frozen2023')


def available_mask(dates, maturity, now, complete):
    return (dates < now) & (maturity < now - dt.timedelta(days=2)) & complete


def fit_map(x, y, method, fallback, min_rows=200):
    if len(x) < min_rows or len(np.unique(y)) < 2:
        return ('constant', float(fallback))
    if method == 'isotonic':
        return ('isotonic', IsotonicRegression(out_of_bounds='clip', y_min=0., y_max=1.).fit(x, y))
    mean, scale = float(np.mean(x)), max(float(np.std(x)), 1e-6)
    xx = (x - mean) / scale
    def objective(parameters):
        intercept, slope = parameters
        z = intercept + slope * xx
        loss = np.mean(np.logaddexp(0, z) - y * z) + slope ** 2 / (2 * len(x))
        error = expit(z) - y
        gradient = np.array([error.mean(), np.mean(error * xx) + slope / len(x)])
        return loss, gradient
    prior = (y.sum() + 1) / (len(y) + 2)
    result = minimize(objective, [np.log(prior / (1 - prior)), 1.], jac=True,
                      method='L-BFGS-B', bounds=[(None, None), (0., None)])
    if not result.success:
        return ('constant', float(prior))
    return ('logistic', mean, scale, result.x)


def apply_map(model, x):
    if model[0] == 'constant':
        return np.full(len(x), model[1])
    if model[0] == 'isotonic':
        return model[1].predict(x)
    _, mean, scale, parameters = model
    return expit(parameters[0] + parameters[1] * (x - mean) / scale)


def calibrate(panel, features, y, maturity):
    """features: rows x experts x horizons. Labels are read only through mature masks."""
    dates, currencies = panel.date.to_numpy(), panel.currency.to_numpy()
    n, experts, horizons = features.shape
    predictions = {mode: np.full_like(features, np.nan) for mode in MODES}
    complete = np.isfinite(y).all(axis=1)
    source = np.isfinite(features).all(axis=(1, 2))
    frozen = {}
    logs = []
    months = pd.period_range('2022-07', pd.Timestamp(max(dates)).to_period('M'), freq='M')
    for month in months:
        origin = month.start_time.date()
        end = (month + 1).start_time.date()
        te = (dates >= origin) & (dates < end) & source
        if not te.any():
            continue
        history = available_mask(dates, maturity, origin, complete)
        for mode in MODES:
            tr = history & source
            if mode in ('rolling365', 'local365'):
                tr &= dates >= origin - dt.timedelta(days=365)
            use_frozen = mode == 'frozen2023' and origin > dt.date(2023, 1, 1)
            fallback_counts = 0
            for e in range(experts):
                for j in range(horizons):
                    prior = (y[history, j].sum() + 1) / (history.sum() + 2)
                    if use_frozen:
                        model = frozen[e, j]
                    else:
                        model = fit_map(features[tr, e, j], y[tr, j], 'isotonic' if mode == 'isotonic' else 'logistic', prior)
                        if mode == 'frozen2023' and origin == dt.date(2023, 1, 1):
                            frozen[e, j] = model
                    fallback_counts += int(model[0] == 'constant')
                    predictions[mode][te, e, j] = apply_map(model, features[te, e, j])
                    if mode == 'local365':
                        for c in CORRIDORS:
                            local = tr & (currencies == c)
                            test = te & (currencies == c)
                            if local.sum() < 60 or not test.any():
                                continue
                            local_model = fit_map(features[local, e, j], y[local, j], 'logistic', prior, 60)
                            fraction = local.sum() / (local.sum() + 250.)
                            predictions[mode][test, e, j] = ((1 - fraction) * predictions[mode][test, e, j] +
                                fraction * apply_map(local_model, features[test, e, j]))
            predictions[mode][te] = np.minimum.accumulate(predictions[mode][te], axis=2)
            actual_origin = dt.date(2023, 1, 1) if use_frozen else origin
            logged = available_mask(dates, maturity, actual_origin, complete) & source
            if mode in ('rolling365', 'local365'):
                logged &= dates >= actual_origin - dt.timedelta(days=365)
            logs.append({'origin': str(origin), 'mode': mode, 'fit_origin': str(actual_origin),
                'n_calibration': int(logged.sum()), 'last_calibration_maturity': str(max(maturity[logged])) if logged.any() else None,
                'n_test': int(te.sum()), 'constant_maps': fallback_counts})
        print(f'AP5 calibration {origin}: {history.sum()} mature historical rows', flush=True)
    return predictions, logs


def exponential_weights(mean_loss, eta):
    logits = -eta * mean_loss
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return .9 * weights + .1 / len(weights)


def delayed_weights(panel, issued, y, maturity, half_life, eta, local=False, freeze=None):
    """Uses issued probabilities, never refits them; date batches prevent row-order leakage."""
    dates, currencies = panel.date.to_numpy(), panel.currency.to_numpy()
    n, experts, _ = issued.shape
    weights = np.full((n, experts), 1 / experts)
    complete = np.isfinite(y).all(axis=1) & np.isfinite(issued).all(axis=(1, 2))
    logs = []
    for day in sorted(set(dates)):
        idx = np.where(dates == day)[0]
        if not np.isfinite(issued[idx]).all():
            continue
        now = min(day, freeze) if freeze else day
        eligible = available_mask(dates, maturity, now, complete)
        if freeze and day < freeze:
            eligible[:] = False
        used = np.where(eligible)[0]
        # Only revealed y is accessed for losses. Future y cannot enter a weight.
        losses = ((issued[used] - y[used, None, :]) ** 2).mean(axis=2)
        ages = np.array([(now - (maturity[i] + dt.timedelta(days=3))).days for i in used])
        decay = 2. ** (-ages / half_life)
        global_loss = np.average(losses, weights=decay, axis=0) if len(used) else np.zeros(experts)
        for i in idx:
            mean_loss = global_loss.copy()
            own = currencies[used] == currencies[i]
            local_mass = float(decay[own].sum())
            if local and own.any():
                own_loss = np.average(losses[own], weights=decay[own], axis=0)
                fraction = local_mass / (local_mass + 50.)
                mean_loss = fraction * own_loss + (1 - fraction) * global_loss
            weights[i] = exponential_weights(mean_loss, eta)
            logs.append({'row': int(i), 'date': str(day), 'currency': currencies[i],
                'feedback_asof': str(now), 'n_revealed': len(used), 'local_effective_mass': local_mass,
                'last_revealed_maturity': str(max(maturity[eligible])) if eligible.any() else None,
                'last_revealed_prediction_date': str(max(dates[eligible])) if eligible.any() else None,
                **{f'loss_{e}': float(v) for e, v in enumerate(mean_loss)},
                **{f'weight_{e}': float(v) for e, v in enumerate(weights[i])}})
    return weights, logs
