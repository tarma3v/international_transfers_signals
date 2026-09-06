"""Monthly meta learning from issued OOS experts and strictly mature labels."""
import datetime as dt

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ml.data import CORRIDORS
from research.after_publication_ap1 import SEED
from research.after_publication_ap5_learning import available_mask

MODEL_NAMES = (
    'positive_exp', 'positive_730', 'logit_exp', 'logit_context_exp',
    'logit_context730', 'local_logit', 'shrunken_logit', 'hist_experts',
    'hist_context', 'hist_context730', 'hist_multi', 'pairwise_context',
    'equal_residual', 'local_residual', 'context_logit', 'context_hist',
)


def ranking_pairs(dates, currencies, y, days=60, max_negatives=8):
    """Receives mature train rows only; no test sample or outcome is accepted."""
    pairs = []
    for i in np.flatnonzero(y == 1):
        negatives = np.flatnonzero((currencies == currencies[i]) & (y == 0))
        distance = np.array([abs((dates[j] - dates[i]).days) for j in negatives])
        order = np.argsort(distance, kind='stable')
        nearest = negatives[order][distance[order] <= days][:max_negatives]
        pairs.extend((i, int(j)) for j in nearest)
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def fit_linear(X, y, positive=False, pairs=None):
    mean, scale = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-6)
    z = (X - mean) / scale
    pairwise = pairs is not None
    if pairwise:
        z = z[pairs[:, 0]] - z[pairs[:, 1]]
        target = np.ones(len(pairs))
        design = z
    else:
        target = y
        design = np.column_stack((np.ones(len(z)), z))
    penalty = np.ones(design.shape[1]) * 10. / len(design)
    if not pairwise:
        penalty[0] = 0.

    def objective(coef):
        score = design @ coef
        loss = np.mean(np.logaddexp(0., score) - target * score) + .5 * np.sum(penalty * coef ** 2)
        gradient = design.T @ (expit(score) - target) / len(design) + penalty * coef
        return loss, gradient

    bounds = [(0., None)] * design.shape[1] if positive else None
    if positive and not pairwise:
        bounds[0] = (None, None)
    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': 1500, 'ftol': 1e-11})
    if not result.success:
        raise RuntimeError(f'Meta linear optimizer failed: {result.message}')
    return {'mean': mean, 'scale': scale, 'coef': result.x, 'pairwise': pairwise}


def predict_linear(model, X):
    z = (X - model['mean']) / model['scale']
    if model['pairwise']:
        return z @ model['coef']
    return expit(model['coef'][0] + z @ model['coef'][1:])


def tree_model(regression=False):
    cls = HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier
    return cls(max_iter=100, learning_rate=.05, max_leaf_nodes=7, min_samples_leaf=40,
               l2_regularization=10., early_stopping=False, random_state=SEED)


def local_fraction(n):
    return n / (n + 250.)


def fit_stacks(panel, experts, context, source_context, equal, y, maturity,
               expert_names=None, context_names=None, source_names=None):
    dates, currencies = panel.date.to_numpy(), panel.currency.to_numpy()
    expert_names = expert_names or [f'expert_{j}' for j in range(experts.shape[1])]
    context_names = context_names or [f'context_{j}' for j in range(context.shape[1])]
    source_names = source_names or [f'source_{j}' for j in range(source_context.shape[1])]
    joint = np.column_stack((experts, context))
    joint_names = expert_names + context_names
    source = np.isfinite(joint).all(axis=1) & np.isfinite(source_context).all(axis=1) & np.isfinite(equal).all(axis=1)
    complete = np.isfinite(y).all(axis=1)
    scores = {k: np.full(len(panel), np.nan) for k in MODEL_NAMES}
    corrections = {k: np.zeros(len(panel)) for k in ('equal', 'local')}
    origins = np.full(len(panel), dt.date.max, dtype=object)
    logs, coefficients = [], []
    months = pd.period_range('2022-07', pd.Timestamp(max(dates)).to_period('M'), freq='M')

    def log_fit(key, origin, tr, te, fitted, pairs=None, currency='ALL'):
        logs.append({'model': key, 'origin': str(origin), 'currency': currency,
            'n_train': int(tr.sum()), 'n_test': int(te.sum()), 'fitted': bool(fitted),
            'last_train_date': str(max(dates[tr])) if tr.any() else None,
            'last_train_maturity': str(max(maturity[tr])) if tr.any() else None,
            'n_pairs': len(pairs) if pairs is not None else 0,
            'last_local_origin': str(max(origins[tr])) if key == 'local_residual' and tr.any() else None})

    def run_linear(key, origin, tr, te, X, names, fallback, positive=False, pairs=None, minimum=200, currency='ALL'):
        can_fit = tr.sum() >= minimum and np.unique(y[tr, 2]).size > 1 and (pairs is None or len(pairs) >= 100)
        pred = fallback.copy()
        if can_fit:
            model = fit_linear(X[tr], y[tr, 2], positive, pairs)
            pred = predict_linear(model, X[te])
            coef = model['coef'] if pairs is not None else model['coef'][1:]
            for j, name in enumerate(names):
                coefficients.append({'origin': str(origin), 'model': key, 'currency': currency,
                    'feature': name, 'coefficient': float(coef[j]), 'mean': float(model['mean'][j]),
                    'scale': float(model['scale'][j]), 'intercept': 0. if pairs is not None else float(model['coef'][0])})
        log_fit(key, origin, tr, te, can_fit, pairs, currency)
        return pred

    def run_tree(key, origin, tr, te, X, target, fallback, regression=False):
        can_fit = tr.sum() >= 200 and np.unique(target[tr]).size > 1
        pred = fallback.copy()
        if can_fit:
            model = tree_model(regression)
            model.fit(X[tr], target[tr])
            pred = model.predict(X[te]) if regression else model.predict_proba(X[te])[:, 1]
        log_fit(key, origin, tr, te, can_fit)
        return pred

    for month in months:
        origin, end = month.start_time.date(), (month + 1).start_time.date()
        te = (dates >= origin) & (dates < end) & source
        if not te.any():
            continue
        tr = available_mask(dates, maturity, origin, complete) & source
        recent = tr & (dates >= origin - dt.timedelta(days=730))
        fallback = equal[te, 2]
        for key, mask, X, names, positive in (
            ('positive_exp', tr, experts, expert_names, True),
            ('positive_730', recent, experts, expert_names, True),
            ('logit_exp', tr, experts, expert_names, False),
            ('logit_context_exp', tr, joint, joint_names, False),
            ('logit_context730', recent, joint, joint_names, False),
            ('context_logit', tr, source_context, source_names, False),
        ):
            scores[key][te] = run_linear(key, origin, mask, te, X, names, fallback, positive)
        for c in CORRIDORS:
            train, test = tr & (currencies == c), te & (currencies == c)
            if not test.any():
                continue
            local = run_linear('local_logit', origin, train, test, experts, expert_names,
                               scores['logit_exp'][test], minimum=60, currency=c)
            scores['local_logit'][test] = local
            fraction = local_fraction(train.sum()) if train.sum() >= 60 else 0.
            scores['shrunken_logit'][test] = fraction * local + (1 - fraction) * scores['logit_exp'][test]
            origins[test] = origin
        for key, mask, X in (
            ('hist_experts', tr, experts), ('hist_context', tr, joint),
            ('hist_context730', recent, joint), ('context_hist', tr, source_context),
        ):
            scores[key][te] = run_tree(key, origin, mask, te, X, y[:, 2], fallback)
        scores['hist_multi'][te] = run_tree('hist_multi', origin, tr, te, joint, y.mean(axis=1), equal[te].mean(axis=1), True)
        pairs = ranking_pairs(dates[tr], currencies[tr], y[tr, 2])
        scores['pairwise_context'][te] = run_linear('pairwise_context', origin, tr, te, joint,
                                                  joint_names, fallback, pairs=pairs)
        for kind, anchor in (('equal', equal[:, 2]), ('local', scores['local_logit'])):
            rr = tr & np.isfinite(anchor)
            if kind == 'local':
                assert (origins[rr] <= dates[rr]).all() and (origins[rr] < origin).all()
            # Only rr labels enter fitting, and the anchor is the issued historical score.
            target = np.full(len(panel), np.nan)
            target[rr] = y[rr, 2] - anchor[rr]
            correction = run_tree(kind + '_residual', origin, rr, te, joint, target, np.zeros(te.sum()), True)
            corrections[kind][te] = correction
            scores[kind + '_residual'][te] = np.clip(anchor[te] + correction, 0., 1.)
        print(f'AP6 {origin}: {tr.sum()} mature OOS rows, {len(pairs)} training pairs', flush=True)
    return scores, origins, logs, coefficients, corrections
